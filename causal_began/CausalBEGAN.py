from __future__ import print_function
from utils import save_image,distribute_input_data,summary_stats,make_summary
import pandas as pd
import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from figure_scripts.pairwise import crosstab
from figure_scripts.sample import intervention2d,condition2d

from utils import summary_stats
from models import *

class CausalBEGAN(object):
    '''
    A quick quirk about this class.
    if the model is built with a gpu, it must
    later be loaded with a gpu in order to preserve
    tensor structure: NCHW/NHCW (number-channel-height-width/number-height-channel-width)

    in paper <-> in code
    b1,c1    <-> b_k, k_t
    b2,c2    <-> b_l, l_t
    b3,c3    <-> b_z, z_t
    '''

    def __init__(self,batch_size,config):
        '''
        batch_size: again a tensorflow placeholder
        config    : see causal_began/config.py
        '''

        self.batch_size=batch_size #a tensor
        self.config=config
        self.use_gpu = config.use_gpu
        self.data_format=self.config.data_format#NHWC or NCHW
        self.TINY = 10**-6

        #number of calls to self.g_optim
        self.step = tf.Variable(0, name='step', trainable=False)

        #optimizers
        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        optimizer = tf.train.AdamOptimizer
        self.g_optimizer, self.d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.lambda_k = config.lambda_k
        self.lambda_l = config.lambda_l
        self.lambda_z = config.lambda_z
        self.gamma = config.gamma
        self.gamma_label = config.gamma_label
        self.zeta=config.zeta
        self.z_dim = config.z_dim
        self.conv_hidden_num = config.conv_hidden_num

        self.model_dir = config.model_dir

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.lr_update_step = config.lr_update_step
        self.is_train = config.is_train

        #Keeps track of params from different devices
        self.tower_dict=dict(
                    c_tower_grads=[],
                    dcc_tower_grads=[],
                    g_tower_grads=[],
                    d_tower_grads=[],
                    tower_g_loss_image=[],
                    tower_d_loss_real=[],
                    tower_g_loss_label=[],
                    tower_d_loss_real_label=[],
                    tower_d_loss_fake_label=[],
            )
        self.k_t = tf.get_variable(name='k_t',initializer=0.,trainable=False)
        self.l_t = tf.get_variable(name='l_t',initializer=0.,trainable=False)
        self.z_t = tf.get_variable(name='z_t',initializer=0.,trainable=False)

    def __call__(self, real_inputs, fake_inputs):
        '''
        in a multi gpu setting, self.__call__ is done once for every device with variables shared so
        that a copy of the tensorflow variables created in self.__call__ resides on
        each device. This would be run multiple times in a loop over devices.

        Parameters:
        fake inputs : a dictionary of labels from cc
        real_inputs : also a dictionary of labels
                      with an additional key 'x' for the real image
        '''
        config=self.config

        #The keys are all the labels union 'x'
        self.real_inputs=real_inputs
        self.fake_inputs=fake_inputs
        n_labels=len(fake_inputs)#number of labels in graph, not dataset

        #[0,255] NHWC
        self.x = self.real_inputs.pop('x')

        #used to change dataformat in data queue
        if self.data_format == 'NCHW':
            #self.x = tf.transpose(self.x, [2, 0, 1])#3D
            self.x = tf.transpose(self.x, [0, 3, 1, 2])#4D
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(self.data_format))

        _, height, width, self.channel = \
                get_conv_shape(self.x, self.data_format)
        self.config.repeat_num= int(np.log2(height)) - 2
        self.config.channel=self.channel

        #There are two versions: "x" and "self.x".
        #    "x" is normalized for computation
        #    "self.x" is unnormalized for saving and summaries
        #    likewise for "G" and "self.G"
        #x in [-1,1]
        x = norm_img(self.x)

        self.real_labels=tf.concat(self.real_inputs.values(),-1)
        self.fake_labels=tf.concat(self.fake_inputs.values(),-1)

        #noise given to generate image in addition to labels
        self.z_gen = tf.random_uniform(
            (self.batch_size, self.z_dim), minval=-1.0, maxval=1.0)

        if self.config.round_fake_labels:#default
            self.z= tf.concat( [tf.round(self.fake_labels), self.z_gen],axis=-1,name='z')
        else:
            self.z= tf.concat( [self.fake_labels, self.z_gen],axis=-1,name='z')

        G, self.G_var = GeneratorCNN(self.z,config)
        d_out, self.D_z, self.D_var = DiscriminatorCNN(tf.concat([G, x],0),config)
        AE_G, AE_x = tf.split(d_out, 2)
        self.D_encode_G, self.D_encode_x=tf.split(self.D_z, 2)#axis=0 by default

        if not self.config.separate_labeler:
            self.D_fake_labels_logits=tf.slice(self.D_encode_G,[0,0],[-1,n_labels])
            self.D_real_labels_logits=tf.slice(self.D_encode_x,[0,0],[-1,n_labels])
        else:#default
            self.D_fake_labels_logits,self.DL_var=Discriminator_labeler(G,n_labels,config)
            self.D_real_labels_logits,_=Discriminator_labeler(x,n_labels,config,reuse=True)
            self.D_var += self.DL_var

        self.D_real_labels=tf.sigmoid(self.D_real_labels_logits)
        self.D_fake_labels=tf.sigmoid(self.D_fake_labels_logits)
        self.D_real_labels_list=tf.split(self.D_real_labels,n_labels,axis=1)
        self.D_fake_labels_list=tf.split(self.D_fake_labels,n_labels,axis=1)

        # sigmoid_cross_entropy_with_logits
        def sxe(logits,labels):
            #use zeros or ones if pass in scalar
            if not isinstance(labels,tf.Tensor):
                labels=labels*tf.ones_like(logits)
            return tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,labels=labels)

        #Round fake labels before calc loss
        if self.config.round_fake_labels:#default
            fake_labels=tf.round(self.fake_labels)
        else:
            fake_labels=self.fake_labels

        #This is here because it's used in cross_entropy calc, but it's not used by default
        self.fake_labels_logits= -tf.log(1/(self.fake_labels+self.TINY)-1)

        #One of three label losses available
        # Default is squared loss, "squarediff"
        self.d_xe_real_label=sxe(self.D_real_labels_logits,self.real_labels)
        self.d_xe_fake_label=sxe(self.D_fake_labels_logits,fake_labels)
        self.g_xe_label=sxe(self.fake_labels_logits, self.D_fake_labels)

        self.d_absdiff_real_label=tf.abs(self.D_real_labels  - self.real_labels)
        self.d_absdiff_fake_label=tf.abs(self.D_fake_labels  - fake_labels)
        self.g_absdiff_label     =tf.abs(fake_labels  -  self.D_fake_labels)

        self.d_squarediff_real_label=tf.square(self.D_real_labels  - self.real_labels)
        self.d_squarediff_fake_label=tf.square(self.D_fake_labels  - fake_labels)
        self.g_squarediff_label     =tf.square(fake_labels  -  self.D_fake_labels)

        if self.config.label_loss=='xe':
            self.d_loss_real_label = tf.reduce_mean(self.d_xe_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_xe_fake_label)
            self.g_loss_label      = tf.reduce_mean(self.g_xe_label)
        elif self.config.label_loss=='absdiff':
            self.d_loss_real_label = tf.reduce_mean(self.d_absdiff_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_absdiff_fake_label)
            self.g_loss_label      = tf.reduce_mean(self.g_absdiff_label)
        elif self.config.label_loss=='squarediff':
            self.d_loss_real_label = tf.reduce_mean(self.d_squarediff_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_squarediff_fake_label)
            self.g_loss_label      = tf.reduce_mean(self.g_squarediff_label)

        #"self.G" is [0,255], "G" is [-1,1]
        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        u1=tf.abs(AE_x - x)
        u2=tf.abs(AE_G - G)
        m1=tf.reduce_mean(u1)
        m2=tf.reduce_mean(u2)
        c1=tf.reduce_mean(tf.square(u1-m1))
        c2=tf.reduce_mean(tf.square(u2-m2))
        self.eqn2 = tf.square(m1-m2)#from orig began paper
        self.eqn1 = (c1+c2-2*tf.sqrt(c1*c2))/self.eqn2#from orig began paper

        self.d_loss_real = tf.reduce_mean(u1)
        self.d_loss_fake = tf.reduce_mean(u2)
        self.g_loss_image = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss_image=self.d_loss_real       -   self.k_t*self.d_loss_fake
        self.d_loss_label=self.d_loss_real_label -   self.l_t*self.d_loss_fake_label
        self.d_loss=self.d_loss_image+self.d_loss_label

        if not self.config.no_third_margin:#normal mode
            #Careful on z_t sign!#(z_t <==> c_3 from paper)
            self.g_loss = self.g_loss_image + self.z_t*self.g_loss_label
        else:
            print('Warning: not using third margin')
            self.g_loss = self.g_loss_image + 1.*self.g_loss_label

        # Calculate the gradients for the batch of data,
        # on this particular gpu tower.
        g_grad=self.g_optimizer.compute_gradients(self.g_loss,var_list=self.G_var)
        d_grad=self.d_optimizer.compute_gradients(self.d_loss,var_list=self.D_var)

        self.tower_dict['g_tower_grads'].append(g_grad)
        self.tower_dict['d_tower_grads'].append(d_grad)
        self.tower_dict['tower_g_loss_image'].append(self.g_loss_image)
        self.tower_dict['tower_d_loss_real'].append(self.d_loss_real)
        self.tower_dict['tower_g_loss_label'].append(self.g_loss_label)
        self.tower_dict['tower_d_loss_real_label'].append(self.d_loss_real_label)
        self.tower_dict['tower_d_loss_fake_label'].append(self.d_loss_fake_label)

        self.var=self.G_var+self.D_var+[self.step]

    def build_train_op(self):
        #Now outside gpu loop

        #attributes starting with ave_ are averaged over devices
        self.ave_d_loss_real       =tf.reduce_mean(self.tower_dict['tower_d_loss_real'])
        self.ave_g_loss_image      =tf.reduce_mean(self.tower_dict['tower_g_loss_image'])
        self.ave_d_loss_real_label =tf.reduce_mean(self.tower_dict['tower_d_loss_real_label'])
        self.ave_d_loss_fake_label =tf.reduce_mean(self.tower_dict['tower_d_loss_fake_label'])
        self.ave_g_loss_label      =tf.reduce_mean(self.tower_dict['tower_g_loss_label'])

        #recalculate balance equations (b1,b2,b3 in paper)
        self.balance_k = self.gamma * self.ave_d_loss_real - self.ave_g_loss_image
        self.balance_l = self.gamma_label * self.ave_d_loss_real_label - self.ave_d_loss_fake_label
        self.balance_z = self.zeta*tf.nn.relu(self.balance_k) - tf.nn.relu(self.balance_l)

        self.measure = self.ave_d_loss_real + tf.abs(self.balance_k)
        self.measure_complete = self.ave_d_loss_real + self.ave_d_loss_real_label + \
            tf.abs(self.balance_k)+tf.abs(self.balance_l)+tf.abs(self.balance_z)

        #update margins coefficients (c1,c2,c3 in paper)
        k_update = tf.assign(
            self.k_t, tf.clip_by_value(self.k_t + self.lambda_k*self.balance_k, 0, 1))
        l_update = tf.assign(
            self.l_t, tf.clip_by_value(self.l_t + self.lambda_l*self.balance_l, 0, 1))
        z_update = tf.assign(
            self.z_t, tf.clip_by_value(self.z_t + self.lambda_z*self.balance_z, 0, 1))

        g_grads=average_gradients(self.tower_dict['g_tower_grads'])
        d_grads=average_gradients(self.tower_dict['d_tower_grads'])

        g_optim = self.g_optimizer.apply_gradients(g_grads, global_step=self.step)
        d_optim = self.d_optimizer.apply_gradients(d_grads)

        #every time train_op is run, run k_update, l_update, z_update
        with tf.control_dependencies([k_update,l_update,z_update]):
            #when train_op is run, run [g_optim,d_optim]
            self.train_op=tf.group(g_optim, d_optim)

    def train_step(self,sess,counter):
        sess.run(self.train_op)

        if counter % self.config.lr_update_step == self.lr_update_step - 1:
            sess.run([self.g_lr_update, self.d_lr_update])

    def build_summary_op(self):
        names,real_labels_list=zip(*self.real_inputs.items())
        _    ,fake_labels_list=zip(*self.fake_inputs.items())
        LabelList=[names,real_labels_list,fake_labels_list,
                   self.D_fake_labels_list,self.D_real_labels_list]
        for name,rlabel,flabel,d_fake_label,d_real_label in zip(*LabelList):
            with tf.name_scope(name):

                d_flabel=tf.cast(tf.round(d_fake_label),tf.int32)
                d_rlabel=tf.cast(tf.round(d_real_label),tf.int32)
                f_acc=tf.contrib.metrics.accuracy(tf.cast(tf.round(flabel),tf.int32),d_flabel)
                r_acc=tf.contrib.metrics.accuracy(tf.cast(tf.round(rlabel),tf.int32),d_rlabel)

                summary_stats('d_fake_label',d_fake_label,hist=True)
                summary_stats('d_real_label',d_real_label,hist=True)

                tf.summary.scalar('ave_d_fake_abs_diff',tf.reduce_mean(tf.abs(flabel-d_fake_label)))
                tf.summary.scalar('ave_d_real_abs_diff',tf.reduce_mean(tf.abs(rlabel-d_real_label)))

                tf.summary.scalar('real_label_ave',tf.reduce_mean(rlabel))
                tf.summary.scalar('real_label_accuracy',r_acc)
                tf.summary.scalar('fake_label_accuracy',f_acc)

        ##Summaries picked from last gpu to run
        tf.summary.scalar('losslabel/d_loss_real_label',tf.reduce_mean(self.ave_d_loss_real_label))
        tf.summary.scalar('losslabel/d_loss_fake_label',tf.reduce_mean(self.ave_d_loss_fake_label))
        tf.summary.scalar('losslabel/g_loss_label',self.g_loss_label)

        tf.summary.image("G", self.G),
        tf.summary.image("AE_G", self.AE_G),
        tf.summary.image("AE_x", self.AE_x),

        tf.summary.scalar("loss/d_loss", self.d_loss),
        tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
        tf.summary.scalar("loss/g_loss", self.g_loss),

        tf.summary.scalar("misc/d_lr", self.d_lr),
        tf.summary.scalar("misc/g_lr", self.g_lr),
        tf.summary.scalar("misc/eqn1", self.eqn1),#From orig BEGAN paper
        tf.summary.scalar("misc/eqn2", self.eqn2),#From orig BEGAN paper

        #summaries of gpu-averaged values
        tf.summary.scalar("loss/d_loss_real",self.ave_d_loss_real),
        tf.summary.scalar("loss/g_loss_image", self.ave_g_loss_image),
        tf.summary.scalar("balance/l", self.balance_l),
        tf.summary.scalar("balance/k", self.balance_k),
        tf.summary.scalar("balance/z", self.balance_z),
        tf.summary.scalar("misc/measure", self.measure),
        tf.summary.scalar("misc/measure_complete", self.measure_complete),
        tf.summary.scalar("misc/k_t", self.k_t),
        tf.summary.scalar("misc/l_t", self.l_t),
        tf.summary.scalar("misc/z_t", self.z_t),

        #doesn't include summaries from causal controller
        #TODO: rework so only 1 copy of summaries if multiple gpu
        self.summary_op=tf.summary.merge_all()

