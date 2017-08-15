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


def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


class CausalBEGAN(object):
    '''
    A quick quirk about this class.
    if the model is built with a gpu, it must
    later be loaded with a gpu in order to preserve
    NCHW/NHCW-ness
    '''

    def __init__(self,batch_size,config):
        self.batch_size=batch_size #a tensor
        self.config=config

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')


        self.lambda_k = config.lambda_k
        self.lambda_l = config.lambda_l
        self.lambda_z = config.lambda_z
        self.gamma = config.gamma
        self.gamma_label = config.gamma_label
        self.zeta=config.zeta
        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path


        self.use_gpu = config.use_gpu
        _, height, width, self.channel = \
                get_conv_shape(data_loader['x'], self.data_format)
        self.repeat_num = int(np.log2(height)) - 2


        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step#Not used
        self.lr_update_step = config.lr_update_step
        self.is_train = config.is_train#used?

        #self.build_model()####multigpu stuff here


        #Should be like this
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=4)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        print('self.model_dir:',self.model_dir)
        gpu_options = tf.GPUOptions(allow_growth=True,
                                  #per_process_gpu_memory_fraction=0.333)
                                  per_process_gpu_memory_fraction=0.95)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)



        #Keeps track of params from different gpus
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

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Optimizer untested {}. Only tested Adam".format(config.optimizer))

        self.g_optimizer, self.d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)




    def __call__(self, real_inputs, fake_inputs):
        '''
        fake inputs is a dictionary of labels from cc
        real_inputs is also a dictionary of labels
            with an additional key 'x' for the real image
        '''
        #The keys of data_loader are all the labels union 'x'
        self.real_inputs=real_inputs
        self.fake_inputs=fake_inputs

        self.x = self.real_inputs.pop('x')
        x = norm_img(self.x)


        ##TOWER##
        self.real_labels=tf.concat(self.real_inputs.values(),-1)
        self.fake_labels=tf.concat(self.fake_inputs.values(),-1)

        self.z_gen = tf.random_uniform(
            (self.batch_size, self.z_num), minval=-1.0, maxval=1.0)

        #This guy is a dictionary of all possible z tensors
        #he has 1 for every causal label plus one called 'z_gen'
        #Use him to sample z and to feed z in

        #self.z_fd=self.cc.sample_z.copy()
        #self.z_fd.update({'z_gen':self.z_gen})

        #self.z= tf.concat( [self.fake_labels_logits, self.z_gen],axis=-1,name='z')
        if self.config.round_fake_labels:
            self.z= tf.concat( [tf.round(self.fake_labels), self.z_gen],axis=-1,name='z')
        else:
            self.z= tf.concat( [self.fake_labels, self.z_gen],axis=-1,name='z')

        G, self.G_var = GeneratorCNN(
                self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format)

        '''
        my approach was to just pretend 3 of the vars in the encoded space
        represented our causal variables
        z_num is only 64,I am using 3 for labels
        so if we use like 20 causal labels, make it larger

        TODO:actually I think it would have been more normal to pass labels through
        encoder and decoder. basically began but with (x,y) in place of x
        we can try this later (especially if there is label collapse)
        '''
        d_out, self.D_z, self.D_var = DiscriminatorCNN(
                tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format)
        AE_G, AE_x = tf.split(d_out, 2)

        self.D_encode_G, self.D_encode_x=tf.split(self.D_z, 2)#axis=0 by default

        if not self.config.separate_labeler:
            self.D_fake_labels_logits=tf.slice(self.D_encode_G,[0,0],[-1,n_labels])
            self.D_real_labels_logits=tf.slice(self.D_encode_x,[0,0],[-1,n_labels])
        else:

            self.D_fake_labels_logits,self.DL_var=Discriminator_labeler(
                G, len(self.cc), self.repeat_num,
                self.conv_hidden_num, self.data_format)

            self.D_real_labels_logits,  _        =Discriminator_labeler(
                x, len(self.cc), self.repeat_num,
                self.conv_hidden_num, self.data_format, reuse=True)

            #label_logits,self.DL_var=Discriminator_labeler(
            #        tf.concat([G, x], 0), len(self.cc.nodes), self.repeat_num,
            #        self.conv_hidden_num, self.data_format)
            #self.D_fake_labels_logits,self.D_real_labels_logits=tf.split(label_logits,2)

            self.D_var += self.DL_var


        self.D_real_labels=tf.sigmoid(self.D_real_labels_logits)
        self.D_fake_labels=tf.sigmoid(self.D_fake_labels_logits)
        self.D_real_labels_list=tf.split(self.D_real_labels,n_labels,axis=1)
        self.D_fake_labels_list=tf.split(self.D_fake_labels,n_labels,axis=1)


        #"sigmoid_cross_entropy_with_logits" is really long
        def sxe(logits,labels):
            #use zeros or ones if pass in scalar
            if not isinstance(labels,tf.Tensor):
                labels=labels*tf.ones_like(logits)
            return tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,labels=labels)



        #Should we round fake labels before calc loss?
        if self.config.round_fake_labels:
            fake_labels=tf.round(self.fake_labels)
        else:
            fake_labels=self.fake_labels

        self.d_xe_real_label=sxe(self.D_real_labels_logits,self.real_labels)
        self.d_xe_fake_label=sxe(self.D_fake_labels_logits,fake_labels)
        self.g_xe_label=sxe(self.fake_labels_logits, self.D_fake_labels)

        self.d_absdiff_real_label=tf.abs(self.D_real_labels  - self.real_labels)
        self.d_absdiff_fake_label=tf.abs(self.D_fake_labels  - fake_labels)
        self.g_absdiff_label=tf.abs(fake_labels  -  self.D_fake_labels)

        self.d_squarediff_real_label=tf.square(self.D_real_labels  - self.real_labels)
        self.d_squarediff_fake_label=tf.square(self.D_fake_labels  - fake_labels)
        self.g_squarediff_label=tf.square(fake_labels  -  self.D_fake_labels)

        if self.config.label_loss=='xe':
            self.d_loss_real_label = tf.reduce_mean(self.d_xe_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_xe_fake_label)
            self.g_loss_label=tf.reduce_mean(self.g_xe_label)
        elif self.config.label_loss=='absdiff':
            self.d_loss_real_label = tf.reduce_mean(self.d_absdiff_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_absdiff_fake_label)
            self.g_loss_label = tf.reduce_mean(self.g_absdiff_label)
        elif self.config.label_loss=='squarediff':
            self.d_loss_real_label = tf.reduce_mean(self.d_squarediff_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_squarediff_fake_label)
            self.g_loss_label = tf.reduce_mean(self.g_squarediff_label)

        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        u1=tf.abs(AE_x - x)
        u2=tf.abs(AE_G - G)
        m1=tf.reduce_mean(u1)
        m2=tf.reduce_mean(u2)
        c1=tf.reduce_mean(tf.square(u1-m1))
        c2=tf.reduce_mean(tf.square(u2-m2))
        self.eqn2 = tf.square(m1-m2)
        self.eqn1 = (c1+c2-2*tf.sqrt(c1*c2))/self.eqn2


        ##New label-margin loss:
        self.d_loss_real = tf.reduce_mean(u1)
        self.d_loss_fake = tf.reduce_mean(u2)
        self.g_loss_image = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss_image=self.d_loss_real       -   self.k_t*self.d_loss_fake
        self.d_loss_label=self.d_loss_real_label -   self.l_t*self.d_loss_fake_label
        self.d_loss=self.d_loss_image+self.d_loss_label

        if not self.config.no_third_margin:#normal mode
            #Careful on z_t sign!
            self.g_loss = self.g_loss_image + self.z_t*self.g_loss_label
        else:
            #can we get away without this complicated third margin?
            #No. rare label images will have poor quality
            print('Warning: not using third margin')
            self.g_loss = self.g_loss_image + 1.*self.g_loss_label

        #pretrain:
        #c_grad=self.c_optimizer.compute_gradients(self.c_loss,var_list=self.cc.train_var)
        #dcc_grad=self.dcc_optimizer.compute_gradients(self.dcc_loss,var_list=self.dcc_var)

        # Calculate the gradients for the batch of data,
        # on this particular gpu tower.

        g_grad=self.g_optimizer.compute_gradients(self.g_loss,var_list=self.G_var)
        d_grad=self.d_optimizer.compute_gradients(self.d_loss,var_list=self.D_var)

        #g_optim=self.g_optimizer.minimize(self.g_loss,var_list=self.G_var)
        #d_optim=self.d_optimizer.minimize(self.d_loss,var_list=self.D_var)


        #self.tower_dict['c_tower_grads'].append(c_grad)
        #self.tower_dict['dcc_tower_grads'].append(dcc_grad)
        self.tower_dict['g_tower_grads'].append(g_grad)
        self.tower_dict['d_tower_grads'].append(d_grad)
        self.tower_dict['tower_g_loss_image'].append(self.g_loss_image)
        self.tower_dict['tower_d_loss_real'].append(self.d_loss_real)
        self.tower_dict['tower_g_loss_label'].append(self.g_loss_label)
        self.tower_dict['tower_d_loss_real_label'].append(self.d_loss_real_label)
        self.tower_dict['tower_d_loss_fake_label'].append(self.d_loss_fake_label)

        self.var=self.G_var+self.D_var+[self.step]

        #return


##MODEL##


    def build_train_op(self):

        #iterate in rev order, makes self.data_loader corresp to gpu0 


        #with tf.variable_scope('tower'):
        #    for gpu,data_loader in self.data_by_gpu.items()[::-1]:
        #        gpu_idx+=1
        #        print('using device:',gpu)
        #        tower=gpu.replace('/','').replace(':','_')
        #        with tf.device(gpu),tf.name_scope(tower):

        #            #Build num_gpu copies of graph: inputs->gradient
        #            #Updates self.tower_dict
        #            self.build_tower(data_loader)

        #        #allow future gpu to use same variables
        #        tf.get_variable_scope().reuse_variables()

        #Now outside gpu loop

        d_loss_real       =tf.reduce_mean(self.tower_dict['tower_d_loss_real'])
        g_loss_image      =tf.reduce_mean(self.tower_dict['tower_g_loss_image'])
        d_loss_real_label =tf.reduce_mean(self.tower_dict['tower_d_loss_real_label'])
        d_loss_fake_label =tf.reduce_mean(self.tower_dict['tower_d_loss_fake_label'])
        g_loss_label      =tf.reduce_mean(self.tower_dict['tower_g_loss_label'])

        self.balance_k = self.gamma * d_loss_real - g_loss_image
        self.balance_l = self.gamma_label * d_loss_real_label - d_loss_fake_label
        self.balance_z = self.zeta*tf.nn.relu(self.balance_k) - tf.nn.relu(self.balance_l)


        self.measure = d_loss_real + tf.abs(self.balance_k)
        self.measure_complete = d_loss_real + d_loss_real_label + \
            tf.abs(self.balance_k)+tf.abs(self.balance_l)+tf.abs(self.balance_z)


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
        with tf.control_dependencies([k_update,l_update,z_update]):
            self.train_op=tf.group(g_optim, d_optim)


        ##*#* Interesting but pass this time around
        ## Track the moving averages of all trainable
        ## variables.
        #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        #variables_averages_op = variable_averages.apply(tf.trainable_variables())
        ## Group all updates to into a single
        ## train op.
        #train_op = tf.group(apply_gradient_op, variables_averages_op)


        #Move some of these summaries to CC
        #Label summaries
        names,real_labels_list=zip(*self.real_inputs.items())
        _    ,fake_labels_list=zip(*self.fake_inputs.items())
        LabelList=[names,real_labels_list,fake_labels_list,
                   self.D_fake_labels_list,self.D_real_labels_list]
        #for node,rlabel,d_fake_label,d_real_label in zip(*LabelList):
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

        tf.summary.scalar('losslabel/d_loss_real_label',tf.reduce_mean(self.d_loss_real_label))
        tf.summary.scalar('losslabel/d_loss_fake_label',tf.reduce_mean(self.d_loss_fake_label))
        tf.summary.scalar('losslabel/g_loss_label',self.g_loss_label)

        tf.summary.image("G", self.G),
        tf.summary.image("AE_G", self.AE_G),
        tf.summary.image("AE_x", self.AE_x),

        tf.summary.scalar("loss/d_loss", self.d_loss),
        tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
        tf.summary.scalar("loss/g_loss", self.g_loss),

        tf.summary.scalar("misc/d_lr", self.d_lr),
        tf.summary.scalar("misc/g_lr", self.g_lr),
        tf.summary.scalar("misc/eqn1", self.eqn1),
        tf.summary.scalar("misc/eqn2", self.eqn2),

        #summaries of gpu-averaged values
        tf.summary.scalar("loss/d_loss_real",d_loss_real),
        tf.summary.scalar("loss/g_loss_image", g_loss_image),
        tf.summary.scalar("balance/l", self.balance_l),
        tf.summary.scalar("balance/k", self.balance_k),
        tf.summary.scalar("balance/z", self.balance_z),
        tf.summary.scalar("misc/measure", self.measure),
        tf.summary.scalar("misc/measure_complete", self.measure_complete),
        tf.summary.scalar("misc/k_t", self.k_t),
        tf.summary.scalar("misc/l_t", self.l_t),
        tf.summary.scalar("misc/z_t", self.z_t),

        self.summary_op=tf.summary.merge_all()



