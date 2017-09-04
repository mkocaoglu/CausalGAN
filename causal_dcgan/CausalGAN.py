from __future__ import division,print_function
from figure_scripts.pairwise import crosstab
from figure_scripts.sample import intervention2d
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pandas as pd
import sys
import scipy.stats as stats

from models import GeneratorCNN,DiscriminatorCNN,discriminator_labeler
from models import discriminator_gen_labeler,discriminator_on_z

from tensorflow.core.framework import summary_pb2
from tensorflow.contrib import slim

from ops import batch_norm,lrelu

from causal_graph import get_causal_graph

def norm_img(image):
    image = image/127.5 - 1.
    return image
def denorm_img(norm):
    return tf.clip_by_value((norm + 1)*127.5, 0, 255)

def tf_truncexpon(batch_size,rate,right):
    '''
    a tensorflow node that returns a random variable
    sampled from an Exp(rate) random variable
    which has been truncated and normalized to [0,right]

    #Leverages that log of uniform is exponential

    batch_size: a tensorflow placeholder to sync batch_size everywhere
    rate: lambda rate parameter for exponential dist
    right: float in (0,inf) where to truncate exp distribution
    '''

    uleft=tf.exp(-1*rate*right)
    U=tf.random_uniform(shape=(batch_size,1),minval=uleft,maxval=1)
    tExp=(-1/rate)*tf.log(U)

    return tExp

def add_texp_noise(batch_size,labels01):
    labels=0.3+labels01*0.4#{0.3,0.7}
    lower, upper, scale = 0, 0.2, 1/25.0
    lower_tail, upper_tail, scale_tail = 0, 0.3, 1/50.0
    #before #t = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
    #b*scale was the right-boundary
    b=(upper-lower)/scale
    b_tail=(upper_tail-lower_tail)/scale_tail

    s=tf_truncexpon(batch_size,rate=b,right=upper)
    s_tail=tf_truncexpon(batch_size,rate=b_tail,right=upper_tail)
    labels = labels + ((0.5-labels)/0.2)*s + ((-0.5+labels)/0.2)*s_tail
    return labels, [s,s_tail]

class CausalGAN(object):
    model_type='dcgan'

    def __init__(self,batch_size,config):

        self.batch_size = batch_size #a tensor
        self.config=config
        self.model_dir=config.model_dir
        self.TINY = 10**-6

        self.step = tf.Variable(0, name='step', trainable=False)
        self.inc_step=tf.assign(self.step,self.step+1)

        #########################################
        ##### Following is not used anymore #####
        #########################################
        self.gamma_k = tf.get_variable(name='gamma_k',initializer=config.gamma_k,trainable=False)
        self.lambda_k = config.lambda_k#0.05
        self.gamma_l = config.gamma_l#self.label_loss_hyperparameter
        self.lambda_l = config.lambda_l#0.005
        self.gamma_m = 1./(self.gamma_k+self.TINY)#gamma_m#4.0 # allowing gan loss to be 8 times labelerR loss
        #self.gamma_m=config.gamma_m
        self.lambda_m =config.lambda_m#0.05
        #########################################

        self.k_t = tf.get_variable(name='k_t',initializer=1.,trainable=False) # kt is the closed loop feedback coefficient to balance the loss between LR and LG

        self.rec_loss_coeff = 0.0
        print('WARNING:CausalGAN.rec_loss_coff=',self.rec_loss_coeff)

        self.hidden_size=config.critic_hidden_size

        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim

        self.loss_function = config.loss_function

    def __call__(self, real_inputs, fake_inputs):
        '''
        This builds the model on the inputs. Potentially this would be called
        multiple times in a multi-gpu situation. Put "setup" type stuff in
        __init__ instead.

        This is like self.build_model()

        fake inputs is a dictionary of labels from cc
        real_inputs is also a dictionary of labels
            with an additional key 'x' for the real image
        '''
        config=self.config#used many times

        #dictionaries
        self.real_inputs=real_inputs
        self.fake_inputs=fake_inputs

        n_labels=len(fake_inputs)
        self.x = self.real_inputs.pop('x')#[0,255]
        x = norm_img(self.x)#put in [-1,1]

        #These are 0,1 labels. To add noise, add noise from here.
        self.real_labels=tf.concat(self.real_inputs.values(),-1)
        self.fake_labels=tf.concat(self.fake_inputs.values(),-1)

        ##BEGIN manipulating labels##

        #Fake labels will already be nearly discrete
        if config.round_fake_labels: #default
            fake_labels=tf.round(self.fake_labels)#{0,1}
            real_labels=tf.round(self.real_labels)#should already be rounded
        else:
            fake_labels=self.fake_labels#{0,1}
            real_labels=self.real_labels

        if config.label_type=='discrete':
            fake_labels=0.3+fake_labels*0.4#{0.3,0.7}
            real_labels=0.3+real_labels*0.4#{0.3,0.7}

        elif config.label_type=='continuous':

            #this is so that they can be set to 0 in label_interpolation
            self.noise_variables=[]

            if config.label_specific_noise:
                #TODO#uniform see above #REFERENCE
                raise Exception('label_specific_noise=True not yet implemented')
            else:#default
                fake_labels,nvfake=add_texp_noise(self.batch_size,fake_labels)
                real_labels,nvreal=add_texp_noise(self.batch_size,real_labels)
                self.noise_variables.extend(nvfake)
                self.noise_variables.extend(nvreal)

            tf.summary.histogram('noisy_fake_labels',fake_labels)
            tf.summary.histogram('noisy_real_labels',real_labels)

        self.fake_labels_logits= -tf.log(1/(fake_labels+self.TINY)-1)
        self.real_labels_logits = -tf.log(1/(real_labels+self.TINY)-1)

        self.noisy_fake_labels=fake_labels
        self.noisy_real_labels=real_labels

        if config.type_input_to_generator=='labels':
            self.fake_labels_inputs=fake_labels
            self.real_labels_inputs=real_labels#for reconstruction
        elif config.type_input_to_generator=='logits': #default
            self.fake_labels_inputs=self.fake_labels_logits
            self.real_labels_inputs=self.real_labels_logits

        ##FINISHED manipulating labels##

        self.z_gen = tf.random_uniform( [self.batch_size, config.z_dim],minval=-1.0, maxval=1.0,name='z_gen')

        self.z= tf.concat( [self.z_gen, self.fake_labels_inputs],axis=-1,name='z')

        G, self.g_vars = GeneratorCNN(self.z,config)#[-1,1]float
        self.G=denorm_img(G)#[0,255]

        #Discriminator
        D_on_real=DiscriminatorCNN(x,config)
        D_on_fake=DiscriminatorCNN(G,config,reuse=True)
        self.D, self.D_logits ,self.features_to_estimate_z_on_input ,self.d_vars=D_on_real
        self.D_,self.D_logits_,self.features_to_estimate_z_on_generated,_ =D_on_fake

        #Discriminator Labeler
        self.D_labels_for_real, self.D_labels_for_real_logits, self.dl_vars =\
                discriminator_labeler(x,n_labels,config)
        self.D_labels_for_fake, self.D_labels_for_fake_logits, _ =\
                discriminator_labeler(G,n_labels,config,reuse=True)

        #Other discriminators
        self.D_gen_labels_for_fake,self.D_gen_labels_for_fake_logits,self.dl_gen_vars=\
            discriminator_gen_labeler(G,n_labels,config)
            #discriminator_gen_labeler(self.G,n_labels,config)

        self.D_on_z_real,_ =discriminator_on_z(self.features_to_estimate_z_on_input,config)
        self.D_on_z,self.dz_vars=discriminator_on_z(self.features_to_estimate_z_on_generated,config,reuse=True)

        #order of concat matters
        self.z_for_real = tf.concat([self.D_on_z_real,self.real_labels_inputs], axis=1 , name ='z_real')
        self.inputs_reconstructed,_ = GeneratorCNN(self.z_for_real,self.config, reuse = True)
        # Reconstructability is an idea that we tried. It does not provide big improvements, hence is not used ini the current version.

        tf.summary.histogram('d',self.D)
        tf.summary.histogram('d_',self.D_)
        tf.summary.image('G',self.G,max_outputs=10)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        # We tried different loss functions: 0,1,2 all have the order of terms in the cross entropy loss flipped, whereas 3,4,5 are not (consistent with theory).
        # Although all works to some extent, we have seen the sharpest images and best image quality with "loss function 1".
        # Difference between 0, 1, 2: This is to see the effect of using different GAN losses, as mentioned in the paper.
        if self.loss_function == 0:
            self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_labels_logits,self.D_labels_for_fake))
            self.g_lossGAN = tf.reduce_mean(
              -sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_))+sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        elif self.loss_function == 1:#default
            self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_labels_logits,self.D_labels_for_fake))
            self.g_lossGAN = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        elif self.loss_function == 2:
            self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_labels_logits,self.D_labels_for_fake))
            self.g_lossGAN = tf.reduce_mean(-sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        elif self.loss_function == 3:
            self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_labels_for_fake_logits, self.fake_labels))
            self.g_lossGAN = tf.reduce_mean(
              -sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_))+sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        elif self.loss_function == 4:
            self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_labels_for_fake_logits, self.fake_labels))
            self.g_lossGAN = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        elif self.loss_function == 5:
            self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_labels_for_fake_logits, self.fake_labels))
            self.g_lossGAN = tf.reduce_mean(-sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        else:
            raise Exception('Something is wrong with the loss function.\
                            self.loss_function=',self.loss_function)

        self.g_lossLabels_GLabeler = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_labels_logits,self.D_gen_labels_for_fake))
        tf.summary.scalar("g_loss_labelerG",self.g_lossLabels_GLabeler)

        self.g_loss_on_z = tf.reduce_mean(tf.abs(self.z_gen - self.D_on_z)**2)
        #x is the real input image
        self.real_reconstruction_loss = tf.reduce_mean(tf.abs(x-self.inputs_reconstructed)**2)

        tf.summary.scalar('real_reconstruction_loss', self.real_reconstruction_loss)

        self.d_loss_real = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        if config.reconstr_loss:
            g_loss_on_z=self.g_loss_on_z
        else:
            g_loss_on_z=0.
            # Default value for now, since reconstructability is not used in the current version.

        if config.off_label_losses:
            self.g_loss = self.g_lossGAN
        else:#default
            self.g_loss = self.g_lossGAN - 1.0*self.k_t*self.g_lossLabels_GLabeler + self.g_lossLabels + g_loss_on_z

        tf.summary.scalar('g_loss_labelerR', self.g_lossLabels)
        tf.summary.scalar('g_lossGAN', self.g_lossGAN)
        tf.summary.scalar('g_loss_on_z', self.g_loss_on_z)
        tf.summary.scalar('coeff_of_negLabelerG_loss_k_t', self.k_t)
        tf.summary.scalar('gamma_k_summary', self.gamma_k)

        self.d_labelLossReal = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_labels_for_real_logits,self.real_labels))

        tf.summary.scalar("d_loss_real", self.d_loss_real)
        tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        tf.summary.scalar("d_loss_real_label", self.d_labelLossReal)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("d_loss", self.d_loss)

    def build_train_op(self):
        config=self.config

        self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_loss, var_list=self.g_vars)

        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.d_loss, var_list=self.d_vars)

        self.d_label_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.d_labelLossReal, var_list=self.dl_vars)

        self.d_gen_label_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_lossLabels_GLabeler, var_list=self.dl_gen_vars)

        self.d_on_z_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_loss_on_z + self.rec_loss_coeff*self.real_reconstruction_loss, var_list=self.dz_vars)

        self.k_t_update = tf.assign(self.k_t, self.k_t*tf.exp(-1.0/config.tau) )

        self.train_op=tf.group(self.d_gen_label_optim,self.d_label_optim,self.d_optim,self.g_optim,self.d_on_z_optim)

    def build_summary_op(self):
        self.summary_op=tf.summary.merge_all()

    def train_step(self,sess,counter):
        '''
        This is a generic function that will be called by the Trainer class
        once per iteration. The simplest body for this part would be simply
        "sess.run(self.train_op)". But you may have more complications.

        Running self.summary_op is handeled by Trainer.Supervisor and doesn't
        need to be addressed here

        Only counters, not epochs are explicitly kept track of
        '''

        ###You can wait until counter>N to do stuff for example:
        if self.config.pretrain_LabelerR and counter < self.config.pretrain_LabelerR_no_of_iters:
            sess.run(self.d_label_optim)

        else:
            if np.mod(counter, 3) == 0:

                sess.run(self.g_optim)
                sess.run([self.train_op,self.k_t_update,self.inc_step])#all ops

            else:
                sess.run([self.g_optim, self.k_t_update ,self.inc_step])
                sess.run(self.g_optim)