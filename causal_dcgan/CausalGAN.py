from __future__ import division
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

from models import GeneratorCNN,DiscriminatorCNN
from utils import *
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


class CausalGAN(object):
    model_type='dcgan'

    def __init__(self,batch_size,config):

        self.batch_size = batch_size #a tensor
        self.config=config
        self.model_dir=config.model_dir

        self.TINY = 10**-8
        self.label_specific_noise=config.label_specific_noise

        self.step = tf.Variable(0, name='step', trainable=False)

        self.gamma_k = tf.get_variable(name='gamma_k',initializer=config.gamma_k,trainable=False)
        self.lambda_k = config.lambda_k#0.05
        self.gamma_l = gamma_l#self.label_loss_hyperparameter
        self.lambda_l = config.lambda_l#0.005
        self.gamma_m = 1./(self.gamma_k+self.TINY)#gamma_m#4.0 # allowing gan loss to be 8 times labelerR loss
        #self.gamma_m=config.gamma_m
        self.lambda_m =config.lambda_m#0.05

        self.pretrain_LabelerR = config.pretrain_LabelerR
        self.pretrain_LabelerR_no_of_epochs = config.pretrain_LabelerR_no_of_epochs
        self.fakeLabels_distribution = config.fakeLabels_distribution
        self.model_ID = model_ID

        self.label_type = config.label_type # discrete vs continuous

        self.k_t = tf.get_variable(name='k_t',initializer=1.,trainable=False) # kt is the closed loop feedback coefficient to balance the loss between LR and LG
        #gamma_k#1.0#tolerates up to labelerR = 2 labelerG

        #self.rec_loss_coeff = 0.0 #This wasn't doing anything.. discarding

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
        else:
            fake_labels=self.fake_labels#{0,1}



        if config.label_type=='discrete':
            fake_labels=0.5+fake_labels*0.2#{0.3,0.7}

        elif config.label_type=='continuous':

            if config.label_specific_noise:
                #TODO#uniform
                raise Exception('label_specific_noise=True not yet implemented')
                #OLD noise model that would need to be replicated here
                #          p = self.means[name]
                #          u = 0.5*(np.array(u)+1)
                #          if u == 1:
                #            u = 0.5 + 0.5*0.5*p+np.random.uniform(-0.25*p, 0.25*p, 1).astype(np.float32)
                #          elif u == 0:
                #            u = 0.5 - 0.5*(0.5-0.5*p)+np.random.uniform(-0.5*(0.5-0.5*p), 0.5*(0.5-0.5*p), 1).astype(np.float32)
                #          return u
                #        def p_independent_noise(u):
                #          return u
            else:#default
                fake_labels=0.5+fake_labels*0.2#{0.3,0.7}
                lower, upper, scale = 0, 0.2, 1/25.0
                lower_tail, upper_tail, scale_tail = 0, 0.3, 1/50.0
                #before #t = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
                #b*scale was the right-boundary
                b=(upper-lower)/scale
                b_tail=(upper_tail-lower_tail)/scale_tail

                s=tf_truncexpon(self.batch_size,rate=b,right=upper)
                s_tail=tf_truncexpon(self.batch_size,rate=b_tail,right=upper_tail)
                fake_labels = fake_labels + ((0.5-fake_labels)/0.2)*s + ((-0.5+fake_labels)/0.2)*s_tail
                #  COPY of OLD NOISE MODEL so murat can verify is equivalent
                #          u = 0.5+np.array(u)*0.2#ranges from 0.3 to 0.7
                #          lower, upper, scale = 0, 0.2, 1/25.0
                #          t = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
                #          s = t.rvs(1)#[0,0.2]
                #          lower_tail, upper_tail, scale_tail = 0, 0.3, 1/50.0
                #          t_tail = stats.truncexpon(b=(upper_tail-lower_tail)/scale_tail, loc=lower, scale=scale_tail)
                #          s_tail = t_tail.rvs(1)#[0,0.3]
                #          u = u + ((0.5-u)/0.2)*s + ((-0.5+u)/0.2)*s_tail

            tf.summary.histogram('noisy_fake_labels',fake_labels)



        self.fake_labels_logits= -tf.log(1/(self.fake_labels+self.TINY)-1)
        self.realLabels_logits = -tf.log(1/(self.realLabels+self.TINY)-1)

        if config.type_input_to_generator=='labels':
            self.fake_labels_inputs=fake_labels
        elif config.type_input_to_generator=='logits':
            self.fake_labels_inputs=self.fake_labels_logits

        ##FINISHED manipulating labels##


        #does this order matter?
        #self.z= tf.concat( [self.fake_labels_inputs, self.z_gen],axis=-1,name='z')
        self.z= tf.concat( [self.z_gen, self.fake_labels_inputs],axis=-1,name='z')


        G = GeneratorCNN(self.z)#[-1,1]float
        self.G=denorm_img(self.G)#[0,255]

        D_on_real=DiscriminatorCNN(x,self.config)
        D_on_fake=DiscriminatorCNN(G,self.config)

        self.D,x self.D_logits,self.features_to_estimate_z_on_input,self.D_var=D_on_real
        self.D_,self.D_logits_,self.features_to_estimate_z_on_generated,_     =D_on_fake

        #self.D, self.D_logits, self.features_to_estimate_z_on_input = self.discriminator(self.inputs, self.realLabels)


        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.dl_bn1 = batch_norm(name='dl_bn1')
        self.dl_bn2 = batch_norm(name='dl_bn2')
        self.dl_bn3 = batch_norm(name='dl_bn3')
#-------



    def sigmoid_cross_entropy_with_logits(x, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    def map_to_zero_one(x):
      return 0.5*(1+x)


    self.D, self.D_logits, self.features_to_estimate_z_on_input = self.discriminator(self.inputs, self.realLabels)
    self.D_labels_for_real, self.D_labels_for_real_logits = self.discriminator_labeler(inputs)

    #self.sampler = self.sampler(self.z_gen, self.zMale, self.zYoung, self.zSmiling)
    #self.sampler_label = self.sampler_label( self.zMale, self.zYoung, self.zSmiling)


    self.D_, self.D_logits_, self.features_to_estimate_z_on_generated = self.discriminator(self.G, self.fake_labels, reuse=True)
    self.D_labels_for_fake, self.D_labels_for_fake_logits = self.discriminator_labeler(self.G, reuse = True)
    
    self.D_gen_labels_for_fake, self.D_gen_labels_for_fake_logits = self.discriminator_gen_labeler(self.G)

    #self.D_on_z = self.discriminator_on_z(self.G)
    self.D_on_z = self.discriminator_on_z(self.features_to_estimate_z_on_generated)
    self.D_on_z_real = self.discriminator_on_z(self.features_to_estimate_z_on_input, reuse = True)
    self.z_for_real = concat([self.D_on_z_real,self.realLabels_logits], axis=1 , name ='z_real')
    self.inputs_reconstructed = self.generator(self.z_for_real, reuse = True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G, max_outputs = 10)


    #New
    if self.loss_function == 0:
        self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_labels_logits,self.D_labels_for_fake))
        self.g_lossGAN = tf.reduce_mean(
          -sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_))+sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    elif self.loss_function == 1:
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
        raise Exception('should not happen.\
                        self.loss_function=',self.loss_function)
    
    self.g_lossLabels_GLabeler = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_labels_logits,self.D_gen_labels_for_fake))
    self.g_lossLabels_GLabeler_sum = scalar_summary("g_loss_labelerG",self.g_lossLabels_GLabeler)
    
    self.g_loss_on_z = tf.reduce_mean(tf.abs(self.z_gen - self.D_on_z)**2)
    self.real_reconstruction_loss = tf.reduce_mean(tf.abs(self.inputs-self.inputs_reconstructed)**2)
    self.real_reconstruction_loss_sum = scalar_summary('real_reconstruction_loss', self.real_reconstruction_loss)
    self.rec_loss_coeff_sum = scalar_summary('rec_loss_coeff',self.rec_loss_coeff)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

    self.g_loss = self.g_lossGAN - 1.0*self.k_t*self.g_lossLabels_GLabeler + self.g_lossLabels + self.g_loss_on_z
    self.g_loss_without_labels = self.g_lossGAN
    self.g_loss_labels_sum = scalar_summary( 'g_loss_labelerR', self.g_lossLabels)
    self.g_lossGAN_sum = scalar_summary( 'g_lossGAN', self.g_lossGAN)
    #self.c_loss_sum = scalar_summary("c_loss", self.c_loss)
    self.g_loss_on_z_sum = scalar_summary('g_loss_on_z', self.g_loss_on_z)

    self.k_t_sum = scalar_summary( 'coeff_of_-LabelerG_loss, k_t', self.k_t)

    self.gamma_k_sum = scalar_summary('gamma_k_summary', self.gamma_k)

    self.d_labelLossReal = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_labels_for_real_logits, self.realLabels))    #self.d_labelLossFake = tf.reduce_mean(

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.d_loss_real_label_sum = scalar_summary("d_loss_real_label", self.d_labelLossReal)
    self.d_loss = self.d_loss_real + self.d_loss_fake 

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()
    self.dl_gen_vars = [var for var in t_vars if 'disc_gen_labeler' in var.name ]
    self.dl_vars = [var for var in t_vars if 'disc_labeler' in var.name ]
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name ]
    self.g_vars = [var for var in t_vars if 'generator' in var.name ]
    self.dz_vars = [var for var in t_vars if 'disc_z_labeler' in var.name]

    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1)

  def train(self, config):
    """Train DCGAN"""
    data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
    #np.random.shuffle(data)




    def builtd_train_op(self):
        d_gen_label_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_lossLabels_GLabeler, var_list=self.dl_gen_vars)
        d_label_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.d_labelLossReal, var_list=self.dl_vars)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_loss, var_list=self.g_vars)
        d_on_z_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_loss_on_z + self.rec_loss_coeff*self.real_reconstruction_loss, var_list=self.dz_vars)
        k_t_update = tf.assign(self.k_t, self.k_t*tf.exp(-1.0/3000.0) ) #tf.constant(0, dtype=tf.float32, name='zero'),tf.constant(1, dtype=tf.float32, name='one')) )

        self.summary_op=tf.summary.merge_all()


#---


    sample_files = data[0:self.sample_num]
    sample = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  is_crop=self.is_crop,
                  is_grayscale=self.is_grayscale) for sample_file in sample_files]
    if (self.is_grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)


    def clamp(x, lower, upper):
      return max(min(upper, x), lower)

    def p_dependent_noise(u,name):
      p = self.means[name]
      u = 0.5*(np.array(u)+1)
      if u == 1:
        u = 0.5 + 0.5*0.5*p+np.random.uniform(-0.25*p, 0.25*p, 1).astype(np.float32)
      elif u == 0:
        u = 0.5 - 0.5*(0.5-0.5*p)+np.random.uniform(-0.5*(0.5-0.5*p), 0.5*(0.5-0.5*p), 1).astype(np.float32)
      return u
    def p_independent_noise(u):
      u = 0.5+np.array(u)*0.2#ranges from 0.3 to 0.7
      lower, upper, scale = 0, 0.2, 1/25.0
      t = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
      s = t.rvs(1)
      lower_tail, upper_tail, scale_tail = 0, 0.3, 1/50.0
      t_tail = stats.truncexpon(b=(upper_tail-lower_tail)/scale_tail, loc=lower, scale=scale_tail)
      s_tail = t_tail.rvs(1)
      u = u + ((0.5-u)/0.2)*s + ((-0.5+u)/0.2)*s_tail
      return u

    def label_mapper(u,name, label_type):
      if label_type == 'discrete':
        return 0.5+np.array(u)*0.2
      elif label_type == 'continuous':
        if self.label_specific_noise:
          return p_dependent_noise(u,name)
        else:
          return p_independent_noise(u)
      else:
        raise Exception("label type is misspecified!")


    def make_summary(name, val):
      return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

    def label_sampler(self):
      nmbr = 100000
      batch_zMale = np.random.uniform(-1, 1, [nmbr, self.MaleDim]).astype(np.float32)
      batch_zYoung = np.random.uniform(-1, 1, [nmbr, self.YoungDim]).astype(np.float32)
      batch_zSmiling = np.random.uniform(-1, 1, [nmbr, self.SmilingDim]).astype(np.float32)
      fd= {self.zMale:    batch_zMale,
          self.zYoung:   batch_zYoung,
          self.zSmiling: batch_zSmiling}
      fake_labels_logits = self.sess.run(self.sampler_label, feed_dict=fd)
      #_, fake_labels_logits, _ = sess.run(self.causalController(batch_zMale,batch_zYoung,batch_zSmiling))
      x = np.sign(fake_labels_logits)
      print x
      y = pd.DataFrame(data=x,index = np.arange(x.shape[0]),columns = ['Male','Young','Smiling'])
      print y
      a = pd.crosstab(index=y['Male'], columns=[y['Young'],y['Smiling']])/y.shape[0]
      print a
      a.to_csv('Joint')

    counter = 1#1
    start_time = time.time()
    #name_list = self.cc.node_names
    name_list = [i[0] for i in self.graph]
    # Fixed noise vectors for visualization:
    fixed_noises = {}
    z_gen_fixed = {}
    name_id = 0
    for name in name_list:
        z_gen_fixed[name] = np.random.uniform(-1, 1, size=(self.batch_size, self.z_gen_dim))
        #z_gen_fixed[name] = np.tile(dum,[self.batch_size,1])
        sub = self.attributes[self.attributes[name] == 1]
        l = sub.shape[0]
        pointer = np.random.random_integers(0,l-33,1)[0]
        # print "l is: "+str(l) 
        # print "pointer is: "+ str(pointer)
        fake_labels_fixed = np.ones((self.batch_size,len(name_list)))
        name_id2 = 0
        for name2 in name_list:
            # print name2
            # print sub[name2].shape
            fake_labels_fixed[0:32,name_id2] = 0.5 + 0.2*sub[name2][pointer:pointer+32].values.reshape((32,))
            name_id2 = name_id2 + 1
        sub = self.attributes[self.attributes[name] == -1]
        l = sub.shape[0]
        pointer = np.random.random_integers(0,l-33,1)[0]
        # print "l is: "+str(l) 
        # print "pointer is: "+ str(pointer)
        name_id2 = 0
        for name2 in name_list:
            # print name2
            # print sub[name2].shape
            fake_labels_fixed[32:64,name_id2] = 0.5 + 0.2*sub[name2][pointer:pointer+32].values.reshape((32,))
            name_id2 = name_id2 + 1

        fixed_noises[name] = fake_labels_fixed
        name_id = name_id + 1
    print name_list
    for epoch in xrange(config.epoch):
      data = glob(os.path.join(
        "./data", config.dataset, self.input_fname_pattern))
      batch_idxs = min(len(data), config.train_size) // config.batch_size
      # last batch has 39 image, drop the last batch
      batch_idxs = batch_idxs-1
      random_shift = np.random.random_integers(3)-1 # 0,1,2

      for idx in xrange(0, batch_idxs):
        idx2 = np.random.random_integers(batch_idxs)
        batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_files_labels = data[idx2*config.batch_size:(idx2+1)*config.batch_size]
        fileNames = [i[-10:] for i in batch_files]
        fileNames_labels = [i[-10:] for i in batch_files_labels]
        realLabels = np.array([np.hstack(\
            tuple([label_mapper(self.attributes.loc[i].loc[label_name],label_name, self.label_type) for label_name in name_list])\
            )\
           for i in fileNames])
        if self.fakeLabels_distribution == "real_joint":
            fakeLabels = np.array([np.hstack(\
               tuple([label_mapper(self.attributes.loc[i].loc[label_name],label_name, self.label_type) for label_name in name_list])\
               )\
              for i in fileNames_labels])
        elif self.fakeLabels_distribution == "iid_uniform" and label_type == 'discrete':
            fakeLabels = 0.5+np.random.random_integers(-1,1,size=(self.batch_size,len(name_list)))*0.2
        elif self.fakeLabels_distribution == "iid_uniform" and label_type == 'continuous':
            fakeLabels = p_independent_noise(np.random.random_integers(-1,1,size=(self.batch_size,len(name_list))))
        else:
            raise Exception("Fake label distribution is not recognized")  
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      is_crop=self.is_crop,
                      is_grayscale=self.is_grayscale) for batch_file in batch_files]
        if (self.is_grayscale):
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        fd= {self.inputs: batch_images,
             self.realLabels:realLabels,
             self.fake_labels:fakeLabels
            }

        #if epoch < 1:
        #if counter < 5001:
        #if counter < 10001:
        if self.pretrain_LabelerR and epoch < self.pretrain_LabelerR_no_of_epochs:# counter < 15001:
          _, summary_str = self.sess.run([d_label_optim, self.summary_op], feed_dict=fd)
        # elif counter == 30: #1*3165+500:
        #   pairwise(self)
          ##if counter%1000==0:
          ##  crosstab(self,counter)#display results

        else:
          #if counter == 3000:
          #  _ = self.sess.run([k_t_update],feed_dict = fd)
          if np.mod(counter+random_shift, 3) == 0:
          #  _,summary_str = self.sess.run([d_optim,self.summary_op],feed_dict = fd)
        
          #else:
            #if np.mod(counter+random_shift,2)==0:
                #_ = self.sess.run([d_label_optim],feed_dict=fd)
            #elif np.mod(counter+random_shift,2)==1:
                #_ = self.sess.run([d_gen_label_optim],feed_dict=fd)
            _ = self.sess.run([g_optim], feed_dict=fd)
            _, _, _, _, summary_str = self.sess.run([d_label_optim,d_gen_label_optim,d_optim,d_on_z_optim, self.summary_op], feed_dict=fd)
            # if counter < 10000: # empirically, it never changes after first few thousand
            #     if np.abs(self.sess.run([self.gamma_k*self.g_lossLabels - self.g_lossLabels_GLabeler],feed_dict=fd)) < 0.005:
            #         _ = self.sess.run([gamma_k_update],feed_dict = fd)

            #_, _, _, _, summary_str = self.sess.run([d_label_optim, d_gen_label_optim, d_optim, g_optim, self.summary_op], feed_dict=fd)
            
            #self.writer.add_summary(summary_str, counter)
            #_, summary_str = self.sess.run([, self.summary_op], feed_dict=fd)
            #_, summary_str = self.sess.run([d_optim, self.summary_op],
            #  feed_dict={ self.inputs: batch_images, self.realLabels:realLabels, self.fakeLabels:fakeLabels, self.z: batch_z })
            #self.writer.add_summary(summary_str, counter)
            #self.writer.add_summary(make_summary('mygamma', self.gamma.eval(self.sess)),counter)          
            # Update G network
            #_, summary_str = self.sess.run([ , self.summary_op], feed_dict=fd)
            #self.writer.add_summary(summary_str, counter)
            #_, _ = self.sess.run([k_t_update,g_optim], feed_dict=fd)
            _, _ = self.sess.run([k_t_update,g_optim], feed_dict=fd)
            #_, _ = self.sess.run([k_t_update, l_t_update],feed_dict=fd)#,feed_dict = fd)
            #self.writer.add_summary(summary_str, counter)
          else:
            _, _ = self.sess.run([k_t_update, g_optim], feed_dict=fd)
            _, summary_str = self.sess.run([g_optim,self.summary_op], feed_dict=fd)
          self.writer.add_summary(summary_str, counter)

        #do this instead
        errD_fake,errD_real,errG= self.sess.run(
            [self.d_loss_fake,self.d_loss_real,self.g_loss], feed_dict=fd)

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, graph:%s, loss: %d, model_ID: %d" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG, self.graph_name,self.loss_function, self.model_ID))


        if np.mod(counter, 300) == 0:
          # for name in self.cc.node_names:
          #   do_dict={name:[0.9,-0.9]}
          #   do_dict_name=name
          #   intervention2d( self, fetch=self.G, do_dict=do_dict,do_dict_name=do_dict_name,step=counter)
          for name in name_list:
            images = self.sess.run(self.G, feed_dict={self.z_gen:z_gen_fixed[name], self.fake_labels:fixed_noises[name]})
            save_images(images, [8, 8], self.checkpoint_dir +'/train_images'+'/test_arange_%s%s.png' % (name,counter))
          self.save(self.checkpoint_dir, counter)
          #if np.mod(counter,900)==0:
          #  intervene_on_current_model()

  def discriminator(self, image, labels, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      # else:
      #   self.gamma = tf.Variable(1.0,name = 'd_gamma')  
      # following is from https://github.com/openai/improved-gan/blob/master/imagenet/discriminator.py#L88
      def add_minibatch_features(image,df_dim,batch_size):
        shape = image.get_shape().as_list()
        dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
        h_mb0 = lrelu(conv2d(image, df_dim, name='d_mb0_conv'))
        h_mb1 = conv2d(h_mb0, df_dim, name='d_mbh1_conv')
        image_ = tf.reshape(h_mb1, [batch_size, -1])
        n_kernels = 300
        dim_per_kernel = 50
        x = linear(image_, n_kernels * dim_per_kernel,'d_mbLinear')
        activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))
        big = np.zeros((batch_size, batch_size), dtype='float32')
        big += np.eye(batch_size)
        big = tf.expand_dims(big, 1)
        # the next step is very complicated. My best understanding is that 
        # the expanded dimension is automatically replicated to 64 to make subtraction possible
        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask

        #NO! Don't do this when you call self.discriminator twice
        #def half(tens, second):
        #  m, n, _ = tens.get_shape()
        #  m = int(m)
        #  n = int(n)
        #  return tf.slice(tens, [0, 0, second * batch_size], [m, n, batch_size])

        f1 = tf.reduce_sum(masked, 2) / tf.reduce_sum(mask)
        # sums over the third dimension, which is of size 64 (captures cross distance to other images in batch)
        #f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))


        #minibatch_features = tf.concat([f1, f2],1)
        mb_features = tf.reshape(f1, [batch_size, 1, 1, n_kernels])
        return conv_cond_concat(image, mb_features)
        #return tf.concat([image] + minibatch_features,1)

      #minibatch used to be applied here:
      #image_=add_minibatch_features(image, self.df_dim, self.batch_size)
      # shape = image.get_shape().as_list()
      # shape_ = image_.get_shape().as_list()
      # print shape
      # print shape_
      #image (16,64,64,3)
      #labels = tf.expand_dims(labels,1)
      #labels = tf.expand_dims(labels,1) # [64,1,1,8]
      #labels = tf.tile(labels,[1,64,64,1]) # [64,64,64,8]
      #h00 = tf.concat([image, labels],3) # labels added as new color channels
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))#16,32,32,64
      h1_ = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))#16,16,16,128
      h1 = add_minibatch_features(h1_, self.df_dim, self.batch_size)#now put minibatch here
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))#16,16,16,248
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h3_flat=tf.reshape(h3, [self.batch_size, -1])
      h4 = linear(h3_flat, 1, 'd_h3_lin')
      # D_labels_logits = linear(h3_flat, self.causal_labels_dim, 'd_h3_Label')
      # D_labels = tf.nn.sigmoid(D_labels_logits)
      return tf.nn.sigmoid(h4), h4, h1_ #, D_labels, D_labels_logits

  def discriminator_labeler(self, image, reuse=False):
    with tf.variable_scope("disc_labeler") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='dl_h0_conv'))#16,32,32,64
      h1 = lrelu(self.dl_bn1(conv2d(h0, self.df_dim*2, name='dl_h1_conv')))#16,16,16,128
      h2 = lrelu(self.dl_bn2(conv2d(h1, self.df_dim*4, name='dl_h2_conv')))#16,16,16,248
      h3 = lrelu(self.dl_bn3(conv2d(h2, self.df_dim*8, name='dl_h3_conv')))
      h3_flat=tf.reshape(h3, [self.batch_size, -1])
      D_labels_logits = linear(h3_flat, self.causal_labels_dim, 'dl_h3_Label')
      D_labels = tf.nn.sigmoid(D_labels_logits)
      return D_labels, D_labels_logits

  def discriminator_gen_labeler(self, image, reuse=False):
    with tf.variable_scope("disc_gen_labeler") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='dgl_h0_conv'))#16,32,32,64
      h1 = lrelu(self.dl_bn1(conv2d(h0, self.df_dim*2, name='dgl_h1_conv')))#16,16,16,128
      h2 = lrelu(self.dl_bn2(conv2d(h1, self.df_dim*4, name='dgl_h2_conv')))#16,16,16,248
      h3 = lrelu(self.dl_bn3(conv2d(h2, self.df_dim*8, name='dgl_h3_conv')))
      h3_flat=tf.reshape(h3, [self.batch_size, -1])
      D_labels_logits = linear(h3_flat, self.causal_labels_dim, 'dgl_h3_Label')
      D_labels = tf.nn.sigmoid(D_labels_logits)
      return D_labels, D_labels_logits

  def discriminator_on_z(self, image, reuse=False):
    with tf.variable_scope("disc_z_labeler") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='dzl_h0_conv'))#16,32,32,64
      h1 = lrelu(self.dl_bn1(conv2d(h0, self.df_dim*2, name='dzl_h1_conv')))#16,16,16,128
      h2 = lrelu(self.dl_bn2(conv2d(h1, self.df_dim*4, name='dzl_h2_conv')))#16,16,16,248
      h3 = lrelu(self.dl_bn3(conv2d(h2, self.df_dim*8, name='dzl_h3_conv')))
      h3_flat=tf.reshape(h3, [self.batch_size, -1])
      D_labels_logits = linear(h3_flat, self.z_gen_dim, 'dzl_h3_Label')
      D_labels = tf.nn.tanh(D_labels_logits)
      return D_labels

  def discriminator_CC(self, labels, reuse=False):
    with tf.variable_scope("disc_CC") as scope:
      if reuse:
        scope.reuse_variables()
      # add minibatch features here to get fake labels with high variation
      def add_minibatch_features_for_labels(labels,batch_size):
        n_kernels = 50
        dim_per_kernel = 20
        shape = labels.get_shape().as_list()
        dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
        input_ = tf.reshape(labels, [-1, dim])           # -1 means "all"  
        x = linear(input_, n_kernels * dim_per_kernel,'dCC_mbLabelLinear')
        activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))
        big = np.zeros((batch_size, batch_size), dtype='float32')
        big += np.eye(batch_size)
        big = tf.expand_dims(big, 1)

        # the next step is very complicated. My best understanding is that 
        # the expanded dimension is automatically replicated to 64 to make subtraction possible
        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask
        f1 = tf.reduce_sum(masked, 2) / tf.reduce_sum(mask)
        # sums over the third dimension, which is of size 64 (captures cross distance to other images in batch)
        #f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

        minibatch_features = tf.concat([labels, f1],1)
        return minibatch_features

      def lrelu(x,leak=0.2,name='lrelu'):
          with tf.variable_scope(name):
              f1=0.5 * (1+leak)#halves memory footprint
              f2=0.5 * (1-leak)#relative to tf.maximum(leak*x,x)
              return f1*x + f2*tf.abs(x)
      h0 = slim.fully_connected(labels,self.hidden_size,activation_fn=lrelu,scope='dCC_0')
      h1 = slim.fully_connected(h0,self.hidden_size,activation_fn=lrelu,scope='dCC_1')
      h1_aug = lrelu(add_minibatch_features_for_labels(h1,self.batch_size),name = 'disc_CC_lrelu')
      h2 = slim.fully_connected(h1_aug,self.hidden_size,activation_fn=lrelu,scope='dCC_2')
      h3 = slim.fully_connected(h2,1,activation_fn=None,scope='dCC_3')
      return tf.nn.sigmoid(h3),h3



#==============================================================================
#       yb = tf.reshape(Labels, [self.batch_size, 1, 1, self.realLabelsDim])
#       x = conv_cond_concat(image, yb)
# 
#       h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv')) # why this output dim?
#       h0 = conv_cond_concat(h0, yb)
# 
#       h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
#       h1 = tf.reshape(h1, [self.batch_size, -1])      
#       h1 = concat([h1, y], 1)
#       
#       h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
#       h2 = concat([h2, y], 1)
# 
#       h3 = linear(h2, 1, 'd_h3_lin')
#       
#       return tf.nn.sigmoid(h3), h3
#       
#==============================================================================


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir,model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    #checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
