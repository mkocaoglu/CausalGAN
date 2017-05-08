from __future__ import division
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

from ops import *
from utils import *
from tensorflow.core.framework import summary_pb2
from tensorflow.contrib import slim

from Causal_controller import CausalController
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


##Use this graph if no graph argument is passed in
#from causal_graph import standard_graph #This is the Male->Smiling<-Young
#from causal_graph import male_causes_beard

from causal_graph import get_causal_graph

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64, #z_dim=100,  
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,
         YoungDim = 10, MaleDim = 10, SmilingDim = 10, LabelDim = 10, hidden_size = 10,
               z_dim_Image=100, intervene_on = None,graph='big_causal_graph'):

    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.intervene_on = intervene_on
    self.graph=get_causal_graph(graph)

    self.y_dim = 3
    self.TINY = 10**-8
    #noise post causal controller
    self.z_gen_dim = 100  #100,10,10,10

########################################################################################
    self.LabelDim = LabelDim
    self.YoungDim = YoungDim
    self.MaleDim = MaleDim
    self.SmilingDim = SmilingDim
    self.hidden_size=hidden_size
########################################################################################        
    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')#why

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern

    self.attributes = pd.read_csv("./data/list_attr_celeba.txt",delim_whitespace=True)

    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    if self.is_crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs
    self.causal_labels_dim=len(self.graph)
    self.realLabels = tf.placeholder(tf.float32,[None, self.causal_labels_dim],
                                     name='causal_labels')

    self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    #Old
    #self.z_gen = tf.placeholder(tf.float32, [None, self.z_gen_dim], name='z_gen')#needed
    #New:
    self.z_gen = tf.random_uniform( [self.batch_size, self.z_gen_dim],minval=-1.0, maxval=1.0,name='z_gen')

    #CC (New)
    self.cc=CausalController(self.graph,self.batch_size)
    self.fake_labels= tf.concat( self.cc.list_labels(),-1 )
    self.fake_labels_logits= tf.concat( self.cc.list_label_logits(),-1 )

    #This part is to make it easy to sample all noise at once
    self.z_fd=self.cc.sample_z.copy()#a dictionary: {'Smiling:[0.2,.1,...]}
    self.z_fd.update({'z_gen':self.z_gen})

    #This is to match up with your notation:
    self.z_fake_labels=self.fake_labels_logits


    ##Old causal section:
    #self.zMale = tf.placeholder(tf.float32, [None, self.MaleDim], name='zMale')
    #self.zYoung = tf.placeholder(tf.float32, [None, self.YoungDim], name='zYoung')
    #self.zSmiling = tf.placeholder(tf.float32, [None, self.SmilingDim], name='zSmiling')

    #self.fake_labels, self.fake_labels_logits, self.z_fake_labels = self.causalController(self.zMale,self.zYoung,self.zSmiling)

    #self.zMaleLabel, self.zYoungLabel, self.zSmilingLabel = tf.split(self.fake_labels_logits, num_or_size_splits=3, axis=1)
    #self.zMaleLabel_avg = tf.reduce_mean(self.zMaleLabel)
    #self.zMaleLabel_std = tf.sqrt(tf.reduce_mean((self.zMaleLabel-self.zMaleLabel_avg)**2))
    #self.zYoungLabel_avg = tf.reduce_mean(self.zYoungLabel)
    #self.zYoungLabel_std = tf.sqrt(tf.reduce_mean((self.zYoungLabel-self.zYoungLabel_avg)**2))
    #self.zSmilingLabel_avg = tf.reduce_mean(self.zSmilingLabel)
    #self.zSmilingLabel_std = tf.sqrt(tf.reduce_mean((self.zSmilingLabel-self.zSmilingLabel_avg)**2))
    #self.zMaleLabel_sum = scalar_summary("zMaleLabel_avg", self.zMaleLabel_avg)
    #self.zYoungLabel_sum = scalar_summary("zYoungLabel_avg", self.zYoungLabel_avg)
    #self.zSmilingLabel_sum = scalar_summary("zSmilingLabel_avg", self.zSmilingLabel_avg)
    #self.zMaleLabel_std_sum = scalar_summary("zMaleLabel_std", self.zMaleLabel_std)
    #self.zYoungLabel_std_sum = scalar_summary("zYoungLabel_std", self.zYoungLabel_std)
    #self.zSmilingLabel_std_sum = scalar_summary("zSmilingLabel_std", self.zSmilingLabel_std)

    self.z= concat( [self.z_gen, self.z_fake_labels], axis=1 , name='z')

    def sigmoid_cross_entropy_with_logits(x, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    def map_to_zero_one(x):
      return 0.5*(1+x)

    self.G = self.generator(self.z)

    self.D, self.D_logits = self.discriminator(inputs)
    self.D_labels_for_real, self.D_labels_for_real_logits = self.discriminator_labeler(inputs)

    #self.sampler = self.sampler(self.z_gen, self.zMale, self.zYoung, self.zSmiling)
    #self.sampler_label = self.sampler_label( self.zMale, self.zYoung, self.zSmiling)

    #New hotfix
    self.sampler=self.G
    self.sampler_label=self.fake_labels_logits

    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    self.D_labels_for_fake, self.D_labels_for_fake_logits = self.discriminator_labeler(self.G, reuse = True)
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G, max_outputs = 10)


    ### DO: Fix the following to get label loss per label.
    # if not self.cc.node_names ==['Male','Young','Smiling']:
    #     raise Exception('This part needs changed for general graph')
    # else:

    #     self.h2Male=


    #     print 'Warning: Depricated Code is specific to Male->Smiling<-Young graph!'
    #     self.zMaleLabel_logits, self.zYoungLabel_logits, self.zSmilingLabel_logits = tf.split(self.fake_labels_logits, num_or_size_splits=3, axis=1)
    #     self.zMaleLabel_disc, self.zYoungLabel_disc, self.zSmilingLabel_disc = tf.split(self.D_labels_for_fake, num_or_size_splits=3, axis=1)
    #     self.realMaleLabel, self.realYoungLabel, self.realSmilingLabel = tf.split(self.realLabels, num_or_size_splits=3, axis=1)
    #     self.g_lossLabels_Male = tf.reduce_mean( sigmoid_cross_entropy_with_logits(self.zMaleLabel_logits, self.zMaleLabel_disc))
    #     self.g_lossLabels_Young = tf.reduce_mean( sigmoid_cross_entropy_with_logits(self.zYoungLabel_logits, self.zYoungLabel_disc))
    #     self.g_lossLabels_Smiling = tf.reduce_mean( sigmoid_cross_entropy_with_logits(self.zSmilingLabel_logits, self.zSmilingLabel_disc))
    #     #self.g_lossLabels = (self.g_lossLabels_Male + self.g_lossLabels_Young + self.g_lossLabels_Smiling)

    #New
    self.g_lossLabels= tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.fake_labels_logits,self.D_labels_for_fake))

    #self.g_lossLabels_Male_sum = scalar_summary("g_loss_label_male", self.g_lossLabels_Male)
    #self.g_lossLabels_Young_sum = scalar_summary("g_loss_label_young", self.g_lossLabels_Young)
    #self.g_lossLabels_Smiling_sum = scalar_summary("g_loss_label_smiling", self.g_lossLabels_Smiling)

    self.DCC_real, self.DCC_real_logits = self.discriminator_CC(tf.random_shuffle(self.realLabels)) # shuffle to avoid any correlated behavior with discriminator
    self.DCC_fake, self.DCC_fake_logits = self.discriminator_CC(self.fake_labels, reuse=True)

    self.dcc_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.DCC_real_logits, tf.ones_like(self.DCC_real)))
    self.dcc_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.DCC_fake_logits, tf.zeros_like(self.DCC_fake)))
    self.c_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.DCC_fake_logits,tf.ones_like(self.DCC_fake)))

    #This is a better way to do summary
    #you don't have to assign them to a variable and group them later
    #the / helps organize them in tensorboard
    ave_dcc_real=tf.reduce_mean(self.DCC_real)
    std_dcc_real=tf.sqrt(tf.reduce_mean(tf.square(ave_dcc_real-self.DCC_real)))
    ave_dcc_fake=tf.reduce_mean(self.DCC_fake)
    std_dcc_fake=tf.sqrt(tf.reduce_mean(tf.square(ave_dcc_fake-self.DCC_fake)))
    tf.summary.scalar('dcc/real_dcc_ave',ave_dcc_real)
    tf.summary.scalar('dcc/real_dcc_std',std_dcc_real)
    tf.summary.scalar('dcc/fake_dcc_ave',ave_dcc_fake)
    tf.summary.scalar('dcc/fake_dcc_std',std_dcc_fake)
    tf.summary.histogram('dcc/real_hist',self.DCC_real)
    tf.summary.histogram('dcc/fake_hist',self.DCC_fake)

    self.dcc_loss_real_sum = scalar_summary("dcc_loss_real", self.dcc_loss_real)
    self.dcc_loss_fake_sum = scalar_summary("dcc_loss_fake", self.dcc_loss_fake)
    self.dcc_loss = self.dcc_loss_real+self.dcc_loss_fake
    self.dcc_loss_sum = scalar_summary("dcc_loss", self.dcc_loss)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_lossGAN = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.g_loss = self.g_lossGAN + self.g_lossLabels#+ self.c_loss
    self.g_loss_labels_sum = scalar_summary( 'g_loss_label', self.g_lossLabels)
    self.g_lossGAN_sum = scalar_summary( 'g_lossGAN', self.g_lossGAN)
    self.c_loss_sum = scalar_summary("c_loss", self.c_loss)

    self.d_labelLossReal = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_labels_for_real_logits, self.realLabels))    #self.d_labelLossFake = tf.reduce_mean(

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.d_loss_real_label_sum = scalar_summary("d_loss_real_label", self.d_labelLossReal)
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.dl_vars = [var for var in t_vars if 'dl_' in var.name ]
    self.d_vars = [var for var in t_vars if 'd_' in var.name ]
    self.g_vars = [var for var in t_vars if 'g_' in var.name ]

    #NEW
    #self.c_vars = tf.get_collection(t_vars,scope='CC')
    #above didn't work
    self.c_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='CC')
    #make sure working:(delete later)
    print 'you have ',len(self.c_vars),' many causal weights'
    #OLD
    #self.c_vars = [var for var in t_vars if 'c_' in var.name ]


    self.dcc_vars = [var for var in t_vars if 'dCC_' in var.name ]
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1)


    #New: Causal summaries:
    #Label summaries
    self.real_labels_list = tf.unstack(self.realLabels ,axis=1)
    self.D_fake_labels_list= tf.unstack(self.D_labels_for_fake,axis=1)
    self.D_real_labels_list= tf.unstack(self.D_labels_for_real,axis=1)

    LabelList=[self.cc.nodes,self.real_labels_list,
               self.D_fake_labels_list,self.D_real_labels_list]
    for node,rlabel,d_fake_label,d_real_label in zip(*LabelList):
        with tf.name_scope(node.name):
            ##CC summaries:
            ave_label=tf.reduce_mean(node.label)
            std_label=tf.sqrt(tf.reduce_mean(tf.square(node.label-ave_label)))
            tf.summary.scalar('ave',ave_label)
            tf.summary.scalar('std',std_label)
            tf.summary.histogram('fake_label_hist',node.label)

            ##Disc summaries
            d_flabel=tf.cast(tf.round(d_fake_label),tf.int32)
            d_rlabel=tf.cast(tf.round(d_real_label),tf.int32)
            f_acc=tf.contrib.metrics.accuracy(tf.cast(tf.round(node.label),tf.int32),d_flabel)
            r_acc=tf.contrib.metrics.accuracy(tf.cast(tf.round(rlabel),tf.int32),d_rlabel)

            ave_d_fake_label=tf.reduce_mean(d_fake_label)
            std_d_fake_label=tf.sqrt(tf.reduce_mean(tf.square(d_fake_label-ave_d_fake_label)))

            ave_d_real_label=tf.reduce_mean(d_real_label)
            std_d_real_label=tf.sqrt(tf.reduce_mean(tf.square(d_real_label-ave_d_real_label)))

            tf.summary.scalar('ave_d_fake_abs_diff',tf.reduce_mean(tf.abs(node.label-d_fake_label)))
            tf.summary.scalar('ave_d_real_abs_diff',tf.reduce_mean(tf.abs(rlabel-d_real_label)))

            tf.summary.scalar('ave_d_fake_label',ave_d_fake_label)
            tf.summary.scalar('std_d_fake_label',std_d_fake_label)
            tf.summary.scalar('ave_d_real_label',ave_d_real_label)
            tf.summary.scalar('std_d_real_label',std_d_real_label)

            tf.summary.histogram('d_fake_label',d_fake_label)
            tf.summary.histogram('d_real_label',d_real_label)

            tf.summary.scalar('real_label_ave',tf.reduce_mean(rlabel))
            tf.summary.scalar('real_label_accuracy',r_acc)
            tf.summary.scalar('fake_label_accuracy',f_acc)


  def train(self, config):
    """Train DCGAN"""
    data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
    #np.random.shuffle(data)
    d_label_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_labelLossReal, var_list=self.dl_vars)
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.c_loss, var_list=self.c_vars)
    dcc_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.dcc_loss, var_list=self.dcc_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    #old
    #self.dcc_sum = merge_summary([self.dcc_loss_real_sum, self.dcc_loss_fake_sum, self.dcc_loss_sum])
    #self.c_sum = merge_summary([self.g_loss_sum, self.c_loss_sum, \
    #  self.zMaleLabel_sum, self.zYoungLabel_sum, self.zSmilingLabel_sum,\
    #  self.zMaleLabel_std_sum, self.zYoungLabel_std_sum, self.zSmilingLabel_std_sum,self.g_lossLabels_Male_sum,\
    #  self.g_lossLabels_Young_sum,self.g_lossLabels_Smiling_sum])
    #self.g_sum = merge_summary([self.G_sum, self.g_loss_sum, self.g_loss_labels_sum, self.g_lossGAN_sum,\
    #  self.g_lossLabels_Male_sum, self.g_lossLabels_Young_sum,self.g_lossLabels_Smiling_sum])
    #self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum])
    #self.dl_sum = merge_summary([self.d_loss_real_label_sum])

    #New
    self.summary_op=tf.summary.merge_all()


    self.writer = SummaryWriter(self.checkpoint_dir, self.sess.graph)
    #self.writer = SummaryWriter("./logs", self.sess.graph)

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

    #Old
    #sample_z= np.random.uniform(-1, 1, size=(self.sample_num, self.z_gen_dim))
    #sample_zMale= np.random.uniform(-1, 1,size=(self.sample_num,self.MaleDim)).astype(np.float32)
    #sample_zYoung= np.random.uniform(-1, 1,size=(self.sample_num,self.YoungDim))
    #sample_zSmiling= np.random.uniform(-1, 1,size=(self.sample_num,self.SmilingDim))

    #New. Draw fixed z to use during evaluation
    # DO: Fix the following
    # z_fixed = self.sess.run(self.z_fd)#string dict
    # sample_fd={self.z_fd[k]:val for k,val in z_fixed.items()}#tensor dict
    # sample_fd.update({self.inputs: sample_inputs})
         #    ,
         #self.z_gen:    sample_z,
         #self.zMale:    sample_zMale,
         #self.zYoung:   sample_zYoung,
         #self.zSmiling: sample_zSmiling}



    def clamp(x, lower, upper):
      return max(min(upper, x), lower)
    def label_mapper(u,s):
      # DO: Implement the following based on the marginal label statistics
      # if s=='male':
      #   p = 0.416754 #bias
      # elif s == 'young':
      #   p = 0.773617
      # elif s=='smiling':
      #   p = 0.482080

      # u = 0.5*(np.array(u)+1)

      # if u == 1:
      #   u = 0.5 + 0.5*0.5*p+np.random.uniform(-0.25*p, 0.25*p, 1).astype(np.float32)
      # elif u == 0:
      #   u = 0.5 - 0.5*(0.5-0.5*p)+np.random.uniform(-0.5*(0.5-0.5*p), 0.5*(0.5-0.5*p), 1).astype(np.float32)
      if u == 1:
        u = 0.5 + np.random.uniform(0, 0.25, 1).astype(np.float32)
      elif u == 0:
        u = 0.5 - np.random.uniform(0, 0.25, 1).astype(np.float32)
      return u
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

    counter = 1
    start_time = time.time()
    name_list = self.cc.node_names
    print name_list
    for epoch in xrange(config.epoch):
      data = glob(os.path.join(
        "./data", config.dataset, self.input_fname_pattern))
      batch_idxs = min(len(data), config.train_size) // config.batch_size
      random_shift = np.random.random_integers(3)-1 # 0,1,2

      for idx in xrange(0, batch_idxs):
        batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
        fileNames = [i[-10:] for i in batch_files]

        #realLabels = 2*np.array([(self.attributes.loc[i].loc['Male'],self.attributes.loc[i].loc['Young'],self.attributes.loc[i].loc['Smiling']) for i in fileNames])-1
        # realLabels = np.array([np.hstack(\
        #     (label_mapper(self.attributes.loc[i].loc['Male'], self.MaleDim,'male'),\
        #   label_mapper(self.attributes.loc[i].loc['Young'], self.YoungDim,'young'),\
        #   label_mapper(self.attributes.loc[i].loc['Smiling'],self.SmilingDim,'smiling'))\
        #     )\
        #    for i in fileNames])
        # DO: Simplify the following
        realLabels = np.array([np.hstack(\
            tuple([label_mapper(self.attributes.loc[i].loc[label_name],label_name) for label_name in name_list])\
            )\
           for i in fileNames])
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

        #batch_zMale = np.random.uniform(-1, 1, [config.batch_size, self.MaleDim]).astype(np.float32)
        #batch_zYoung = np.random.uniform(-1, 1, [config.batch_size, self.YoungDim]).astype(np.float32)
        #batch_zSmiling = np.random.uniform(-1, 1, [config.batch_size, self.SmilingDim]).astype(np.float32)
        #batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_gen_dim]).astype(np.float32)

        #New, z not needed
        fd= {self.inputs: batch_images,
             self.realLabels:realLabels,
             #self.z_gen:    batch_z,
             #self.zMale:    batch_zMale,
             #self.zYoung:   batch_zYoung,
             #self.zSmiling: batch_zSmiling
            }

        #New
        #I changed self.dl_sum, g_sum etc for just summary_op, which has all the summaries
        #If it takes appreciably longer, you only have to run summary_op
        # once every 50 iterations or so, not 6 times per iteration.
        if counter > 10:
          _, summary_str = self.sess.run([d_label_optim, self.summary_op], feed_dict=fd)
          self.writer.add_summary(summary_str, counter)
          #self.writer.add_summary(make_summary('mygamma', self.gamma.eval(self.sess)),counter)          
          _, summary_str = self.sess.run([dcc_optim, self.summary_op], feed_dict=fd)
          self.writer.add_summary(summary_str, counter)
          _, summary_str = self.sess.run([c_optim, self.summary_op], feed_dict=fd)
          self.writer.add_summary(summary_str, counter)
        elif counter == 1*3165+500:
          label_sampler(self)
        else:

          if np.mod(counter+random_shift, 3) == 0:

            _, summary_str = self.sess.run([d_label_optim, self.summary_op], feed_dict=fd)
            self.writer.add_summary(summary_str, counter)
            _, summary_str = self.sess.run([d_optim, self.summary_op], feed_dict=fd)
            #_, summary_str = self.sess.run([d_optim, self.summary_op],
            #  feed_dict={ self.inputs: batch_images, self.realLabels:realLabels, self.fakeLabels:fakeLabels, self.z: batch_z })
            self.writer.add_summary(summary_str, counter)
            #self.writer.add_summary(make_summary('mygamma', self.gamma.eval(self.sess)),counter)          
            # Update G network
            _, summary_str = self.sess.run([ g_optim, self.summary_op], feed_dict=fd)
            self.writer.add_summary(summary_str, counter)
            _, summary_str = self.sess.run([g_optim, self.summary_op], feed_dict=fd)
            self.writer.add_summary(summary_str, counter)
          else:
            _, summary_str = self.sess.run([g_optim, self.summary_op], feed_dict=fd)
            self.writer.add_summary(summary_str, counter)
            _, summary_str = self.sess.run([g_optim, self.summary_op], feed_dict=fd)
            self.writer.add_summary(summary_str, counter)

        #do this instead
        errD_fake,errD_real,errG= self.sess.run(
            [self.d_loss_fake,self.d_loss_real,self.g_loss], feed_dict=fd)

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))


        if np.mod(counter, 3000) == 0:
          self.save(config.checkpoint_dir, counter)
  def regularizer(self, m_logit, y_logit, s_logit):
    p1 = 0.023134
    p2 = 0.050301
    p3 = 0.244853
    p4 = 0.264957
    p5 = 0.087172
    p6 = 0.065775
    p7 = 0.162760
    p8 = 0.101047
    m = (tf.sign(m_logit)+1.0)*0.5
    y = (tf.sign(y_logit)+1.0)*0.5
    s = (tf.sign(s_logit)+1.0)*0.5
    #m = (m_logit>0).astype(np.float)
    #y = (y_logit>0).astype(np.float)
    #s = (s_logit>0).astype(np.float)
    total = (1.0/p1 + 1.0/p2 +1.0/p3 + 1.0/p4 + 1.0/p5 + 1.0/p6 + 1.0/p7 + 1.0/p8)
    r = ( (1-m)*(1-y)*(1-s)/p1 + (1-m)*(1-y)*(s)/p2 + (1-m)*(y)*(1-s)/p3 + (1-m)*(y)*(s)/p4 \
        + (m)*(1-y)*(1-s)/p5 +     (m)*(1-y)*(s)/p6 +  (m)*(y)*(1-s)/p7 + (m)*(y)*(s)/p8 )/total
    return r

  def regularizer_real(self, m_label, y_label, s_label):
    p1 = 0.023134
    p2 = 0.050301
    p3 = 0.244853
    p4 = 0.264957
    p5 = 0.087172
    p6 = 0.065775
    p7 = 0.162760
    p8 = 0.101047
    m = tf.cast(tf.round(m_label),tf.float32)
    y = tf.cast(tf.round(y_label),tf.float32)
    s = tf.cast(tf.round(s_label),tf.float32)
    #m = (m_logit>0).astype(np.float)
    #y = (y_logit>0).astype(np.float)
    #s = (s_logit>0).astype(np.float)
    total = (1.0/p1 + 1.0/p2 +1.0/p3 + 1.0/p4 + 1.0/p5 + 1.0/p6 + 1.0/p7 + 1.0/p8)
    r = ( (1-m)*(1-y)*(1-s)/p1 + (1-m)*(1-y)*(s)/p2 + (1-m)*(y)*(1-s)/p3 + (1-m)*(y)*(s)/p4 \
        + (m)*(1-y)*(1-s)/p5 +     (m)*(1-y)*(s)/p6 +  (m)*(y)*(1-s)/p7 + (m)*(y)*(s)/p8 )/total
    return r
  def discriminator(self, image, reuse=False):
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
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))#16,32,32,64
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))#16,16,16,128
      h1 = add_minibatch_features(h1, self.df_dim, self.batch_size)#now put minibatch here
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))#16,16,16,248
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h3_flat=tf.reshape(h3, [self.batch_size, -1])
      h4 = linear(h3_flat, 1, 'd_h3_lin')
      D_labels_logits = linear(h3_flat, self.causal_labels_dim, 'd_h3_Label')
      D_labels = tf.nn.sigmoid(D_labels_logits)
      return tf.nn.sigmoid(h4), h4 #, D_labels, D_labels_logits

  def discriminator_labeler(self, image, reuse=False):
    with tf.variable_scope("discriminator_labeler") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='dl_h0_conv'))#16,32,32,64
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='dl_h1_conv')))#16,16,16,128
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='dl_h2_conv')))#16,16,16,248
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='dl_h3_conv')))
      h3_flat=tf.reshape(h3, [self.batch_size, -1])
      D_labels_logits = linear(h3_flat, self.causal_labels_dim, 'dl_h3_Label')
      D_labels = tf.nn.sigmoid(D_labels_logits)
      return D_labels, D_labels_logits

  def independence_checker(self,fake_labels_logits, D_logits_, real_labels_logits, D_logits):
    #with tf.variable_scope("discriminator") as scope:
    ic0_ = tf.tanh(linear(fake_labels_logits,10,'i_checker_fake_linear0'))
    ic1_ = tf.tanh(linear(ic0_,10,'i_checker_fake_linear1'))
    ic2_ = tf.tanh(linear(ic1_,10,'i_checker_fake_linear2'))
    ic0 = tf.tanh(linear(real_labels_logits,10,'i_checker_real_linear0'))
    ic1 = tf.tanh(linear(ic0,10,'i_checker_real_linear1'))
    ic2 = tf.tanh(linear(ic1,10,'i_checker_real_linear2'))

    #v = linear(real_labels_logits,1,'i_checker_real_linear')
    ic2_mean = tf.reduce_mean(ic2,0)
    ic2_std = tf.sqrt( tf.reduce_mean(ic2**2 - ic2_mean**2) )
    ic2__mean = tf.reduce_mean(ic2_,0)
    ic2__std = tf.sqrt( tf.reduce_mean(ic2_**2 - ic2__mean**2) )

    pr_fake = tf.reduce_mean(D_logits_)
    std_fake = tf.sqrt( tf.reduce_mean(D_logits_**2 - pr_fake**2) )
    pr_real = tf.reduce_mean(D_logits)
    std_real = tf.sqrt( tf.reduce_mean(D_logits**2 - pr_real**2) )
    return tf.reduce_mean(tf.abs(tf.reduce_mean((ic2_-ic2__mean)*(D_logits_ - pr_fake),0))/(ic2__std*std_fake+self.TINY)), tf.reduce_mean(tf.abs(tf.reduce_mean((ic2-ic2_mean)*(D_logits - pr_real),0)/(ic2_std*std_real+self.TINY)))#tf.abs(tf.reduce_mean((v-v_mean)*(D_logits - pr_real)))

  def discriminator_CC(self, labels, reuse=False):
    with tf.variable_scope("discriminator_CC") as scope:
      if reuse:
        scope.reuse_variables()
      # add minibatch features here to get fake labels with high variation
      def add_minibatch_features_for_labels(labels,batch_size):
        n_kernels = 50
        dim_per_kernel = 20
        shape = labels.get_shape().as_list()
        dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
        input_ = tf.reshape(labels, [-1, dim])           # -1 means "all"  
        x = linear(input_, n_kernels * dim_per_kernel,'d_mbLabelLinear')
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
      #Old code (5-2-17):
      #labels_= add_minibatch_features_for_labels(labels,self.batch_size)
      #h0 = lrelu(labels_, self.hidden_size, 'dCC_0')
      #h1 = lrelu(h0, self.hidden_size,'dCC_1')
      #h2 = lrelu(h1, self.hidden_size, 'dCC_2')
      #h3 = linear(h2, 1, 'dCC_3')
      #return tf.nn.sigmoid(h3), h3

      #Old WARNING: BUG FOUND HERE
      #(no matrix multiplication or bias; only nonlinearity)
      ##Suggested replacement:
      #h0 = lrelu(labels, self.hidden_size, 'dCC_0')
      #h1 = lrelu(h0, self.hidden_size,'dCC_1')
      #h1 = add_minibatch_features_for_labels(h1, self.batch_size)
      #h2 = lrelu(h1, self.hidden_size, 'dCC_2')
      #h3 = linear(h2, 1, 'dCC_3')
      #return tf.nn.sigmoid(h3), h3

      #New BugFree code
      def lrelu(x,leak=0.2,name='lrelu'):
          with tf.variable_scope(name):
              f1=0.5 * (1+leak)#halves memory footprint
              f2=0.5 * (1-leak)#relative to tf.maximum(leak*x,x)
              return f1*x + f2*tf.abs(x)
      h0 = slim.fully_connected(labels,self.hidden_size,activation_fn=lrelu,scope='dCC_0')
      h1 = slim.fully_connected(labels,self.hidden_size,activation_fn=lrelu,scope='dCC_1')
      h1 = lrelu(add_minibatch_features_for_labels(h1,self.batch_size))
      h2 = slim.fully_connected(labels,self.hidden_size,activation_fn=lrelu,scope='dCC_2')
      h3 = slim.fully_connected(labels,self.hidden_size,activation_fn=None,scope='dCC_3')
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
############################################################


    #output later:
    #self.y_fake, self.z_fake_labels = self.causalController(self.zMale,self.zYoung,self.zSmiling)
  # def causalController(self, zMale,zYoung,zSmiling):# latent will capture noise to generate labels
  #   with tf.variable_scope("causal") as scope:
  #     # a nn from zMale to cMale
  #     h0Male = tf.tanh(linear(zMale, self.hidden_size, 'c_h0M'))
  #     #self.h1Male = tf.tanh(linear(self.h0Male, self.hidden_size, 'c_h1M'))
  #     #self.h1Male = linear(self.h0Male, self.hidden_size, 'c_h1M')
  #     h1Male = tf.tanh(linear(h0Male, self.hidden_size, 'c_h1M'))
  #     h2Male = linear(h1Male, 1, 'c_h2M')
  #     #self.cMale = tf.tanh(linear(self.h1Male, 1, 'c_M'))
  #     cMale = tf.sigmoid(h2Male)
  #     cMale_sym = 2*(cMale-0.5)
  #     # cMale_sym = tf.sign(h2Male)
  #     # cMale = 0.5*(cMale_sym+1)

  #     # a nn from zYoung to cYoung
  #     h0Young = tf.tanh(linear(tf.concat([zYoung,h2Male],1), self.hidden_size, 'c_h0Y'))
  #     #self.h1Young = tf.tanh(linear(self.h0Young, self.hidden_size, 'c_h1Y'))
  #     #self.h1Young = linear(self.h0Young, self.hidden_size, 'c_h1Y')
  #     #self.h1Young = linear(self.h0Young, 1, 'c_h1Y')
  #     h1Young = tf.tanh(linear(h0Young, self.hidden_size, 'c_h1Y'))
  #     h2Young = linear(h1Young, 1, 'c_h2Y')
  #     #self.cYoung = tf.tanh(linear(self.h1Young, 1, 'c_Y'))
  #     cYoung = tf.sigmoid(h2Young)
  #     cYoung_sym = 2*(cYoung-0.5)
  #     # cYoung_sym = tf.sign(h2Young)
  #     # cYoung = 0.5*(cYoung_sym+1)

  #     # a nn to generate cSmiling from cYoung, cMale and zSmiling
  #     #TODO 3xhidden_size -> hidden_size in one layer
  #     zSmilingTotal = tf.concat([h2Male, h2Young, zSmiling], 1)
  #     h0Smiling = tf.tanh(linear(zSmilingTotal, self.hidden_size, 'c_h0S'))
  #     #self.h1Smiling = tf.tanh(linear(self.h0Smiling, self.hidden_size, 'c_h1S'))
  #     #self.h1Smiling = linear(self.h0Smiling, self.hidden_size, 'c_h1S')
  #     #self.h1Smiling = linear(self.h0Smiling, 1, 'c_h1S')
  #     h1Smiling = tf.tanh(linear(h0Smiling, self.hidden_size, 'c_h1S'))
  #     h2Smiling = linear(h1Smiling, 1, 'c_h2S')
  #     #self.cSmiling = tf.tanh(linear(self.h1Smiling,1, 'c_S'))
  #     cSmiling = tf.sigmoid(h2Smiling)
  #     cSmiling_sym = 2*(cSmiling-0.5)
  #     # cSmiling_sym = tf.sign(h2Smiling)
  #     # cSmiling = 0.5*(cSmiling_sym+1)

  #     fake_labels=concat([cMale, cYoung, cSmiling],axis=1)
  #     fake_labels_logits = concat([h2Male,h2Young,h2Smiling], axis = 1)
  #     z_fake_labels = fake_labels_logits

  #   return fake_labels, fake_labels_logits, z_fake_labels


############################################################      
  def generator(self, z, y=None):
    #removed "if y_dim" part
    with tf.variable_scope("generator") as scope:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
#hehe
#  def sampler(self,z_gen,zMale,zYoung,zSmiling): 
#    rep_size = 1 # reduced from 3 to 1
#    with tf.variable_scope("causal") as scope:
#      print "Successfully in sampler"
#      scope.reuse_variables()
#      # a nn from zMale to cMale
#      h0Male = tf.tanh(linear(zMale, self.hidden_size, 'c_h0M'))
#      #self.h1Male = tf.tanh(linear(self.h0Male, self.hidden_size, 'c_h1M'))
#      #self.h1Male = linear(self.h0Male, self.hidden_size, 'c_h1M')
#      h1Male = tf.tanh(linear(h0Male, self.hidden_size, 'c_h1M'))
#      self.h2Male = linear(h1Male, 1, 'c_h2M')
#      #self.cMale = tf.tanh(linear(self.h1Male, 1, 'c_M'))
#
#      # cMale_sym = tf.sign(h2Male)
#      # cMale = 0.5*(cMale_sym+1)
#      #if intervention_set[0]:
#      #h2Male = (1-intervention_set[0])*h2Male + intervention_set[0]*intervention[:,0]
#      #  h2Male = intervention[:,0]
#      #  print "intervene_on successfully set to Male"
#      cMale = tf.sigmoid(self.h2Male)
#      cMale_sym = 2*(cMale-0.5) 
#
#      # a nn from zYoung to cYoung
#      h0Young = tf.tanh(linear(tf.concat([zYoung,self.h2Male],1), self.hidden_size, 'c_h0Y'))
#      #self.h1Young = tf.tanh(linear(self.h0Young, self.hidden_size, 'c_h1Y'))
#      #self.h1Young = linear(self.h0Young, self.hidden_size, 'c_h1Y')
#      #self.h1Young = linear(self.h0Young, 1, 'c_h1Y')
#      h1Young = tf.tanh(linear(h0Young, self.hidden_size, 'c_h1Y'))
#      self.h2Young = linear(h1Young, 1, 'c_h2Y')
#      #self.cYoung = tf.tanh(linear(self.h1Young, 1, 'c_Y'))
#
#      # cYoung_sym = tf.sign(h2Young)
#      # cYoung = 0.5*(cYoung_sym+1)
#      #if intervention_set[1]:
#      #h2Young = (1-intervention_set[1])*h2Young + intervention_set[1]*intervention[:,1]  
#        #h2Young = intervention[:,1]
#      #  print "intervene_on successfully set to Young"
#      cYoung = tf.sigmoid(self.h2Young)
#      cYoung_sym = 2*(cYoung-0.5) 
#      # a nn to generate cSmiling from cYoung, cMale and zSmiling
#      #TODO 3xhidden_size -> hidden_size in one layer
#      zSmilingTotal = tf.concat([self.h2Male, self.h2Young, zSmiling], 1)
#      h0Smiling = tf.tanh(linear(zSmilingTotal, self.hidden_size, 'c_h0S'))
#      #self.h1Smiling = tf.tanh(linear(self.h0Smiling, self.hidden_size, 'c_h1S'))
#      #self.h1Smiling = linear(self.h0Smiling, self.hidden_size, 'c_h1S')
#      #self.h1Smiling = linear(self.h0Smiling, 1, 'c_h1S')
#      h1Smiling = tf.tanh(linear(h0Smiling, self.hidden_size, 'c_h1S'))
#      self.h2Smiling = linear(h1Smiling, 1, 'c_h2S')
#      #self.cSmiling = tf.tanh(linear(self.h1Smiling,1, 'c_S'))
#
#      cSmiling = tf.sigmoid(self.h2Smiling)
#      cSmiling_sym = 2*(cSmiling-0.5) 
#
#      fake_labels=concat([cMale, cYoung, cSmiling],axis=1)
#      fake_labels_logits = concat([self.h2Male,self.h2Young,self.h2Smiling], axis = 1)
#      #self.interventional_labels = fake_labels_logits
#      z_fake_labels = fake_labels_logits
#      z = concat( [z_gen,z_fake_labels], axis=1)
#
#    with tf.variable_scope("generator") as scope:
#      scope.reuse_variables()
#  
#      s_h, s_w = self.output_height, self.output_width
#      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
#      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
#      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
#      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
#  
#       # project `z` and reshape
#      h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),[-1, s_h16, s_w16, self.gf_dim * 8])
#      h0 = tf.nn.relu(self.g_bn0(h0, train=False))
#  
#      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
#      h1 = tf.nn.relu(self.g_bn1(h1, train=False))
#  
#      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
#      h2 = tf.nn.relu(self.g_bn2(h2, train=False))
#  
#      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
#      h3 = tf.nn.relu(self.g_bn3(h3, train=False))
#  
#      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
#
#    return tf.nn.tanh(h4)


#  def sampler_label(self,zMale,zYoung,zSmiling):#z, y=None):
#    rep_size = 1 # reduced from 3 to 1
#    self.intervene_on = None
#    with tf.variable_scope("causal") as scope:
#      print "Successfully in sampler"
#      scope.reuse_variables()
#      # a nn from zMale to cMale
#      h0Male = tf.tanh(linear(zMale, self.hidden_size, 'c_h0M'))
#      #self.h1Male = tf.tanh(linear(self.h0Male, self.hidden_size, 'c_h1M'))
#      #self.h1Male = linear(self.h0Male, self.hidden_size, 'c_h1M')
#      h1Male = tf.tanh(linear(h0Male, self.hidden_size, 'c_h1M'))
#      h2Male = linear(h1Male, 1, 'c_h2M')
#      #self.cMale = tf.tanh(linear(self.h1Male, 1, 'c_M'))
#
#      # cMale_sym = tf.sign(h2Male)
#      # cMale = 0.5*(cMale_sym+1)
#
#      cMale = tf.sigmoid(h2Male)
#      cMale_sym = 2*(cMale-0.5) 
#
#      # a nn from zYoung to cYoung
#      h0Young = tf.tanh(linear(tf.concat([zYoung,h2Male],1), self.hidden_size, 'c_h0Y'))
#      #self.h1Young = tf.tanh(linear(self.h0Young, self.hidden_size, 'c_h1Y'))
#      #self.h1Young = linear(self.h0Young, self.hidden_size, 'c_h1Y')
#      #self.h1Young = linear(self.h0Young, 1, 'c_h1Y')
#      h1Young = tf.tanh(linear(h0Young, self.hidden_size, 'c_h1Y'))
#      h2Young = linear(h1Young, 1, 'c_h2Y')
#      #self.cYoung = tf.tanh(linear(self.h1Young, 1, 'c_Y'))
#
#      # cYoung_sym = tf.sign(h2Young)
#      # cYoung = 0.5*(cYoung_sym+1)
#
#      cYoung = tf.sigmoid(h2Young)
#      cYoung_sym = 2*(cYoung-0.5) 
#      # a nn to generate cSmiling from cYoung, cMale and zSmiling
#      #TODO 3xhidden_size -> hidden_size in one layer
#      zSmilingTotal = tf.concat([h2Male, h2Young, zSmiling], 1)
#      h0Smiling = tf.tanh(linear(zSmilingTotal, self.hidden_size, 'c_h0S'))
#      #self.h1Smiling = tf.tanh(linear(self.h0Smiling, self.hidden_size, 'c_h1S'))
#      #self.h1Smiling = linear(self.h0Smiling, self.hidden_size, 'c_h1S')
#      #self.h1Smiling = linear(self.h0Smiling, 1, 'c_h1S')
#      h1Smiling = tf.tanh(linear(h0Smiling, self.hidden_size, 'c_h1S'))
#      h2Smiling = linear(h1Smiling, 1, 'c_h2S')
#      #self.cSmiling = tf.tanh(linear(self.h1Smiling,1, 'c_S'))
#
#      # cSmiling_sym = tf.sign(h2Smiling)
#      # cSmiling = 0.5*(cSmiling_sym+1)
#
#      cSmiling = tf.sigmoid(h2Smiling)
#      cSmiling_sym = 2*(cSmiling-0.5) 
#
#      fake_labels=concat([cMale, cYoung, cSmiling],axis=1)
#      fake_labels_logits = concat([h2Male,h2Young,h2Smiling], axis = 1)
#    return fake_labels_logits


  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
