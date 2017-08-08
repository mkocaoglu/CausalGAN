#from __future__ import print_function
import tensorflow as tf
#import scipy
import scipy.misc
import numpy as np
from tqdm import trange
import os
import pandas as pd
from itertools import combinations
import sys
from Causal_controller import *
from began.models import GeneratorCNN, DiscriminatorCNN
from utils import to_nhwc,read_prepared_uint8_image,make_encode_dir

from utils import transform, inverse_transform #dcgan img norm
from utils import norm_img, denorm_img #began norm image

def var_like_z(z_ten,name):
    z_dim=z_ten.get_shape().as_list()[-1]
    return tf.get_variable(name,shape=(1,z_dim))
def noise_like_z(z_ten,name):
    z_dim=z_ten.get_shape().as_list()[-1]
    noise=tf.random_uniform([1,z_dim],minval=-1.,maxval=1.,)
    return noise


class Encoder:
    '''
    This is a class where you pass a model, and an image file
    and it creates more tensorflow variables, along with
    surrounding saving and summary functionality for encoding
    that image back into the hidden space using gradient descent
    '''
    model_name = "Encode.model"
    model_type= 'encoder'
    summ_col='encoder_summaries'
    def __init__(self,model,image,image_name=None,max_tr_steps=50000,load_path=''):
        '''
        image is assumed to be a path to a precropped 64x64x3 uint8 image
        '''

        #Some hardcoded defaults here
        self.log_step=500
        self.lr=0.0005
        self.max_tr_steps=max_tr_steps

        self.model=model
        self.load_path=load_path

        self.image_name=image_name or os.path.basename(image).replace('.','_')
        self.encode_dir=make_encode_dir(model,self.image_name)
        self.model_dir=self.encode_dir#different from self.model.model_dir
        self.save_dir=os.path.join(self.model_dir,'save')

        self.sess=self.model.sess#session should already be in progress

        if model.model_type =='dcgan':
            self.data_format='NHWC'#Don't change
        elif model.model_type == 'began':
            self.data_format=model.data_format#'NCHW' if gpu
        else:
            raise Exception('Should not happen. model_type=',model.model_type)

        #Notation:
        #self.uint_x/G ; 3D [0,255]
        #self.x/G ; 4D [-1,1]
        self.uint_x=read_prepared_uint8_image(image)#x is [0,255]

        print('Read image shape',self.uint_x.shape)
        self.x=norm_img(np.expand_dims(self.uint_x,0),self.data_format)#bs=1
        #self.x=norm_img(tf.expand_dims(self.uint_x,0),self.data_format)#bs=1
        print('Shape after norm:',self.x.get_shape().as_list())


        ##All variables created under encoder have uniform init
        vs=tf.variable_scope('encoder',
             initializer=tf.random_uniform_initializer(minval=-1.,maxval=1.),
             dtype=tf.float32)


        with vs as scope:
            #avoid creating adams params
            optimizer = tf.train.GradientDescentOptimizer
            #optimizer = tf.train.AdamOptimizer
            self.g_optimizer = optimizer(self.lr)

            encode_var={n.name:var_like_z(n.z,n.name) for n in model.cc.nodes}
            encode_var['gen']=var_like_z(model.z_gen,'gen')
            print 'encode variables created'
            self.train_var = tf.contrib.framework.get_variables(scope)
            self.step=tf.Variable(0,name='step')
            self.var = tf.contrib.framework.get_variables(scope)

        #all encode vars created by now
        self.saver = tf.train.Saver(var_list=self.var)
        print('Summaries will be written to ',self.model_dir)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        #load or initialize enmodel variables
        self.init()

        if model.model_type =='dcgan':
            self.cc=CausalController(graph=model.graph, input_dict=encode_var, reuse=True)
            self.fake_labels_logits= tf.concat( self.cc.list_label_logits(),-1 )
            self.z_fake_labels=self.fake_labels_logits
            #self.z_gen = noise_like_z( self.model.z_gen,'en_z_gen')
            self.z_gen=encode_var['gen']
            self.z= tf.concat( [self.z_gen, self.z_fake_labels], axis=1 , name='z')

            self.G=model.generator( self.z , bs=1, reuse=True)

        elif model.model_type == 'began':
            with tf.variable_scope('tower'):#reproduce variable scope
                self.cc=CausalController(graph=model.graph, input_dict=encode_var, reuse=True)

                self.fake_labels= tf.concat( self.cc.list_labels(),-1 )
                self.fake_labels_logits= tf.concat( self.cc.list_label_logits(),-1 )
                #self.z_gen = noise_like_z( self.model.z_gen,'en_z_gen')
                self.z_gen=encode_var['gen']
                self.z= tf.concat( [self.fake_labels, self.z_gen],axis=-1,name='z')

                self.G,_ = GeneratorCNN(
                        self.z, model.conv_hidden_num, model.channel,
                        model.repeat_num, model.data_format,reuse=True)

                d_out, self.D_zG, self.D_var = DiscriminatorCNN(
                        self.G, model.channel, model.z_num,
                    model.repeat_num, model.conv_hidden_num,
                    model.data_format,reuse=True)

                _   , self.D_zX, _           = DiscriminatorCNN(
                        self.x, model.channel, model.z_num,
                    model.repeat_num, model.conv_hidden_num,
                    model.data_format,reuse=True)
                self.norm_AE_G=d_out

                #AE_G, AE_x = tf.split(d_out, 2)
                self.AE_G=denorm_img(self.norm_AE_G, model.data_format)
            self.aeg_sum=tf.summary.image('encoder/AE_G',self.AE_G)

        node_summaries=[]
        for node in self.cc.nodes:
            with tf.name_scope(node.name):
                ave_label=tf.reduce_mean(node.label)
                node_summaries.append(tf.summary.scalar('ave',ave_label))


        #unclear how scope with adam param works
        #with tf.variable_scope('encoderGD') as scope:

        #use L1 loss
        #self.g_loss_image = tf.reduce_mean(tf.abs(self.x - self.G))

        #use L2 loss
        #self.g_loss_image = tf.reduce_mean(tf.square(self.x - self.G))

        #use autoencoder reconstruction loss  #3.1.1 series
        #self.g_loss_image = tf.reduce_mean(tf.abs(self.x - self.norm_AE_G))

        #use L1 in autoencoded space# 3.2
        self.g_loss_image = tf.reduce_mean(tf.abs(self.D_zX - self.D_zG))

        g_loss_sum=tf.summary.scalar( 'encoder/g_loss_image',\
                          self.g_loss_image,self.summ_col)

        self.g_loss= self.g_loss_image
        self.train_op=self.g_optimizer.minimize(self.g_loss,
               var_list=self.train_var,global_step=self.step)

        self.uint_G=tf.squeeze(denorm_img( self.G ,self.data_format))#3D[0,255]
        gimg_sum=tf.summary.image( 'encoder/Reconstruct',tf.stack([self.uint_x,self.uint_G]),\
                max_outputs=2,collections=self.summ_col)

        #self.summary_op=tf.summary.merge_all(self.summ_col)
        #self.summary_op=tf.summary.merge_all(self.summ_col)

        if model.model_type=='dcgan':
            self.summary_op=tf.summary.merge([g_loss_sum,gimg_sum]+node_summaries)
        elif model.model_type=='began':
            self.summary_op=tf.summary.merge([g_loss_sum,gimg_sum,self.aeg_sum]+node_summaries)


        #print 'encoder summaries:',self.summ_col
        #print 'encoder summaries:',tf.get_collection(self.summ_col)


    def init(self):
        if self.load_path:
            print 'Attempting to load directly from path:',
            print self.load_path
            self.saver.restore(self.sess,self.load_path)
        else:
            print 'New ENCODE Model..init new Z parameters'
            init=tf.variables_initializer(var_list=self.var)
            print 'Initializing following variables:'
            for v in self.var:
                print v.name, v.get_shape().as_list()

            self.model.sess.run(init)

    def save(self, step=None):
        if step is None:
            step=self.sess.run(self.step)

        if not os.path.exists(self.save_dir):
            print 'Creating Directory:',self.save_dir
            os.makedirs(self.save_dir)
        savefile=os.path.join(self.save_dir,self.model_name)
        print 'Saving file:',savefile
        self.saver.save(self.model.sess,savefile,global_step=step)

    def train(self, n_step=None):
        max_step=n_step or self.max_tr_steps

        if False:#debug
            print 'a'
            self.sess.run(self.train_op)
            print 'b'
            self.sess.run(self.summary_op)
            print 'c'
            self.sess.run(self.g_loss)
            print 'd'

        print 'max_step;',max_step
        for counter in trange(max_step):

            fetch_dict = {
                "train_op": self.train_op,
            }
            if counter%self.log_step==0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "global_step":self.step
                    })

            result = self.sess.run(fetch_dict)

            if counter % self.log_step == 0:
                g_loss=result['g_loss']
                step=result['global_step']
                self.summary_writer.add_summary(result['summary'],step)
                self.summary_writer.flush()

                print("[{}/{}] Reconstr Loss_G: {:.6f}".format(counter,max_step,g_loss))

            if counter % (10.*self.log_step) == 0:
                self.save(step=step)

        self.save()



##Just for reference##
    #def load(self, checkpoint_dir):
    #    print(" [*] Reading checkpoints...")
    #    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    #    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #    if ckpt and ckpt.model_checkpoint_path:
    #        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #        print(" [*] Success to read {}".format(ckpt_name))
    #        return True
    #    else:
    #        print(" [*] Failed to find a checkpoint")
    #        return False
#def norm_img(image, data_format=None):
#    image = image/127.5 - 1.
#    if data_format:
#        image = to_nhwc(image, data_format)
#    return image
#def transform:
#    stuff
#  return np.array(cropped_image)/127.5 - 1.
#def denorm_img(norm, data_format):
#    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)
#def inverse_transform(images):
#  return (images+1.)/2.



#if model.model_name=='began':
#    fake_labels=model.fake_labels
#    D_fake_labels=model.D_fake_labels
#    #result_dir=os.path.join('began',model.model_dir)
#    result_dir=model.model_dir
#    if str_step=='':
#        str_step=str( model.sess.run(model.step) )+'_'
#    attr=model.attr[list(model.cc.node_names)]
#elif model.model_name=='dcgan':
#    fake_labels=model.fake_labels
#    D_fake_labels=model.D_labels_for_fake
#    result_dir=model.checkpoint_dir
#    attr=0.5*(model.attributes+1)
#    attr=attr[list(model.cc.names)]

