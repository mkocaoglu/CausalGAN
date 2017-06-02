from __future__ import print_function
import tensorflow as tf
#import scipy
import scipy.misc
import numpy as np
from tqdm import trange
import os
import pandas as pd
from itertools import combinations
import sys
from CausalController import *
from utils import to_nhwc,read_prepared_uint8_image,make_encode_dir

from utils import transform, inverse_transform #dcgan img norm
from utils import norm_img, denorm_img #began norm image

def var_like_z(z_ten,name):
    z_dim=node.z.get_shape().as_list()[-1]
    return tf.get_variable(name,shape=(1,z_dim))

class Encoder:
    '''
    This is a class where you pass a model, and an image file
    and it creates more tensorflow variables, along with
    surrounding saving and summary functionality for encoding
    that image back into the hidden space using gradient descent
    '''
    model_name = "DCGAN.model"
    model_type= 'encoder'
    summ_col='encoder_summaries'
    def __init__(self,model,image,image_name=None,load_path=''):
        '''
        image is assumed to be a path to a precropped 64x64x3 uint8 image

        '''

        #Some hardcoded defaults here
        self.log_step=100
        self.lr=0.0005
        self.max_tr_steps=50000

        self.model=model
        self.load_path=load_path
        self.encode_dir=make_encode_dir(model)
        self.model_dir=encode_dir#different from self.model.model_dir

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
        x=norm_img(self.uint_x,self.data_format)
        self.x=tf.expand_dims(x,0)#batch_size=1

        self.image_name=image_name or os.path.basename(image).replace('.','_')
        self.g_optimizer = optimizer(self.lr)

        ##All variables created under encoder have uniform init
        vs=tf.variable_scope('encoder',
             initializer=tf.random_uniform_initializer(minval=-1.,maxval=1.),
             dtype=tf.float32)

        if self.load_path:
            print 'Attempting to load directly from path:',
            print self.load_path
            self.saver.restore(self.load_path)

        with vs as scope:
            encode_var={n.name:var_like_z(n.z,n.name) for n in model.cc.nodes}
            encode_var['gen':var_like_z(model.z_gen,'gen')]
            print 'encode variables created'
            self.train_var = tf.contrib.framework.get_variables(scope)
            self.step=tf.Variable(0.,'step')
            self.var = tf.contrib.framework.get_variables(scope)

        #all encode vars created by now
        self.saver = tf.train.Saver(var_list=self.var)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        if model.model_type =='dcgan':
            self.cc=CausalController(graph=model.graph, input_dict=encode_var, reuse=True)
            self.fake_labels_logits= tf.concat( self.cc.list_label_logits(),-1 )
            self.z_fake_labels=self.fake_labels_logits
            self.z= concat( [self.z_gen, self.z_fake_labels], axis=1 , name='z')

            self.G=model.generator( self.z )

        elif model.model_type == 'began':
            with tf.variable_scope('tower'):#reproduce variable scope
                self.cc=CausalController(graph=model.graph, input_dict=encode_var, reuse=True)

                self.fake_labels= tf.concat( self.cc.list_labels(),-1 )
                self.fake_labels_logits= tf.concat( self.cc.list_label_logits(),-1 )
                self.z= tf.concat( [self.fake_labels, self.z_gen],axis=-1,name='z')

                self.G,_ = GeneratorCNN(
                        self.z, model.conv_hidden_num, model.channel,
                        model.repeat_num, model.data_format)

        #use L1 loss
        self.g_loss_image = tf.reduce_mean(tf.abs(self.x - self.G))
        tf.summary.scalar( 'encoder/g_loss_image',
                          self.g_loss_image,self.summ_col)

        self.g_loss= self.g_loss_image
        self.train_op=self.g_optimizer.minimize(self.g_loss,
               var_list=self.train_var,global_step=self.step)

        self.uint_G=tf.squeeze(denorm_img( G ))#3D[0,255]
        tf.summary.image( 'encoder/Reconstruct',tf.stack([self.uint_x,self.uint_G]),
                max_outputs=2,collections=self.summ_col)

        self.summary_op=tf.summary.merge_all(self.summ_col)

    def save(self, checkpoint_dir=None, step):
        checkpoint_dir=checkpoint_dir or self.model_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
            os.path.join(checkpoint_dir,model_name),
            global_step=step)

    def train(self, n_step=None):
        max_step=n_step or self.max_tr_steps
        for counter in trange(max_step):
            fetch_dict = {
                "train_op": self.train_op,
            }
            if counter%self.log_step==0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    })
            result = self.sess.run(fetch_dict)


            if counter % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'],
                                                result['global_step'])
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Reconstr Loss_G: {:.6f}".format(counter,max_step,g_loss,))




##Just for reference##
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





if model.model_name=='began':
    fake_labels=model.fake_labels
    D_fake_labels=model.D_fake_labels
    #result_dir=os.path.join('began',model.model_dir)
    result_dir=model.model_dir
    if str_step=='':
        str_step=str( model.sess.run(model.step) )+'_'
    attr=model.attr[list(model.cc.node_names)]
elif model.model_name=='dcgan':
    fake_labels=model.fake_labels
    D_fake_labels=model.D_labels_for_fake
    result_dir=model.checkpoint_dir
    attr=0.5*(model.attributes+1)
    attr=attr[list(model.cc.names)]

