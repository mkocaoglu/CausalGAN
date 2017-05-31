import tensorflow as tf
#import scipy
import scipy.misc
import numpy as np
from tqdm import trange
import os
import pandas as pd
from itertools import combinations
import sys


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def read_prepared_image(img_path):
    '''
    img_path should point to a uint8 image that is
    already cropped and resized
    '''
    scipy.misc.imread(path).astype(np.float)
    img=array(cropped_image)/127.5 - 1.
    ####
    print 'WARN'
    if began: 
        if data_format:
            image = to_nhwc(image, data_format)

  return img


class Encoder:

def __init__(self,model,image,image_name=None,load_path=''):
    '''
    image is assumed to be a path to a precropped 64x64x3 uint8 image

    '''
    self.model=model
    self.image=read_prepared_image(image)

    self.image_name=image_name or os.path.basename(image)

    vs=tf.variable_scope('z_encode',
             initializer=tf.random_uniform_initializer(minval=0,maxval=1),
             dtype=tf.float32)
    def var_like_z(z_ten,name):
        z_dim=node.z.get_shape.as_list()[-1]
        return tf.get_variable(name,shape=(1,z_dim))
    with vs as scope:
        encode_var={n.name:var_like_z(n.z,n.name) for n in model.cc.nodes}
        encode_var['gen':var_like_z(model.z_gen,'gen']





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

