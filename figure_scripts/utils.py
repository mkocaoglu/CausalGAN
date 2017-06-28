from __future__ import print_function,division
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import shutil
import sys
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime

import tensorflow as tf
from PIL import Image

import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

pp = pprint.PrettyPrinter()

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])
def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    #image = tf.cast(image,tf.float32)/127.5 - 1.
    image = image/127.5 - 1.
    #if data_format:
        #image = to_nhwc(image, data_format)
    if data_format=='NCHW':
        image = to_nchw_numpy(image)

    image=tf.cast(image,tf.float32)
    return image


#Denorming
def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])
def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image
def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def read_prepared_uint8_image(img_path):
    '''
    img_path should point to a uint8 image that is
    already cropped and resized
    '''
    cropped_image=scipy.misc.imread(img_path)
    if not np.all( np.array([64,64,3])==cropped_image.shape):
        raise ValueError('image must already be cropped and resized:',img_path)
    #TODO: warn if wrong dtype
    return cropped_image

def make_encode_dir(model,image_name):
    #Terminology
    if model.model_type=='began':
        result_dir=model.model_dir
    elif model.model_type=='dcgan':
        print('DCGAN')
        result_dir=model.checkpoint_dir
    encode_dir=os.path.join(result_dir,'encode_'+str(image_name))
    if not os.path.exists(encode_dir):
        os.mkdir(encode_dir)
    return encode_dir

def make_sample_dir(model):
    #Terminology
    if model.model_type=='began':
        result_dir=model.model_dir
    elif model.model_type=='dcgan':
        print('DCGAN')
        result_dir=model.checkpoint_dir

    sample_dir=os.path.join(result_dir,'sample_figures')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    return sample_dir

def guess_model_step(model):
    if model.model_type=='began':
        str_step=str( model.sess.run(model.step) )+'_'
    elif model.model_type=='dcgan':
        result_dir=model.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(result_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        str_step=ckpt_name[-5:]+'_'
    return str_step

def infer_grid_image_shape(N):
    if N%8==0:
        size=[8,N//8]
    else:
        size=[8,8]
    return size


def save_figure_images(model_type, tensor, filename, size, padding=2, normalize=False, scale_each=False):

    print('[*] saving:',filename)

    #nrow=size[0]
    nrow=size[1]#Was this number per row and now number of rows?

    if model_type=='began':
        began_save_image(tensor,filename,nrow,padding,normalize,scale_each)
    elif model_type=='dcgan':
        #images = np.split(tensor,len(tensor))
        images=tensor
        dcgan_save_images(images,size,filename)


#Began originally
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def began_save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)



#Dcgan originally
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, is_crop)

def dcgan_save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, is_crop=True):
  if is_crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.


