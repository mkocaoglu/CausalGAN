"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import os 

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, is_crop)

def save_images(images, size, image_path):
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

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  pMale = 0.416754
  pYoung = 0.773617
  pSmiling = 0.482080
  # my_intervention = np.zeros((config.batch_size,3))
  # my_set = np.array([0.0,0.0,0.0])
  if option == 0:
    # randomly sample male vector
    if not os.path.exists(config.sample_dir+'MaleRandom_z'):
      os.makedirs(config.sample_dir+'MaleRandom_z')
      ## Following is changing each coordinate at a time
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-1, 1, size=(image_frame_dim, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Young = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim,1])
      #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
      z_Smiling = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
      z_Male = np.random.uniform(-1,1,size=(image_frame_dim,dcgan.SmilingDim))
      z_Male = np.repeat(z_Male,image_frame_dim,axis = 0)
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesMaleRandom_z/test_arange_%s.png' % (idx))
      ## Following is starting from [1,1,1,1,1,0,0,0,0,0] and moving to [0,0,0,0,0,1,1,1,1,1] and flipping a bit at a time
      ##
  elif option == 1:
    if not os.path.exists(config.sample_dir+'Male'):
      os.makedirs(config.sample_dir+'Male')
    # this was default, so modified here
   
#     values = np.arange(-1, 1, 1./image_frame_dim)
#     #z_gen_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_gen_dim))
#     z_gen_sample = np.random.uniform(-1, 1, size=(image_frame_dim, dcgan.z_gen_dim))
#     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
#     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
#     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
#     z_Young = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.YoungDim))
#     z_Young = np.tile(z_Young,[image_frame_dim,1])
#     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
#     z_Smiling = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.SmilingDim))
#     z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
#     for idx in xrange(dcgan.YoungDim): # was hard coded to 100 before
#       print(" [*] %d" % idx)
#       #z_gen_sample = np.zeros([config.batch_size, dcgan.z_gen_dim])
#       z_Male = np.zeros([image_frame_dim,dcgan.MaleDim])
#       #z_Young = np.zeros([config.batch_size,dcgan.YoungDim])
#       #z_Smiling = np.zeros([config.batch_size,dcgan.SmilingDim])
      
# #      for kdx, z_g in enumerate(z_gen_sample):
# #        z_g[idx] = values[kdx]
#       for kdx, z_M in enumerate(z_Male):
#         z_M[idx] = values[kdx]
# #      for kdx, z_Y in enumerate(z_Young):
# #        z_Y[idx] = values[kdx]
#       #for kdx, z_S in enumerate(z_Smiling):
#       #  z_S[idx] = values[kdx]
#       z_Male = np.repeat(z_Male,image_frame_dim,axis = 0)
#       if config.dataset == "mnist":
#         y = np.random.choice(10, config.batch_size)
#         y_one_hot = np.zeros((config.batch_size, 10))
#         y_one_hot[np.arange(config.batch_size), y] = 1

#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
#       else:
#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})

#       save_images(samples, [image_frame_dim, image_frame_dim], './samplesMale/test_arange_%s.png' % (idx))
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-1, 1, size=(image_frame_dim, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
      z_Young = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim,1])
      z_Smiling = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
      z_Male = np.array([ [ 1,  1,  1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1,  1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  -1,  1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1]])
      z_Male = np.repeat(z_Male,image_frame_dim,axis = 0)
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesMale/test_arange_%s.png' % (idx))
  elif option == 2:
    if not os.path.exists(config.sample_dir+'Young'):
      os.makedirs(config.sample_dir+'Young')
    # this was default, so modified here
   
#     values = np.arange(-1, 1, 1./image_frame_dim)
#     #z_gen_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_gen_dim))
#     z_gen_sample = np.random.uniform(-1, 1, size=(image_frame_dim, dcgan.z_gen_dim))
#     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
#     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
#     z_Male = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.MaleDim))
#     z_Male = np.tile(z_Male,[image_frame_dim,1])
#     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
#     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
#     z_Smiling = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.SmilingDim))
#     z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
#     for idx in xrange(dcgan.YoungDim): # was hard coded to 100 before
#       print(" [*] %d" % idx)
#       #z_gen_sample = np.zeros([config.batch_size, dcgan.z_gen_dim])
#       #z_Male = np.zeros([config.batch_size,dcgan.MaleDim])
#       z_Young = np.zeros([image_frame_dim,dcgan.YoungDim])
#       #z_Smiling = np.zeros([config.batch_size,dcgan.SmilingDim])
      
# #      for kdx, z_g in enumerate(z_gen_sample):
# #        z_g[idx] = values[kdx]
#       #for kdx, z_M in enumerate(z_Male):
#       #  z_M[idx] = values[kdx]
#       for kdx, z_Y in enumerate(z_Young):
#         z_Y[idx] = values[kdx]
#       #for kdx, z_S in enumerate(z_Smiling):
#       #  z_S[idx] = values[kdx]
#       z_Young = np.repeat(z_Young,image_frame_dim,axis = 0)
#       if config.dataset == "mnist":
#         y = np.random.choice(10, config.batch_size)
#         y_one_hot = np.zeros((config.batch_size, 10))
#         y_one_hot[np.arange(config.batch_size), y] = 1

#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
#       else:
#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})

#       save_images(samples, [image_frame_dim, image_frame_dim], './samplesYoung/test_arange_%s.png' % (idx))
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-1, 1, size=(image_frame_dim, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
      z_Male = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.YoungDim))
      z_Male = np.tile(z_Male,[image_frame_dim,1])
      z_Smiling = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
      z_Young = np.array([ [ 1,  1,  1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1,  1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  -1,  1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1]])
      z_Young = np.repeat(z_Young,image_frame_dim,axis = 0)
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesYoung/test_arange_%s.png' % (idx))
  elif option == 3:
    if not os.path.exists(config.sample_dir+'Smiling'):
      os.makedirs(config.sample_dir+'Smiling')
    # this was default, so modified here
   
#     values = np.arange(-1, 1, 1./image_frame_dim)
#     #z_gen_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_gen_dim))
#     z_gen_sample = np.random.uniform(-1, 1, size=(image_frame_dim, dcgan.z_gen_dim))
#     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
#     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
#     z_Male = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.MaleDim))
#     z_Male = np.tile(z_Male,[image_frame_dim,1])
#     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
#     z_Young = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.YoungDim))
#     z_Young = np.tile(z_Young,[image_frame_dim,1])
#     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
#     for idx in xrange(dcgan.YoungDim): # was hard coded to 100 before
#       print(" [*] %d" % idx)
#       #z_gen_sample = np.zeros([config.batch_size, dcgan.z_gen_dim])
#       #z_Male = np.zeros([config.batch_size,dcgan.MaleDim])
#       #z_Young = np.zeros([config.batch_size,dcgan.YoungDim])
#       z_Smiling = np.zeros([image_frame_dim,dcgan.SmilingDim])
      
# #      for kdx, z_g in enumerate(z_gen_sample):
# #        z_g[idx] = values[kdx]
#       #for kdx, z_M in enumerate(z_Male):
#       #  z_M[idx] = values[kdx]
#       #for kdx, z_Y in enumerate(z_Young):
#       #  z_Y[idx] = values[kdx]
#       for kdx, z_S in enumerate(z_Smiling):
#         z_S[idx] = values[kdx]
#       z_Smiling = np.repeat(z_Smiling,image_frame_dim,axis = 0)
#       if config.dataset == "mnist":
#         y = np.random.choice(10, config.batch_size)
#         y_one_hot = np.zeros((config.batch_size, 10))
#         y_one_hot[np.arange(config.batch_size), y] = 1

#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
#       else:
#         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
      
#       save_images(samples, [image_frame_dim, image_frame_dim], './samplesSmiling/test_arange_%s.png' % (idx))
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-1, 1, size=(image_frame_dim, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
      z_Male = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.YoungDim))
      z_Male = np.tile(z_Male,[image_frame_dim,1])
      z_Young = np.random.uniform(-1, 1, size=(image_frame_dim,dcgan.SmilingDim))
      z_Young = np.tile(z_Young,[image_frame_dim,1])
      z_Smiling = np.array([ [ 1,  1,  1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1,  1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1,  1,  1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  1, 1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  1, 1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  -1,  1, 1],\
                          [-1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1]])
      z_Smiling = np.repeat(z_Smiling,image_frame_dim,axis = 0)
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesSmiling/test_arange_%s.png' % (idx))

  # elif option == 4:
  #         # randomly sample male vector
  #   if not os.path.exists(config.sample_dir+'MaleRandom_z'):
  #     os.makedirs(config.sample_dir+'MaleRandom_z')
  #     ## Following is changing each coordinate at a time
  #   for idx in xrange(10):
  #     print(" [*] %d" % idx)
  #     z_gen_sample = np.random.uniform(-1, 1, size=(1, dcgan.z_gen_dim))
  #     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
  #     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
  #     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
  #     z_Smiling = np.random.uniform(-1, 1, size=(1,dcgan.YoungDim))
  #     z_Smiling = np.tile(z_Smiling,[image_frame_dim**2,1])
  #     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
  #     z_Young = np.random.uniform(-1, 1, size=(1,dcgan.SmilingDim))
  #     z_Young = np.tile(z_Young,[image_frame_dim**2,1])
  #     z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
  #     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
  #     save_images(samples, [image_frame_dim, image_frame_dim], './samplesMaleRandom_z/test_arange_%s.png' % (idx))

  # elif option == 5:
  #       # randomly sample male vector
  #   if not os.path.exists(config.sample_dir+'YoungRandom_z'):
  #     os.makedirs(config.sample_dir+'YoungRandom_z')
  #     ## Following is changing each coordinate at a time
  #   for idx in xrange(10):
  #     print(" [*] %d" % idx)
  #     z_gen_sample = np.random.uniform(-1, 1, size=(1, dcgan.z_gen_dim))
  #     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
  #     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
  #     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
  #     z_Smiling = np.random.uniform(-1, 1, size=(1,dcgan.YoungDim))
  #     z_Smiling = np.tile(z_Smiling,[image_frame_dim**2,1])
  #     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
  #     z_Male = np.random.uniform(-1, 1, size=(1,dcgan.SmilingDim))
  #     z_Male = np.tile(z_Male,[image_frame_dim**2,1])
  #     z_Young = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
  #     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
  #     save_images(samples, [image_frame_dim, image_frame_dim], './samplesYoungRandom_z/test_arange_%s.png' % (idx))

  # elif option == 6:
  #       # randomly sample male vector
  #   if not os.path.exists(config.sample_dir+'SmilingRandom_z'):
  #     os.makedirs(config.sample_dir+'SmilingRandom_z')
  #     ## Following is changing each coordinate at a time
  #   for idx in xrange(10):
  #     print(" [*] %d" % idx)
  #     z_gen_sample = np.random.uniform(-1, 1, size=(1, dcgan.z_gen_dim))
  #     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
  #     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
  #     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
  #     z_Young = np.random.uniform(-1, 1, size=(1,dcgan.YoungDim))
  #     z_Young = np.tile(z_Young,[image_frame_dim**2,1])
  #     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
  #     z_Male = np.random.uniform(-1, 1, size=(1,dcgan.SmilingDim))
  #     z_Male = np.tile(z_Male,[image_frame_dim**2,1])
  #     z_Smiling = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
  #     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
  #     save_images(samples, [image_frame_dim, image_frame_dim], './samplesSmilingRandom_z/test_arange_%s.png' % (idx))


  # elif option == 4:
  #       # randomly sample male vector
  #   if not os.path.exists(config.sample_dir+'MaleRandom_z'):
  #     os.makedirs(config.sample_dir+'MaleRandom_z')
  #     ## Following is changing each coordinate at a time
  #   alpha = np.linspace(0,1,image_frame_dim)
  #   for idx in xrange(10):
  #     print(" [*] %d" % idx)
  #     z_gen_sample = np.random.uniform(-1, 1, size=(1, dcgan.z_gen_dim))
  #     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
  #     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
  #     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
  #     z_Smiling = np.random.uniform(-1, 1, size=(1,dcgan.YoungDim))
  #     z_Smiling = np.tile(z_Smiling,[image_frame_dim**2,1])
  #     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
  #     z_Young = np.random.uniform(-1, 1, size=(1,dcgan.SmilingDim))
  #     z_Young = np.tile(z_Young,[image_frame_dim**2,1])
  #     #z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
  #     z_Male_begin = np.random.uniform(-1,1,size=(image_frame_dim,dcgan.SmilingDim))
  #     z_Male_end = np.random.uniform(-1,1,size=(image_frame_dim,dcgan.SmilingDim)) 
  #     z_Male = z_Male_begin
  #     for index in range(1,image_frame_dim):
  #       dum = z_Male_begin*(1-alpha[index])+z_Male_end*(alpha[index])
  #       z_Male=np.vstack((z_Male,dum))
  #     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
  #     save_images(samples, [image_frame_dim, image_frame_dim], './samplesMaleRandom_z/test_arange_%s.png' % (idx))

  # elif option == 5:
  #       # randomly sample male vector
  #   if not os.path.exists(config.sample_dir+'YoungRandom_z'):
  #     os.makedirs(config.sample_dir+'YoungRandom_z')
  #     ## Following is changing each coordinate at a time
  #   alpha = np.linspace(0,1,image_frame_dim)
  #   for idx in xrange(10):
  #     print(" [*] %d" % idx)
  #     z_gen_sample = np.random.uniform(-0.5, 0.5, size=(1, dcgan.z_gen_dim))
  #     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
  #     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
  #     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
  #     z_Smiling = np.random.uniform(-0.5, 0.5, size=(1,dcgan.YoungDim))
  #     z_Smiling = np.tile(z_Smiling,[image_frame_dim**2,1])
  #     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
  #     z_Male = np.random.uniform(-0.5, 0.5, size=(1,dcgan.SmilingDim))
  #     z_Male = np.tile(z_Male,[image_frame_dim**2,1])
  #     #z_Young = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
  #     z_Young_begin = np.random.uniform(-0.5,0.5,size=(image_frame_dim,dcgan.SmilingDim))
  #     z_Young_end = np.random.uniform(-0.5,0.5,size=(image_frame_dim,dcgan.SmilingDim))
  #     z_Young = z_Young_begin
  #     for index in range(1,image_frame_dim):
  #       dum = z_Young_begin*(1-alpha[index])+z_Young_end*(alpha[index])
  #       z_Young=np.vstack((z_Young,dum))
  #     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
  #     save_images(samples, [image_frame_dim, image_frame_dim], './samplesYoungRandom_z/test_arange_%s.png' % (idx))

  # elif option == 6:
  #       # randomly sample male vector
  #   if not os.path.exists(config.sample_dir+'SmilingRandom_z'):
  #     os.makedirs(config.sample_dir+'SmilingRandom_z')
  #     ## Following is changing each coordinate at a time
  #   alpha = np.linspace(0,1,image_frame_dim)
  #   for idx in xrange(10):
  #     print(" [*] %d" % idx)
  #     z_gen_sample = np.random.uniform(-0.5, 0.5, size=(1, dcgan.z_gen_dim))
  #     z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
  #     #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
  #     #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
  #     z_Young = np.random.uniform(-0.5, 0.5, size=(1,dcgan.YoungDim))
  #     z_Young = np.tile(z_Young,[image_frame_dim**2,1])
  #     #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
  #     z_Male = np.random.uniform(-0.5, 0.5, size=(1,dcgan.SmilingDim))
  #     z_Male = np.tile(z_Male,[image_frame_dim**2,1])
  #     #z_Smiling = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
  #     z_Smiling_begin = np.random.uniform(-0.5,0.5,size=(image_frame_dim,dcgan.SmilingDim))
  #     z_Smiling_end = np.random.uniform(-0.5,0.5,size=(image_frame_dim,dcgan.SmilingDim))
  #     z_Smiling = z_Smiling_begin
  #     for index in range(1,image_frame_dim):
  #       dum = z_Smiling_begin*(1-alpha[index])+z_Smiling_end*(alpha[index])
  #       z_Smiling=np.vstack((z_Smiling,dum))
  #     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
  #     save_images(samples, [image_frame_dim, image_frame_dim], './samplesSmilingRandom_z/test_arange_%s.png' % (idx))



  elif option == 4:
    p = 0.416754
        # randomly sample male vector
    if not os.path.exists(config.sample_dir+'Male'):
      os.makedirs(config.sample_dir+'Male')
      ## Following is changing each coordinate at a time
    alpha = np.linspace(0,1,image_frame_dim)

    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-0.5, 0.5, size=(image_frame_dim, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Smiling = np.random.uniform(-2*(1-pSmiling), 2*pSmiling, size=(image_frame_dim,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
      #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
      z_Young = np.random.uniform(-2*(1-pYoung), 2*pYoung, size=(image_frame_dim,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim,1])
      #z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      z_Male = np.random.uniform(-2*(1-pMale), 2*pMale, size=(image_frame_dim,dcgan.MaleDim))
      z_Male = np.tile(z_Male,[image_frame_dim,1])

      #dcgan.intervene_on = 'Male' # I don't know why setting this here didn't work
      c_Male_begin = 2*p*np.ones((image_frame_dim,1))
      c_Male_end = -2*(1-p)*np.ones((image_frame_dim,1)) 
      c_Male = c_Male_begin
      for index in range(1,image_frame_dim):
        dum = c_Male_begin*(1-alpha[index])+c_Male_end*(alpha[index])
        c_Male=np.vstack((c_Male,dum))
      #samples = sess.run(dcgan.sampler_male, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling, dcgan.intervention:c_Male})
      # my_dict={}
      # my_dict["Male"] = c_Male
      # my_set[0]=1
      # my_intervention[:,0] = c_Male.reshape((config.batch_size,))
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, \
        dcgan.zSmiling:z_Smiling, dcgan.h2Male:c_Male})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesMale/test_arange_%s.png' % (idx))

  elif option == 5:
    p = 0.773617
        # randomly sample male vector
    if not os.path.exists(config.sample_dir+'Young'):
      os.makedirs(config.sample_dir+'Young')
      ## Following is changing each coordinate at a time
    alpha = np.linspace(0,1,image_frame_dim)
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-0.5, 0.5, size=(image_frame_dim, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Smiling = np.random.uniform(-2*(1-pSmiling), 2*pSmiling, size=(image_frame_dim,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
      #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
      z_Young = np.random.uniform(-2*(1-pYoung), 2*pYoung, size=(image_frame_dim,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim,1])
      #z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      z_Male = np.random.uniform(-2*(1-pMale), 2*pMale, size=(image_frame_dim,dcgan.MaleDim))
      z_Male = np.tile(z_Male,[image_frame_dim,1])

      #z_Young = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      #dcgan.intervene_on = 'Young'
      c_Young_begin = 2*p*np.ones((image_frame_dim,1)) # Need to use mean and variance of the fake label for the range to sweep
      c_Young_end = -2*(1-p)*np.ones((image_frame_dim,1)) #10*0.2 before
      c_Young = c_Young_begin
      for index in range(1,image_frame_dim):
        dum = c_Young_begin*(1-alpha[index])+c_Young_end*(alpha[index])
        c_Young=np.vstack((c_Young,dum))
      #samples = sess.run(dcgan.sampler_young, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling, dcgan.intervention:c_Young})
      # my_dict = {}
      # my_dict["Young"] = c_Young
      # my_set[1]=1
      # my_intervention[:,1] = c_Young 
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, \
        dcgan.zSmiling:z_Smiling, dcgan.h2Young: c_Young})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesYoung/test_arange_%s.png' % (idx))

  elif option == 6:
    p = 0.482080
        # randomly sample male vector
    if not os.path.exists(config.sample_dir+'Smiling'):
      os.makedirs(config.sample_dir+'Smiling')
      ## Following is changing each coordinate at a time
    alpha = np.linspace(0,1,image_frame_dim)
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-0.5, 0.5, size=(image_frame_dim, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim,1])
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Smiling = np.random.uniform(-2*(1-pSmiling), 2*pSmiling, size=(image_frame_dim,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim,1])
      #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
      z_Young = np.random.uniform(-2*(1-pYoung), 2*pYoung, size=(image_frame_dim,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim,1])
      #z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      z_Male = np.random.uniform(-2*(1-pMale), 2*pMale, size=(image_frame_dim,dcgan.MaleDim))
      z_Male = np.tile(z_Male,[image_frame_dim,1])

      #z_Smiling = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      #dcgan.intervene_on = 'Smiling'
      c_Smiling_begin = 2*p*np.ones((image_frame_dim,1))
      c_Smiling_end = -2*(1-p)*np.ones((image_frame_dim,1))
      c_Smiling = c_Smiling_begin
      for index in range(1,image_frame_dim):
        dum = c_Smiling_begin*(1-alpha[index])+c_Smiling_end*(alpha[index])
        c_Smiling=np.vstack((c_Smiling,dum))
      #samples = sess.run(dcgan.sampler_smiling, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling, dcgan.intervention:c_Smiling})
      # my_dict={}
      # my_dict["Smiling"] = c_Smiling
      # my_set[2]=1
      # my_intervention[:,2] = c_Smiling
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, \
        dcgan.zSmiling:z_Smiling, dcgan.h2Smiling:c_Smiling})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesSmiling/test_arange_%s.png' % (idx))





  elif option == 7:
        # randomly sample male vector
    if not os.path.exists(config.sample_dir+'AllRandom_z'):
      os.makedirs(config.sample_dir+'AllRandom_z')
      ## Following is changing each coordinate at a time
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-0.5, 0.5, size=(image_frame_dim**2, dcgan.z_gen_dim))
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Smiling = np.random.uniform(-2*(1-pSmiling), 2*pSmiling, size=(image_frame_dim**2,dcgan.SmilingDim))
      z_Young = np.random.uniform(-2*(1-pYoung), 2*pYoung, size=(image_frame_dim**2,dcgan.YoungDim))
      z_Male = np.random.uniform(-2*(1-pMale), 2*pMale, size=(image_frame_dim**2,dcgan.MaleDim))
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesAllRandom_z/test_arange_%s.png' % (idx))

  elif option == 8:
        # randomly sample male vector
    p = 0.416754
    if not os.path.exists(config.sample_dir+'MaleFineRange'):
      os.makedirs(config.sample_dir+'MaleFineRange')
      ## Following is changing each coordinate at a time
    alpha = np.linspace(0,1,image_frame_dim**2)

    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-0.5, 0.5, size=(1, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Smiling = np.random.uniform(-2*(1-pSmiling), 2*pSmiling, size=(1,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim**2,1])
      #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
      z_Young = np.random.uniform(-2*(1-pYoung), 2*pYoung, size=(1,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim**2,1])
      #z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      z_Male = np.random.uniform(-2*(1-pMale), 2*pMale, size=(1,dcgan.MaleDim))
      z_Male = np.tile(z_Male,[image_frame_dim**2,1])

      #dcgan.intervene_on = 'Male' # I don't know why setting this here didn't work
      c_Male_begin = 2*p*np.ones((1,1))
      c_Male_end = -2*(1-p)*np.ones((1,1)) 
      c_Male = c_Male_begin
      for index in range(1,image_frame_dim**2):
        dum = c_Male_begin*(1-alpha[index])+c_Male_end*(alpha[index])
        c_Male=np.vstack((c_Male,dum))
      #samples = sess.run(dcgan.sampler_male, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling, dcgan.intervention:c_Male})
      # my_dict={}
      # my_dict["Male"] = c_Male
      # my_set[0]=1
      # my_intervention[:,0] = c_Male
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, \
        dcgan.zSmiling:z_Smiling, dcgan.h2Male:c_Male})      
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesMaleFineRange/test_arange_%s.png' % (idx))

  elif option == 9:
    p = 0.773617
    if not os.path.exists(config.sample_dir+'YoungFineRange'):
      os.makedirs(config.sample_dir+'YoungFineRange')
      ## Following is changing each coordinate at a time
    alpha = np.linspace(0,1,image_frame_dim**2)
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-0.5, 0.5, size=(1, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Smiling = np.random.uniform(-2*(1-pSmiling), 2*pSmiling, size=(1,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim**2,1])
      #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
      z_Young = np.random.uniform(-2*(1-pYoung), 2*pYoung, size=(1,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim**2,1])
      #z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      z_Male = np.random.uniform(-2*(1-pMale), 2*pMale, size=(1,dcgan.MaleDim))
      z_Male = np.tile(z_Male,[image_frame_dim**2,1])

      #z_Young = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      #dcgan.intervene_on = 'Young'
      c_Young_begin = 2*p*np.ones((1,1)) # Need to use mean and variance of the fake label for the range to sweep
      c_Young_end = -2*(1-p)*np.ones((1,1)) #10*0.2 before
      c_Young = c_Young_begin
      for index in range(1,image_frame_dim**2):
        dum = c_Young_begin*(1-alpha[index])+c_Young_end*(alpha[index])
        c_Young=np.vstack((c_Young,dum))
      #samples = sess.run(dcgan.sampler_young, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling, dcgan.intervention:c_Young})
      # my_dict = {}
      # my_dict["Young"] = c_Young
      # my_set[1]=1
      # my_intervention[:,1] = c_Young 
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, \
        dcgan.zSmiling:z_Smiling, dcgan.h2Young:c_Young})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesYoungFineRange/test_arange_%s.png' % (idx))

  elif option == 10:
    p = 0.482080
    if not os.path.exists(config.sample_dir+'SmilingFineRange'):
      os.makedirs(config.sample_dir+'SmilingFineRange')
      ## Following is changing each coordinate at a time
    alpha = np.linspace(0,1,image_frame_dim**2)
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_gen_sample = np.random.uniform(-0.5, 0.5, size=(1, dcgan.z_gen_dim))
      z_gen_sample = np.tile(z_gen_sample,[image_frame_dim**2,1])
      #z_Male = np.random.uniform(-0.5, 0.5, size=(config.batch_size,dcgan.MaleDim))
      #z_Young=np.random.uniform(-1, 1, size=(config.batch_size,dcgan.YoungDim))
      z_Smiling = np.random.uniform(-2*(1-pSmiling), 2*pSmiling, size=(1,dcgan.SmilingDim))
      z_Smiling = np.tile(z_Smiling,[image_frame_dim**2,1])
      #z_Smiling = np.random.uniform(-1, 1, size=(config.batch_size,dcgan.SmilingDim))
      z_Young = np.random.uniform(-2*(1-pYoung), 2*pYoung, size=(1,dcgan.YoungDim))
      z_Young = np.tile(z_Young,[image_frame_dim**2,1])
      #z_Male = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      z_Male = np.random.uniform(-2*(1-pMale), 2*pMale, size=(1,dcgan.MaleDim))
      z_Male = np.tile(z_Male,[image_frame_dim**2,1])

      #z_Smiling = np.random.uniform(-1,1,size=(image_frame_dim**2,dcgan.SmilingDim))
      #dcgan.intervene_on = 'Smiling'
      c_Smiling_begin = 2*p*np.ones((1,1))
      c_Smiling_end = -2*(1-p)*np.ones((1,1))
      c_Smiling = c_Smiling_begin
      for index in range(1,image_frame_dim**2):
        dum = c_Smiling_begin*(1-alpha[index])+c_Smiling_end*(alpha[index])
        c_Smiling=np.vstack((c_Smiling,dum))
      #samples = sess.run(dcgan.sampler_smiling, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, dcgan.zSmiling:z_Smiling, dcgan.intervention:c_Smiling})
      #my_dict={}
      #my_dict["Smiling"] = c_Smiling
      # my_set[2]=1
      # my_intervention[:,2] = c_Smiling
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, \
        dcgan.zSmiling:z_Smiling, dcgan.h2Smiling:c_Smiling})
      save_images(samples, [image_frame_dim, image_frame_dim], './samplesSmilingFineRange/test_arange_%s.png' % (idx))

  elif option == 11:
    causal_controller = dcgan.cc
    means = dcgan.means
    alpha = np.linspace(0,1,image_frame_dim)
    for node in causal_controller.nodes:
      p = means[node.name]
      if dcgan.label_specific_noise:
        int_begin = 2*p*np.ones((1,1))
        int_end = -2*(1-p)*np.ones((1,1))
      else:
        r = 1.386
        int_begin = r
        int_end = -r
      intervention = int_begin
      for index in range(1,image_frame_dim):
        dum = int_begin*(1-alpha[index])+int_end*(alpha[index])
        intervention=np.vstack((intervention,dum))
      intervention = np.repeat(intervention, image_frame_dim ,axis = 0)

      if not os.path.exists(config.sample_dir+node.name):
        os.makedirs(config.sample_dir+node.name)
        ## Following is changing each coordinate at a time
      # def indx(i,j,dim):
      #   return i*dim + j 
      for idx in xrange(10):
        print(" [*] %d" % idx)
        z_sample = dcgan.sess.run(dcgan.z_fd)
        z_sample = {key:np.tile(val[:image_frame_dim],[image_frame_dim,1]) for key,val in z_sample.items()}
        fd={dcgan.z_fd[k]:val for k,val in z_sample.items()}
        # for nodeAll in causal_controller.nodes:
        #   p = means[nodeAll.name]
        #   nodeIntervention = np.random.uniform(-2*(1-p), 2*p, size=(1,1))
        #   nodeIntervention = np.tile(nodeIntervention,[image_frame_dim**2,1])
        #   fd.update({nodeAll.label_logit:nodeIntervention})
        fd.update({node.label_logit: intervention})
        samples = sess.run(dcgan.G,feed_dict = fd)  
        save_images(samples, [image_frame_dim, image_frame_dim], './samples'+str(node.name)+'/test_arange_%s.png' % (idx))
        # sess.run(self.cc.Beard.label, {self.cc.Male.label_logit:0.7*np.ones([64,1]),   self.z_gen: sample_z['z_gen']} )
        # samples = sess.run(dcgan.sampler, feed_dict={dcgan.z_gen: z_gen_sample, dcgan.zMale:z_Male, dcgan.zYoung:z_Young, \
        #   dcgan.zSmiling:z_Smiling, dcgan.h2Male:c_Male})
        # save_images(samples, [image_frame_dim, image_frame_dim], './samplesMale/test_arange_%s.png' % (idx))















