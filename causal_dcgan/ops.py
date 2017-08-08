import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        f1=0.5 * (1+leak)
        f2=0.5 * (1-leak)
        return f1*x + f2*tf.abs(x)

#This takes more memory than above
#def lrelu(x, leak=0.2, name="lrelu"):
#  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias



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

