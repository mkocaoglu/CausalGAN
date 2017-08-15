import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *



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
    #print('input x:',x.get_shape().as_list())
    #print('input y:',y.get_shape().as_list())

    xshape=x.get_shape()
    #tile by [1,64,64,1]

    tile_shape=tf.stack([1,xshape[1],xshape[2],1])
    tile_y=tf.tile(y,tile_shape)

    #print('tile y:',tile_y.get_shape().as_list())

    return tf.concat([x,tile_y],axis=3)


    #x_shapes = x.get_shape()
    #y_shapes = y.get_shape()
    #return tf.concat([
    #x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    conv=tf.nn.bias_add(conv,biases)

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        tf_output_shape=tf.stack(output_shape)
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=tf_output_shape,
                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf_output_shape)

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

    #mat_shape=tf.stack([tf.shape(input_)[1],output_size])
    mat_shape=[shape[1],output_size]

    with tf.variable_scope(scope or "Linear"):
        #matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
        matrix = tf.get_variable("Matrix", mat_shape, tf.float32,
                     tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                   initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


#minibatch method that improves on openai
#because it doesn't fix batchsize:
#TODO: recheck when not sleepy
def add_minibatch_features(image,df_dim):
    shape = image.get_shape().as_list()
    dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
    h_mb0 = lrelu(conv2d(image, df_dim, name='d_mb0_conv'))
    h_mb1 = conv2d(h_mb0, df_dim, name='d_mbh1_conv')

    dims=h_mb1.get_shape().as_list()
    conv_dims=np.prod(dims[1:])

    image_ = tf.reshape(h_mb1, tf.stack([-1, conv_dims]))
    #image_ = tf.reshape(h_mb1, tf.stack([batch_size, -1]))

    n_kernels = 300
    dim_per_kernel = 50
    x = linear(image_, n_kernels * dim_per_kernel,'d_mbLinear')
    act = tf.reshape(x, (-1, n_kernels, dim_per_kernel))

    act= tf.reshape(x, (-1, n_kernels, dim_per_kernel))
    act_tp=tf.transpose(act, [1,2,0])
    #bs x n_ker x dim_ker x bs -> bs x n_ker x bs :
    abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(act, 3) - tf.expand_dims(act_tp, 0)), 2)
    eye=tf.expand_dims( tf.eye( tf.shape(abs_dif)[0] ), 1)#bs x 1 x bs
    masked=tf.exp(-abs_dif) - eye
    f1=tf.reduce_mean( masked, 2)
    mb_features = tf.reshape(f1, [-1, 1, 1, n_kernels])
    return conv_cond_concat(image, mb_features)

## following is from https://github.com/openai/improved-gan/blob/master/imagenet/discriminator.py#L88
#def add_minibatch_features(image,df_dim,batch_size):
#    shape = image.get_shape().as_list()
#    dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
#    h_mb0 = lrelu(conv2d(image, df_dim, name='d_mb0_conv'))
#    h_mb1 = conv2d(h_mb0, df_dim, name='d_mbh1_conv')
#
#    dims=h_mb1.get_shape().as_list()
#    conv_dims=np.prod(dims[1:])
#
#    image_ = tf.reshape(h_mb1, tf.stack([-1, conv_dims]))
#    #image_ = tf.reshape(h_mb1, tf.stack([batch_size, -1]))
#
#    n_kernels = 300
#    dim_per_kernel = 50
#    x = linear(image_, n_kernels * dim_per_kernel,'d_mbLinear')
#    activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))
#    big = np.zeros((batch_size, batch_size), dtype='float32')
#    big += np.eye(batch_size)
#    big = tf.expand_dims(big, 1)
#    abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
#    mask = 1. - big
#    masked = tf.exp(-abs_dif) * mask
#    f1 = tf.reduce_sum(masked, 2) / tf.reduce_sum(mask)
#    mb_features = tf.reshape(f1, [batch_size, 1, 1, n_kernels])
#    return conv_cond_concat(image, mb_features)





