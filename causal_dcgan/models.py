import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import math

from ops import lrelu,linear,conv_cond_concat,batch_norm,add_minibatch_features

from ops import conv2d,deconv2d


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def GeneratorCNN( z, config, reuse=None):
    '''
    maps z to a 64x64 images with values in [-1,1]
    uses batch normalization internally
    '''

    #trying to get around batch_size like this:
    batch_size=tf.shape(z)[0]
    #batch_size=tf.placeholder_with_default(64,[],'bs')

    with tf.variable_scope("generator",reuse=reuse) as vs:
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')

        s_h, s_w = config.gf_dim, config.gf_dim#64,64
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)



        # project `z` and reshape
        z_, self_h0_w, self_h0_b = linear(
            z, config.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self_h0 = tf.reshape(
            z_, [-1, s_h16, s_w16, config.gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(self_h0))

        h1, h1_w, h1_b = deconv2d(
            h0, [batch_size, s_h8, s_w8, config.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))

        h2, h2_w, h2_b = deconv2d(
            h1, [batch_size, s_h4, s_w4, config.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))

        h3, h3_w, h3_b = deconv2d(
            h2, [batch_size, s_h2, s_w2, config.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))

        h4, h4_w, h4_b = deconv2d(
            h3, [batch_size, s_h, s_w, config.c_dim], name='g_h4', with_w=True)
        out=tf.nn.tanh(h4)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def DiscriminatorCNN(image, config, reuse=None):
    '''
    Discriminator for GAN model.

    image      : batch_size x 64x64x3 image
    config     : see causal_dcgan/config.py
    reuse      : pass True if not calling for first time

    returns: probabilities(real)
           : logits(real)
           : first layer activation used to estimate z from
           : variables list
    '''
    with tf.variable_scope("discriminator",reuse=reuse) as vs:
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')
        d_bn3 = batch_norm(name='d_bn3')

        if not config.stab_proj:
            h0 = lrelu(conv2d(image, config.df_dim, name='d_h0_conv'))#16,32,32,64

        else:#method to restrict disc from winning
            #I think this is equivalent to just not letting disc optimize first layer
            #and also removing nonlinearity

            #k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
            #paper used 8x8 kernel, but I'm using 5x5 because it is more similar to my achitecture
            #n_projs=config.df_dim#64 instead of 32 in paper
            n_projs=config.n_stab_proj#64 instead of 32 in paper

            print("WARNING:STAB_PROJ active, using ",n_projs," projections")

            w_proj = tf.get_variable('w_proj', [5, 5, image.get_shape()[-1],n_projs],
                initializer=tf.truncated_normal_initializer(stddev=0.02),trainable=False)
            conv = tf.nn.conv2d(image, w_proj, strides=[1, 2, 2, 1], padding='SAME')

            b_proj = tf.get_variable('b_proj', [n_projs],#does nothing
                 initializer=tf.constant_initializer(0.0),trainable=False)
            h0=tf.nn.bias_add(conv,b_proj)


        h1_ = lrelu(d_bn1(conv2d(h0, config.df_dim*2, name='d_h1_conv')))#16,16,16,128

        h1 = add_minibatch_features(h1_, config.df_dim)
        h2 = lrelu(d_bn2(conv2d(h1, config.df_dim*4, name='d_h2_conv')))#16,16,16,248
        h3 = lrelu(d_bn3(conv2d(h2, config.df_dim*8, name='d_h3_conv')))
        #print('h3shape: ',h3.get_shape().as_list())
        #print('8df_dim:',config.df_dim*8)
        #dim3=tf.reduce_prod(tf.shape(h3)[1:])
        dim3=np.prod(h3.get_shape().as_list()[1:])
        h3_flat=tf.reshape(h3, [-1,dim3])
        h4 = linear(h3_flat, 1, 'd_h3_lin')

        prob=tf.nn.sigmoid(h4)

        variables = tf.contrib.framework.get_variables(vs,collection=tf.GraphKeys.TRAINABLE_VARIABLES)

    return prob, h4, h1_, variables


def discriminator_labeler(image, output_dim, config, reuse=None):
    batch_size=tf.shape(image)[0]
    with tf.variable_scope("disc_labeler",reuse=reuse) as vs:
        dl_bn1 = batch_norm(name='dl_bn1')
        dl_bn2 = batch_norm(name='dl_bn2')
        dl_bn3 = batch_norm(name='dl_bn3')

        h0 = lrelu(conv2d(image, config.df_dim, name='dl_h0_conv'))#16,32,32,64
        h1 = lrelu(dl_bn1(conv2d(h0, config.df_dim*2, name='dl_h1_conv')))#16,16,16,128
        h2 = lrelu(dl_bn2(conv2d(h1, config.df_dim*4, name='dl_h2_conv')))#16,16,16,248
        h3 = lrelu(dl_bn3(conv2d(h2, config.df_dim*8, name='dl_h3_conv')))
        dim3=np.prod(h3.get_shape().as_list()[1:])
        h3_flat=tf.reshape(h3, [-1,dim3])
        D_labels_logits = linear(h3_flat, output_dim, 'dl_h3_Label')
        D_labels = tf.nn.sigmoid(D_labels_logits)
        variables = tf.contrib.framework.get_variables(vs)
    return D_labels, D_labels_logits, variables

def discriminator_gen_labeler(image, output_dim, config, reuse=None):
    batch_size=tf.shape(image)[0]
    with tf.variable_scope("disc_gen_labeler",reuse=reuse) as vs:
        dl_bn1 = batch_norm(name='dl_bn1')
        dl_bn2 = batch_norm(name='dl_bn2')
        dl_bn3 = batch_norm(name='dl_bn3')

        h0 = lrelu(conv2d(image, config.df_dim, name='dgl_h0_conv'))#16,32,32,64
        h1 = lrelu(dl_bn1(conv2d(h0, config.df_dim*2, name='dgl_h1_conv')))#16,16,16,128
        h2 = lrelu(dl_bn2(conv2d(h1, config.df_dim*4, name='dgl_h2_conv')))#16,16,16,248
        h3 = lrelu(dl_bn3(conv2d(h2, config.df_dim*8, name='dgl_h3_conv')))
        dim3=np.prod(h3.get_shape().as_list()[1:])
        h3_flat=tf.reshape(h3, [-1,dim3])
        D_labels_logits = linear(h3_flat, output_dim, 'dgl_h3_Label')
        D_labels = tf.nn.sigmoid(D_labels_logits)
        variables = tf.contrib.framework.get_variables(vs)
    return D_labels, D_labels_logits,variables

def discriminator_on_z(image, config, reuse=None):
    batch_size=tf.shape(image)[0]
    with tf.variable_scope("disc_z_labeler",reuse=reuse) as vs:
        dl_bn1 = batch_norm(name='dl_bn1')
        dl_bn2 = batch_norm(name='dl_bn2')
        dl_bn3 = batch_norm(name='dl_bn3')

        h0 = lrelu(conv2d(image, config.df_dim, name='dzl_h0_conv'))#16,32,32,64
        h1 = lrelu(dl_bn1(conv2d(h0, config.df_dim*2, name='dzl_h1_conv')))#16,16,16,128
        h2 = lrelu(dl_bn2(conv2d(h1, config.df_dim*4, name='dzl_h2_conv')))#16,16,16,248
        h3 = lrelu(dl_bn3(conv2d(h2, config.df_dim*8, name='dzl_h3_conv')))
        dim3=np.prod(h3.get_shape().as_list()[1:])
        h3_flat=tf.reshape(h3, [-1,dim3])
        D_labels_logits = linear(h3_flat, config.z_dim, 'dzl_h3_Label')
        D_labels = tf.nn.tanh(D_labels_logits)
        variables = tf.contrib.framework.get_variables(vs)
    return D_labels,variables



