import tensorflow as tf
slim = tf.contrib.slim
import math

from ops import lrelu,linear,conv_cond_concat,batch_norm,add_minibatch_features

from ops import conv2d


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def GeneratorCNN( z, config, reuse=None):
    '''
    maps z to a 64x64 images with values in [-1,1]
    uses batch normalization internally
    '''

    #trying to get around batch_size like this:
    batch_size=z.shape[0]

    with tf.variable_scope("generator",reuse=reuse) as vs:
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')

        s_h, s_w = config.gf_dim, self.config.gf_dim#64,64
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

def DiscriminatorCNN(X, config, reuse=None):
    '''
    Discriminator for GAN model.

    X      : batch_size x 64x64x3 image
    config : see causal_dcgan/config.py
    reuse  : pass True if not calling for first time

    returns: probabilities(real)
           : logits(real)
           : first layer activation used to estimate z from
           : variables list
    '''

    batch_size=X.shape[0]


    with tf.variable_scope("discriminator",reuse=reuse) as vs:
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')
        d_bn3 = batch_norm(name='d_bn3')

        h0 = lrelu(conv2d(image, config.df_dim, name='d_h0_conv'))#16,32,32,64
        h1_ = lrelu(d_bn1(conv2d(h0, config.df_dim*2, name='d_h1_conv')))#16,16,16,128
        h1 = add_minibatch_features(h1_, config.df_dim, batch_size)
        h2 = lrelu(d_bn2(conv2d(h1, config.df_dim*4, name='d_h2_conv')))#16,16,16,248
        h3 = lrelu(d_bn3(conv2d(h2, config.df_dim*8, name='d_h3_conv')))
        h3_flat=tf.reshape(h3, [batch_size, -1])
        h4 = linear(h3_flat, 1, 'd_h3_lin')

        prob=tf.nn.sigmoid(h4)
        variables = tf.contrib.framework.get_variables(vs)

    return prob, h4, h1_, variables



