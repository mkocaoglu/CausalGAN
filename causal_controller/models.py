import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        #Trick that saves memory by avoiding tf.max
        f1=0.5 * (1+leak)
        f2=0.5 * (1-leak)
        return f1*x + f2*tf.abs(x)


def DiscriminatorW(labels,batch_size, n_hidden, config, reuse=None):
    '''
    A simple discriminator to be used with Wasserstein optimization.
    No minibatch features or batch normalization is used.
    '''
    with tf.variable_scope("WasserDisc") as scope:
        if reuse:
            scope.reuse_variables()
        h=labels
        act_fn=lrelu
        n_neurons=n_hidden
        for i in range(config.critic_layers):
            if i==config.critic_layers-1:
                act_fn=None
                n_neurons=1
            scp='WD'+str(i)
            h = slim.fully_connected(h,n_neurons,activation_fn=act_fn,scope=scp)
        variables = tf.contrib.framework.get_variables(scope)
        return tf.nn.sigmoid(h),h,variables


def Grad_Penalty(real_data,fake_data,Discriminator,config):
    '''
    Implemention from "Improved training of Wasserstein"
    Interpolation based estimation of the gradient of the discriminator.
    Used to penalize the derivative rather than explicitly constrain lipschitz.
    '''
    batch_size=config.batch_size
    LAMBDA=config.lambda_W
    n_hidden=config.critic_hidden_size
    alpha = tf.random_uniform([batch_size,1],0.,1.)
    interpolates = alpha*real_data + ((1-alpha)*fake_data)#Could do more if not fixed batch_size
    disc_interpolates = Discriminator(interpolates,batch_size,n_hidden=n_hidden,config=config, reuse=True)[1]#logits
    gradients = tf.gradients(disc_interpolates,[interpolates])[0]#orig
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                           reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
    grad_cost = LAMBDA*gradient_penalty
    return grad_cost,slopes

