import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        f1=0.5 * (1+leak)
        f2=0.5 * (1-leak)
        return f1*x + f2*tf.abs(x)

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format,reuse=None):
    with tf.variable_scope("G",reuse=reuse) as vs:
        x = slim.fully_connected(z, np.prod([8, 8, hidden_num]),activation_fn=None,scope='fc1')
        x = reshape(x, 8, 8, hidden_num, data_format)

        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format,scope='conv'+str(idx)+'a')
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format,scope='conv'+str(idx)+'b')
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, 3, 3, 1, activation_fn=None,data_format=data_format,scope='conv'+str(idx+1))

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format,reuse=None):
    with tf.variable_scope("D") as vs:
        # Encoder
        with tf.variable_scope('encoder',reuse=reuse):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format,scope='conv0')

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx+1)+'a')
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx+1)+'b')
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu,
                                    data_format=data_format,scope='conv'+str(idx+1)+'c')
                    #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            z = x = slim.fully_connected(x, z_num, activation_fn=None,scope='proj')

        # Decoder
        with tf.variable_scope('decoder',reuse=reuse):
            x = slim.fully_connected(x, np.prod([8, 8, hidden_num]), activation_fn=None)
            x = reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx)+'a')
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx)+'b')
                if idx < repeat_num - 1:
                    x = upscale(x, 2, data_format)
            out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None,
                              data_format=data_format,scope='proj')

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables

def Discriminator_CC(labels,batch_size, reuse=None, n_hidden=10):
    #If you make the scope DCC instead of dcc, it will get matched with Disc scope
    with tf.variable_scope("dcc",reuse=reuse) as scope:
        def add_minibatch_features_for_labels(labels,batch_size):
            with tf.variable_scope('minibatch'):
                n_kernels = 50
                dim_per_kernel = 20
                shape = labels.get_shape().as_list()
                dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
                input_ = tf.reshape(labels, [-1, dim])           # -1 means "all"  
                #x = linear(input_, n_kernels * dim_per_kernel,'d_mbLabelLinear')
                x =slim.fully_connected(input_, n_kernels * dim_per_kernel,
                                activation_fn=None,scope='d_mbLabelLinear')
                activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))
                big = np.zeros((batch_size, batch_size), dtype='float32')
                big += np.eye(batch_size)
                big = tf.expand_dims(big, 1)

                abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
                mask = 1. - big
                masked = tf.exp(-abs_dif) * mask
                f1 = tf.reduce_sum(masked, 2) / tf.reduce_sum(mask)

                minibatch_features = tf.concat([labels, f1],1)
                return minibatch_features

        h0 = slim.fully_connected(labels,n_hidden,activation_fn=lrelu,scope='layer0')
        h1 = slim.fully_connected(h0,n_hidden,activation_fn=lrelu,scope='layer1')
        h1_aug = lrelu(add_minibatch_features_for_labels(h1,batch_size))
        h2 = slim.fully_connected(h1_aug,n_hidden,activation_fn=lrelu,scope='layer2')
        #print('WARNING: using n_hidden for im disc_CC output')
        #h3 = slim.fully_connected(h2,n_hidden,activation_fn=None,scope='layer3')
        h3 = slim.fully_connected(h2,1,activation_fn=None,scope='layer3')
    variables = tf.contrib.framework.get_variables(scope)

    return tf.nn.sigmoid(h3), h3, variables


def DiscriminatorW(labels,batch_size, n_hidden, config, reuse=None):
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
    batch_size=config.batch_size
    LAMBDA=config.lambda_W
    n_hidden=config.critic_hidden_size
    #gradient penalty "Improved training of Wasserstein"
    alpha = tf.random_uniform([batch_size,1],0.,1.)
    interpolates = alpha*real_data + ((1-alpha)*fake_data)#Could do more if not fixed batch_size
    disc_interpolates = Discriminator(interpolates,batch_size,n_hidden=n_hidden,config=config, reuse=True)[1]#logits
    gradients = tf.gradients(disc_interpolates,[interpolates])[0]#orig
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                           reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
    grad_cost = LAMBDA*gradient_penalty
    return grad_cost,slopes

def FactorizedNetwork(graph,Net,config):
    node_names, parent_names=zip(*graph)
    def fDCC(real_labels, fake_labels, batch_size, n_hidden=10):
        with tf.variable_scope('gfactorized'):
            list_real_labels=tf.unstack(real_labels,axis=1)
            list_fake_labels=tf.unstack(fake_labels,axis=1)
            real_label_dict={n:l for n,l in zip(node_names,list_real_labels)}
            fake_label_dict={n:l for n,l in zip(node_names,list_fake_labels)}
            real_parent_inputs=[ [real_label_dict[n] for n in p] for p in parent_names]
            fake_parent_inputs=[ [fake_label_dict[n] for n in p] for p in parent_names]
            real_inputs=[tf.stack( [real_label_dict[n]]+par,axis=1) for n,par in zip(node_names,real_parent_inputs)]
            fake_inputs=[tf.stack( [fake_label_dict[n]]+par,axis=1) for n,par in zip(node_names,fake_parent_inputs)]

            #dcc_dict={}
            dcc_dict={'real_prob':{},'real_logit':{},
                      'fake_prob':{},'fake_logit':{},
                      'var':{},'grad_cost':{},'slopes':{}}
            logit_sum=0
            list_logits=[]
            net_var=[]
            for n,rx,fx in zip(node_names,real_inputs,fake_inputs):
                with tf.variable_scope(n):
                    prob,log,var=Net(rx,batch_size,n_hidden,config,reuse=False)
                    dcc_dict['real_prob'][n]=prob
                    dcc_dict['real_logit'][n]=log
                    dcc_dict['var'][n]=var

                    prob,log,_  =Net(fx,batch_size,n_hidden,config,reuse=True)
                    dcc_dict['fake_prob'][n]=prob
                    dcc_dict['fake_logit'][n]=log

                    list_logits.append(log)
                    logit_sum+=log
                    net_var+=var

                    grad_cost,slopes=Grad_Penalty(rx,fx,Net,config)
                    dcc_dict['grad_cost'][n]=grad_cost
                    dcc_dict['slopes']=slopes

            logit_sum/=len(list_logits)#optional
            prob=tf.nn.sigmoid(logit_sum)
        return dcc_dict
    return fDCC




def Discriminator_labeler(image, output_size, repeat_num, hidden_num, data_format):
    with tf.variable_scope("discriminator_labeler") as scope:

        x = slim.conv2d(image, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                        data_format=data_format,scope='conv0')

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format,scope='conv'+str(idx+1)+'a')
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format,scope='conv'+str(idx+1)+'b')
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx+1)+'c')
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        label_logit = slim.fully_connected(x, output_size, activation_fn=None,scope='proj')

        variables = tf.contrib.framework.get_variables(scope)
        return label_logit,variables


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)



#https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L168
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples.
    The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers.  So ..  we will just return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads




