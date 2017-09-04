import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        f1=0.5 * (1+leak)
        f2=0.5 * (1-leak)
        return f1*x + f2*tf.abs(x)

def GeneratorCNN( z, config, reuse=None):
    hidden_num=config.conv_hidden_num
    output_num=config.c_dim
    repeat_num=config.repeat_num
    data_format=config.data_format

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

def DiscriminatorCNN(image, config, reuse=None):
    hidden_num=config.conv_hidden_num
    data_format=config.data_format
    input_channel=config.channel

    with tf.variable_scope("D",reuse=reuse) as vs:
        # Encoder
        with tf.variable_scope('encoder'):
            x = slim.conv2d(image, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                            data_format=data_format,scope='conv0')

            prev_channel_num = hidden_num
            for idx in range(config.repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx+1)+'a')
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx+1)+'b')
                if idx < config.repeat_num - 1:
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu,
                                    data_format=data_format,scope='conv'+str(idx+1)+'c')
                    #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            z = x = slim.fully_connected(x, config.z_num, activation_fn=None,scope='proj')

        # Decoder
        with tf.variable_scope('decoder'):
            x = slim.fully_connected(x, np.prod([8, 8, hidden_num]), activation_fn=None)
            x = reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(config.repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx)+'a')
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu,
                                data_format=data_format,scope='conv'+str(idx)+'b')
                if idx < config.repeat_num - 1:
                    x = upscale(x, 2, data_format)
            out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None,
                              data_format=data_format,scope='proj')

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables


def Discriminator_labeler(image, output_size, config, reuse=None):
    hidden_num=config.conv_hidden_num
    repeat_num=config.repeat_num
    data_format=config.data_format
    with tf.variable_scope("discriminator_labeler",reuse=reuse) as scope:

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

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        #Isn't this backward?
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

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




