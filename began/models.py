import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        f1=0.5 * (1+leak)
        f2=0.5 * (1-leak)
        return f1*x + f2*tf.abs(x)


class CausalController(object):
    def __init__(self, graph,batch_size,indep_causal=False,n_hidden=10):
        '''a causal graph is specified as follows:
            just supply a list of pairs (node, node_parents)

            Example: A->B<-C; D->E

            [ ['A',[]],
              ['B',['A','C']],
              ['C',[]],
              ['D',[]],
              ['E',['D']]
            ]

            I use a list right now instead of a dict because I don't think
            dict.keys() are gauranteed to be returned a particular order.
            TODO:A good improvement would be to use collections.OrderedDict
        '''
        with tf.variable_scope('CC') as vs:
            self.graph=graph
            self.n_hidden=n_hidden

            if indep_causal:
                NodeClass=UniformNode
            else:
                NodeClass=CausalNode

            NodeClass.batch_size=batch_size
            self.node_names, self.parent_names=zip(*graph)
            self.nodes=[NodeClass(name=n) for n in self.node_names]

            #={n:CausalNode(n) for n in self.node_names}
            for node,rents in zip(self.nodes,self.parent_names):
                node.parents=[n for n in self.nodes if n.name in rents]
            self.node_dict={n.name:n for n in self.nodes}

            ##construct graph##
            for node in self.nodes:
                node.setup_tensor()

        #enable access directly. Little dangerous
        self.__dict__.update(self.node_dict)

        self.var = tf.contrib.framework.get_variables(vs)

    @property
    def feed_z(self):#might have to makethese phw/default
        return {key:val.z for key,val in self.node_dict.iteritems()}
    @property
    def sample_z(self):
        return {key:val.z for key,val in self.node_dict.iteritems()}

    def list_placeholders(self):
        return [n.z for n in self.nodes]
    def list_labels(self):
        return [n.label for n in self.nodes]
    def list_label_logits(self):
        return [n.label_logit for n in self.nodes]


class CausalNode(object):
    '''
    A CausalNode sets up a small neural network:
    z_noise+[,other causes] -> label_logit

    Everything is defined in terms of @property
    to allow tensorflow graph to be lazily generated as called
    because I don't enforce that a node's parent tf_graph
    is constructed during class.setup_tensor
    '''
    batch_size=1#class variable. set all at once
    name=None
    #logit is going to be 1 dim with sigmoid
    #as opposed to 2 dim with softmax
    _label_logit=None
    _label=None
    parents=[]#list of CausalNodes
    def __init__(self,name=None,n_hidden=10):
        self.name=name
        self.n_hidden=n_hidden#also is z_dim

        #Use tf.random_uniform instead of placeholder for noise
        n=self.batch_size*self.n_hidden
        #print 'CN n',n
        with tf.variable_scope(self.name):
            self.z = tf.random_uniform(
                    (self.batch_size, self.n_hidden), minval=-1.0, maxval=1.0)
    def setup_tensor(self):
        if self._label is not None:#already setup
            return
        tf_parents=[self.z]+[node.label_logit for node in self.parents]
        with tf.variable_scope(self.name):
            vec_parents=tf.concat(tf_parents,-1)
            h0=slim.fully_connected(vec_parents,self.n_hidden,activation_fn=tf.nn.tanh,scope='layer0')
            h1=slim.fully_connected(h0,self.n_hidden,activation_fn=tf.nn.tanh,scope='layer1')
            self._label_logit = slim.fully_connected(h1,1,activation_fn=None,scope='proj')
            self._label=tf.nn.sigmoid( self._label_logit )

    @property
    def label_logit(self):
        if self._label_logit is not None:
            return self._label_logit
        else:
            self.setup_tensor()
            return self._label_logit
    @property
    def label(self):
        if self._label is not None:
            return self._label
        else:
            self.setup_tensor()
            return self._label

class UniformNode(CausalNode):
    def setup_tensor(self):
        self._label_logit=tf.constant(-1)
        self._label=tf.random_uniform((self.batch_size,1), minval=0.0, maxval=1.0)
        self._label_logit=self._label
        self.z=self._label

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format):
    with tf.variable_scope("G") as vs:
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

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        with tf.variable_scope('encoder'):
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
        with tf.variable_scope('decoder'):
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
        print('WARNING: using n_hidden for im disc_CC output')
        h3 = slim.fully_connected(h2,n_hidden,activation_fn=None,scope='layer3')

    variables = tf.contrib.framework.get_variables(scope)

    return tf.nn.sigmoid(h3), h3, variables

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




