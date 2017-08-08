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


def discriminatorW(labels,hidden_size,reuse=False):
    with tf.variable_scope("Wdisc_CC") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = slim.fully_connected(labels,hidden_size,activation_fn=lrelu,scope='dCC_0')
        h1 = slim.fully_connected(h0,hidden_size,activation_fn=lrelu,scope='dCC_1')
        h2 = slim.fully_connected(h1,hidden_size,activation_fn=lrelu,scope='dCC_2')
        h3 = slim.fully_connected(h2,1,activation_fn=None,scope='dCC_3')
        return tf.nn.sigmoid(h3),h3


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




