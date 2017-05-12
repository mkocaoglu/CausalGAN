import numpy as np
import tensorflow as tf
slim = tf.contrib.slim





class CausalController(object):
    def __init__(self, graph,batch_size,indep_causal=False,n_hidden=10, train = True):
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


            Pass indep_causal=True to use Unif[0,1] labels


            Access nodes ether with:
            model.cc.node_dict['Male']
            or with:
            model.cc.Male

        '''
        with tf.variable_scope('causal_controller') as vs:
            self.graph=graph
            self.n_hidden=n_hidden
            self.train = train
            if indep_causal:
                NodeClass=UniformNode
            else:
                NodeClass=CausalNode

            NodeClass.batch_size=batch_size
            self.node_names, self.parent_names=zip(*graph)
            self.nodes=[NodeClass(name=n,train=self.train) for n in self.node_names]

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

    Uniform[-1,1] + other causes pases through 2 fully connected layers. 
    '''
    train = True
    batch_size=1#class variable. set all at once
    name=None
    #logit is going to be 1 dim with sigmoid
    #as opposed to 2 dim with softmax
    _label_logit=None
    _label=None
    parents=[]#list of CausalNodes

    def __init__(self,train=True,name=None,n_hidden=10):
        self.name=name
        self.n_hidden=n_hidden#also is z_dim
        self.train = train

        #Use tf.random_uniform instead of placeholder for noise
        n=self.batch_size*self.n_hidden
        #print 'CN n',n
    def setup_tensor(self):
        with tf.variable_scope(self.name):
            if self.train:
                self.z = tf.random_uniform(
                        (self.batch_size, self.n_hidden), minval=-1.0, maxval=1.0)
            else:
                self.z = tf.random_uniform(
                        (self.batch_size, self.n_hidden), minval=-0.5, maxval=0.5)                
            tf_parents=[self.z]+[node.label_logit for node in self.parents]
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

