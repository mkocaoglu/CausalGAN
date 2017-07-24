import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from models import *
from label_loader import get_label_loader

debug=False


class CausalController(object):
    def load(self,sess,path):
        '''
        sess is a tf.Session object
        path is the path of the file you want to load, (not the directory)
        Example
        ./checkpoint/somemodel/saved/model.ckpt-3000
        (leave off the extensions)
        '''
        if not hasattr(self,'saver'):
            self.saver=tf.train.Saver(var_list=self.var)
        print('Attempting to load model:',path)
        self.saver.restore(sess,path)

    #def __init__(self,graph,batch_size=1,indep_causal=False,n_layers=3,n_hidden=10,input_dict=None,reuse=None):
    def __init__(self,graph,config,batch_size=1,input_dict=None,reuse=None):
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

            input_dict allows the model to take in some aritrary input instead
            of using tf_random_uniform nodes

            #It should be a dictionary {name:var}, where name corresponds to
            the node names at play

            pass reuse if constructing for a second time
        '''


        indep_causal=self.config.indep_causal
        n_layers=self.config.cc_n_layers
        n_hidden=self.config.cc_n_hidden
        self.config=config

        if reuse:
            assert input_dict is not None

        with tf.variable_scope('causal_controller',reuse=reuse) as vs:
            self.graph=graph
            self.n_hidden=n_hidden
            if indep_causal:
                NodeClass=UniformNode
            else:
                NodeClass=CausalNode
                if debug:
                    print('Using ',n_layers,'between each causal node')
                NodeClass.n_layers=n_layers
                NodeClass.n_hidden=self.n_hidden

            self.step= tf.Variable(0, name='step', trainable=False)

            NodeClass.batch_size=batch_size
            self.node_names, self.parent_names=zip(*graph)
            self.node_names=list(self.node_names)

            if not input_dict:
                #normal mode, use random uniform noise asexogenous
                self.nodes=[NodeClass(name=n) for n in self.node_names]
            else:
                self.nodes=[NodeClass(name=n,input_z=input_dict[n]) for n in self.node_names]


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

        trainable=tf.get_collection('trainable_variables')
        self.train_var=[v for v in self.var if v in trainable]

        self.labels=tf.concat(self.list_labels(),-1)

        self.fake_labels=self.labels
        self.fake_labels_logits= tf.concat( self.list_label_logits(),-1 )
        n_hidden=self.config.critic_hidden_size


        if config.is_pretrain:
            self.build_pretrain()

    def build_pretrain(self):
        #Use Native queue
        label_loader,label_stats= get_label_loader(config,config.data_path,config.batch_size)

        #Pretraining setup
        if self.config.pretrain_type=='wasserstein':
            self.DCC=DiscriminatorW
        elif self.config.pretrain_type=='gan':
            self.DCC=Discriminator_CC

        #Assuming factorized right now
        #if self.config.pt_factorized:
            #self.DCC=FactorizedNetwork(self.graph,self.DCC,self.config)

        for node in self.nodes:
            node.setup_pretrain(config,label_loader)


    def __len__(self):
        return len(self.node_dict)

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
    n_layers=3
    n_hidden=10

    def __init__(self,train=True,name=None,input_z=None,reuse=None):
        self.name=name
        self.train = train
        self.reuse=reuse

        #Use tf.random_uniform instead of placeholder for noise
        n=self.batch_size*self.n_hidden
        #print 'CN n',n
        with tf.variable_scope(self.name) as vs:
            self.z=input_z or  tf.random_uniform(
                    (self.batch_size,self.n_hidden),minval=-1.0,maxval=1.0)
            if debug:
                print 'self.',self.name,' using input_z ', input_z

            self.var = tf.contrib.framework.get_variables(vs)

    def setup_tensor(self):
        if self._label is not None:#already setup
            if debug:
                print 'self.',self.name,' has refuted setting up tensor'
            return
        tf_parents=[self.z]+[node.label_logit for node in self.parents]
        with tf.variable_scope(self.name,reuse=self.reuse):
            vec_parents=tf.concat(tf_parents,-1)
            h=vec_parents
            for l in range(self.n_layers-1):
                h=slim.fully_connected(h,self.n_hidden,activation_fn=tf.nn.tanh,scope='layer'+str(l))
            self._label_logit = slim.fully_connected(h,1,activation_fn=None,scope='proj')
            self._label=tf.nn.sigmoid( self._label_logit )
            if debug:
                print 'self.',self.name,' has setup _label=',self._label
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


    def setup_pretrain(self,config,label_loader):
        self.config=config

        n_hidden=self.config.critic_hidden_size
        for node in self.cc.nodes:
            parent_names=[p.name for p in node.parents]
            real_inputs=tf.stack([label_loader[n] for n in parent_names]+[node.label],axis=1)
            fake_inputs=tf.stack([p.label for p in node.parents]+[node.label],axis=1)

            real_prob,self.dcc_real_logit,self.dcc_var=self.DCC(real_inputs,self.config.batch_size,n_hidden,self.config)
            fake_prob,self.dcc_fake_logit,_=self.DCC(fake_inputs,self.config.batch_size,n_hidden,self.config,reuse=True)
            grad_cost,slopes=Grad_Penalty(real_inputs,fake_inputs,self.DCC,self.config)

            if self.config.pretrain_type=='gan':
                self.dcc_xe_real=tf.reduce_mean(sxe(self.dcc_real_logit,1))
                self.dcc_xe_fake=tf.reduce_mean(sxe(self.dcc_fake_logit,0))
                self.dcc_loss_real = tf.reduce_mean(self.dcc_xe_real)
                self.dcc_loss_fake = tf.reduce_mean(self.dcc_xe_fake)
                self.dcc_loss=self.dcc_loss_real+self.dcc_loss_fake
                self.c_xe_fake=sxe(self.dcc_fake_logit,1)
                self.c_loss=tf.reduce_mean(self.c_xe_fake)
            elif self.config.pretrain_type=='wasserstein':
                self.dcc_diff = self.dcc_fake_logit - self.dcc_real_logit
                self.dcc_gan_loss=tf.reduce_mean(self.dcc_diff)
                self.dcc_grad_loss=grad_cost
                self.dcc_loss=self.dcc_gan_loss+self.dcc_grad_loss#
                self.c_loss=-tf.reduce_mean(self.dcc_fake_logit)#




            list_real_labels=tf.unstack(real_labels,axis=1)
            list_fake_labels=tf.unstack(fake_labels,axis=1)
            real_label_dict={n:l for n,l in zip(node_names,list_real_labels)}
            fake_label_dict={n:l for n,l in zip(node_names,list_fake_labels)}
            real_parent_inputs=[ [real_label_dict[n] for n in p] for p in parent_names]
            fake_parent_inputs=[ [fake_label_dict[n] for n in p] for p in parent_names]
            real_inputs=[tf.stack( [real_label_dict[n]]+par,axis=1) for n,par in zip(node_names,real_parent_inputs)]
            fake_inputs=[tf.stack( [fake_label_dict[n]]+par,axis=1) for n,par in zip(node_names,fake_parent_inputs)]



        self.dcc_dict=self.DCC(self.real_labels,self.fake_labels,self.batch_size,n_hidden=n_hidden)

        self.dcc_real=tf.add_n(self.dcc_dict['real_prob'].values())/len(self.cc.node_names)
        self.dcc_real=tf.add_n(self.dcc_dict['fake_prob'].values())/len(self.cc.node_names)
        self.dcc_real_logit=tf.add_n(self.dcc_dict['real_logit'].values())/len(self.cc.node_names)
        self.dcc_fake_logit=tf.add_n(self.dcc_dict['fake_logit'].values())/len(self.cc.node_names)
        self.dcc_var=list(chain.from_iterable(self.dcc_dict['var'].values()))



class UniformNode(CausalNode):
    def setup_tensor(self):
        self._label_logit=tf.constant(-1)
        self._label=tf.random_uniform((self.batch_size,1), minval=0.0, maxval=1.0)
        self._label_logit=self._label
        self.z=self._label

