import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *

#class Data3d

def sxe(logits,labels):
    #use zeros or ones if pass in scalar
    if not isinstance(labels,tf.Tensor):
        labels=labels*tf.ones_like(logits)
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,labels=labels)

#def linear(input_, output_dim, scope=None, stddev=10.):
def linear(input_, output_dim, scope=None, stddev=.7):
    unif = tf.uniform_unit_scaling_initializer()
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        #w = tf.get_variable('w', [input_.get_shape()[1], output_dim], initializer=unif)
        w = tf.get_variable('w', [input_.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input_, w) + b


class Arrows:
    x_dim=3
    e_dim=3
    bdry_buffer=0.05# output in [bdry_buffer,1-bdry_buffer]
    def __init__(self,N):
        with tf.variable_scope('Arrow') as scope:
            self.N=tf.placeholder_with_default(N,shape=[])
            #self.N=tf.constant(N) #how many to sample at a time
            self.e1=tf.random_uniform([self.N,1],0,1)
            self.e2=tf.random_uniform([self.N,1],0,1)
            self.e3=tf.random_uniform([self.N,1],0,1)
            self.build()
            #WARN. some of these are not trainable: i.e. poly
            self.var = tf.contrib.framework.get_variables(scope)
    def build(self):
        pass

    def normalize_output(self,X):
        '''
        I think that data literally in [0,1] was difficult for sigmoid network.
        Therefore, I am normalizing it to [bdry_buffer,1-bdry_buffer]

        X: assumed to be in [0,1]
        '''
        return (1.-2*self.bdry_buffer)*X + self.bdry_buffer



class Generator:
    x_dim=3
    def __init__(self, N, hidden_size=10,z_dim=10):
        with tf.variable_scope('Gen') as scope:
            self.N=tf.placeholder_with_default(N,shape=[])
            self.hidden_size=hidden_size
            self.z_dim=z_dim
            self.build()
            self.tr_var = tf.contrib.framework.get_variables(scope)
            self.step=tf.Variable(0,name='step',trainable=False)
            self.var = tf.contrib.framework.get_variables(scope)
    def build(self):
        raise Exception('must override')
    def smallNN(self,inputs,name='smallNN'):
        with tf.variable_scope(name):
            if isinstance(inputs,list):
                inputs=tf.concat(inputs,axis=1)
            h01 = tf.tanh(linear(inputs, self.hidden_size, name+'l1'))
            h11 = tf.tanh(linear(h01, self.hidden_size, name+'l21'))
            #h21 = output_nonlinearity(linear(h11, 1, name+'l31'))
            #h21 = linear(h11, 1, name+'l31')
            h21 = tf.sigmoid(linear(h11, 1, name+'l31'))

        return h21#rank2
        #return tf.sigmoid(h21)#rank2


randunif=tf.random_uniform_initializer(0,1,dtype=tf.float32)
def poly(cause,cause2=None,cause3=None,name='poly1d',reuse=None):
    #assumes input is in [0,1]. Enforces output is in [0,1]
    #if cause2 is not given, this is a cubic poly is 1 variable

    #cause and cause2 should be given as tensors like (N,1)

    #Check conditions
    if isinstance(cause2,str):
        raise ValueError('cause2 was a string. you probably forgot to include\
                         the "name=" keyword when specifying only 1 cause')
    if isinstance(cause3,str):
        raise ValueError('cause3 was a string. you probably forgot to include\
                         the "name=" keyword when specifying only 1 cause')
    if not len(cause.shape)>=2:
        cshape=cause.get_shape().as_list()
        raise ValueError('cause and cause2 must have len(shape)>=2. shape was' , cshape )
    if cause2 is not None:
        if not len(cause2.get_shape().as_list())>=2:
            cshape2=cause2.get_shape().as_list()
            raise ValueError('cause and cause2 must have len(shape)>=2. shape was %r'%(cshape2))
    if cause3 is not None:
        if not len(cause3.get_shape().as_list())>=2:
            cshape3=cause3.get_shape().as_list()
            raise ValueError('cause and cause3 must have len(shape)>=2. shape was %r'%(cshape3))

    #Start
    with tf.variable_scope(name,reuse=reuse):
        if cause2 is not None and cause3 is not None:
            inputs=[tf.ones_like(cause),cause,cause2,cause3]
        if cause2 is not None and cause3 is None:
            inputs=[tf.ones_like(cause),cause,cause2]
        else:
            inputs=[tf.ones_like(cause),cause]
        dim=len(inputs)#2 or 3 or 4

        C=np.random.rand(1,dim,dim,dim).astype(np.float32)#unif
        C=2*C-1 #unif[-1,1]

        n=200
        N=n**(dim-1)
        grids=np.mgrid[[slice(0,1,1./n) for i in inputs[1:]]]
        y=np.hstack([np.ones((N,1))]+[g.reshape(N,1) for g in grids])
        y1=np.reshape(y,[N,-1,1,1])
        y2=np.reshape(y,[N,1,-1,1])
        y3=np.reshape(y,[N,1,1,-1])

        test_poly=np.sum(y1*y2*y3*C,axis=(1,2,3))
        Cmin=np.min(test_poly)
        Cmax=np.max(test_poly)
        #normalize [0,1]->[0,1]
        C[0,0,0,0]-=Cmin
        C/=(Cmax-Cmin)

        coeff=tf.Variable(C,name='coef',trainable=False)

        #M=cause.get_shape.as_list()[0]
        x=tf.concat(inputs,axis=1)
        x1=tf.reshape(x,[-1,dim,1,1])
        x2=tf.reshape(x,[-1,1,dim,1])
        x3=tf.reshape(x,[-1,1,1,dim])

        poly=tf.reduce_sum(x1*x2*x3*coeff,axis=[1,2,3])
        return tf.reshape(poly,[-1,1])


class CompleteArrows(Arrows): # Data generated from the causal graph X1->X2->X3
    name='complete'
    def build(self):
        with tf.variable_scope(self.name):
            self.X1=poly(self.e1,name='X1')
            #self.X2=0.5*poly(self.X1,name='X1cX2')+0.5*self.e2
            #self.X3=0.5*poly(self.X1,self.X2,name='X1X2cX3')+0.5*self.e3
            self.X2=poly(self.X1,self.e2,name='X1cX2')
            self.X3=poly(self.X1,self.X2,self.e3,name='X1X2cX3')
            self.X=tf.concat([self.X1,self.X2,self.X3],axis=1)
            self.X=self.normalize_output(self.X)
            #print 'completearrowX.shape:',self.X.get_shape().as_list()
class CompleteGenerator(Generator):
    name='complete'
    def build(self):
        with tf.variable_scope(self.name):
            self.z=tf.random_uniform((self.N,self.x_dim*self.z_dim), 0,1,name='z')
            z1,z2,z3=tf.split( self.z ,3,axis=1)#3=x_dim
            self.X1=self.smallNN(z1,'X1')
            self.X2=self.smallNN([self.X1,z2],'X1cX2')
            self.X3=self.smallNN([self.X1,self.X2,z3],'X1X2cX3')
            self.X=tf.concat([self.X1,self.X2,self.X3],axis=1)
            #print 'completegenX.shape:',self.X.get_shape().as_list()

class ColliderArrows(Arrows):
    name='collider'
    def build(self):
        with tf.variable_scope(self.name):
            self.X1=poly(self.e1,name='X1')
            self.X3=poly(self.e3,name='X3')
            #self.X2=0.5*poly(self.X1,self.X3,name='X1X3cX2')+0.5*self.e2
            self.X2=poly(self.X1,self.X3,self.e2,name='X1X3cX2')
            self.X=tf.concat([self.X1,self.X2,self.X3],axis=1)
            self.X=self.normalize_output(self.X)
class ColliderGenerator(Generator):
    name='collider'
    def build(self):
        with tf.variable_scope(self.name):
            self.z=tf.random_uniform((self.N,self.x_dim*self.z_dim), 0,1,name='z')
            z1,z2,z3=tf.split( self.z ,3,axis=1)#3=x_dim
            self.X1=self.smallNN(z1,'X1')
            self.X3=self.smallNN(z3,'X3')
            self.X2=self.smallNN([self.X1,self.X3,z2],'X1X3cX2')
            self.X=tf.concat([self.X1,self.X2,self.X3],axis=1)

class LinearArrows(Arrows):
    name='linear'
    def build(self):
        with tf.variable_scope(self.name):
            self.X1=poly(self.e1,name='X1')
            #self.X2=0.5*poly(self.X1,name='X2')+0.5*self.e2
            #self.X3=0.5*poly(self.X2,name='X3')+0.5*self.e3
            self.X2=poly(self.X1,self.e2,name='X2')
            self.X3=poly(self.X2,self.e3,name='X3')
            self.X=tf.concat([self.X1,self.X2,self.X3],axis=1)
            self.X=self.normalize_output(self.X)
class LinearGenerator(Generator):
    name='linear'
    def build(self):
        with tf.variable_scope(self.name):
            self.z=tf.random_uniform((self.N,self.x_dim*self.z_dim), 0,1,name='z')
            z1,z2,z3=tf.split( self.z ,3,axis=1)#3=x_dim
            self.X1=self.smallNN(z1,'X1')
            self.X2=self.smallNN([self.X1,z2],'X2')
            self.X3=self.smallNN([self.X2,z3],'X3')
            self.X=tf.concat([self.X1,self.X2,self.X3],axis=1)

class NetworkArrows(Arrows):
    name='network'
    def build(self):
        with tf.variable_scope(self.name):
            self.hidden_size=10
            h0 = tf.tanh(linear(self.e1, self.hidden_size, 'netarrow0'))
            h1 = tf.tanh(linear(h0, self.hidden_size, 'netarrow1'))
            h2 = tf.tanh(linear(h1, self.hidden_size, 'netarrow2'))
            h3 = tf.tanh(linear(h2, self.hidden_size, 'netarrow3'))
            h4 = tf.sigmoid(linear(h3, self.x_dim, 'netarrow4'))
            self.X=self.normalize_output(h4)

class FC3_Generator(Generator):
    name='fc3'
    def build(self):
        z=tf.random_uniform((self.N,self.x_dim*self.z_dim), 0,1,name='z')
        z1,z2,z3=tf.split( z ,3,axis=1)#3=x_dim
        h0 = tf.tanh(linear(z1, self.hidden_size, 'fc3gen0'))
        h1 = tf.tanh(linear(h0, self.hidden_size, 'fc3gen1'))
        h2 = tf.sigmoid(linear(h1, self.x_dim, 'fc3gen2'))
        self.X=h2

class FC5_Generator(Generator):
    name='fc5'
    def build(self):
        z=tf.random_uniform((self.N,self.x_dim*self.z_dim), 0,1,name='z')
        z1,z2,z3=tf.split( z ,3,axis=1)#3=x_dim
        h0 = tf.tanh(linear(z1, self.hidden_size, 'fc5gen0'))
        h1 = tf.tanh(linear(h0, self.hidden_size, 'fc5gen1'))
        h2 = tf.tanh(linear(h1, self.hidden_size, 'fc5gen2'))
        h3 = tf.tanh(linear(h2, self.hidden_size, 'fc5gen3'))
        h4 = tf.sigmoid(linear(h3, self.x_dim, 'fc5gen4'))
        self.X=h4

class FC10_Generator(Generator):
    name='fc10'
    def build(self):
        z=tf.random_uniform((self.N,self.x_dim*self.z_dim), 0,1,name='z')
        z1,z2,z3=tf.split( z ,3,axis=1)#3=x_dim
        h0 = tf.tanh(linear(z1, self.hidden_size, 'fc10gen0'))
        h1 = tf.tanh(linear(h0, self.hidden_size, 'fc10gen1'))
        h2 = tf.tanh(linear(h1, self.hidden_size, 'fc10gen2'))
        h3 = tf.tanh(linear(h2, self.hidden_size, 'fc10gen3'))
        h4 = tf.tanh(linear(h3, self.hidden_size, 'fc10gen4'))
        h5 = tf.tanh(linear(h4, self.hidden_size, 'fc10gen5'))
        h6 = tf.tanh(linear(h5, self.hidden_size, 'fc10gen6'))
        h7 = tf.tanh(linear(h6, self.hidden_size, 'fc10gen7'))
        h8 = tf.tanh(linear(h7, self.hidden_size, 'fc10gen8'))
        h9 = tf.sigmoid(linear(h8, self.x_dim, 'fc10gen9'))
        self.X=h9


def minibatch(input_, num_kernels=5, kernel_dim=3):
    x = linear(input_, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input_, minibatch_features],1)


def Discriminator(input_, hidden_size,minibatch_layer=True,alpha=0.5,reuse=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        h0_ = tf.nn.relu(linear(input_, hidden_size, 'disc0'))
        h0 = tf.maximum(alpha*h0_,h0_)
        h1_ = tf.nn.relu(linear(h0, hidden_size, 'disc1'))
        h1 = tf.maximum(alpha*h1_,h1_)
        if minibatch_layer:
            h2 = minibatch(h1)
        else:
            h2_ = tf.nn.relu(linear(h1, hidden_size, 'disc2'))
            h2 = tf.maximum(alpha*h2_,h2_)
        h3 = linear(h2, 1, 'disc3')
        return h3



GeneratorTypes={CompleteGenerator.name:CompleteGenerator,
            ColliderGenerator.name:ColliderGenerator,
            LinearGenerator.name:LinearGenerator,
            FC3_Generator.name:FC3_Generator,
            FC5_Generator.name:FC5_Generator,
            FC10_Generator.name:FC10_Generator}
DataTypes={CompleteArrows.name:CompleteArrows,
           ColliderArrows.name:ColliderArrows,
           LinearArrows.name:LinearArrows,
           NetworkArrows.name:NetworkArrows}

#def poly1d(cause,name='poly1d',reuse=None):
#    #assumes input is in [0,1]. Enforces output is in [0,1]
#    print 'Warning poly1d not ready yet'
#    with tf.variable_scope(name,initializer=randunif,reuse=reuse):
#        #C=np.random.rand(1,2,2).astype(np.float32)#unif
#        C=np.random.rand(1,2,2,2).astype(np.float32)#unif
#
#        #find min and max
#        N=2000
#        y=np.hstack([np.ones((N,1)),np.linspace(0,1.,N).reshape((N,1))])
#        y1=np.reshape(y,[N,2,1,1])
#        y2=np.reshape(y,[N,1,2,1])
#        y3=np.reshape(y,[N,1,1,2])
#
#        test_poly=np.sum(y1*y2*y3*C,axis=(1,2,3))
#        Cmin=np.min(test_poly)
#        Cmax=np.max(test_poly)
#
#        #normalize [0,1]->[0,1]
#        C[0,0,0,0]-=Cmin
#        C/=(Cmax-Cmin)
#
#        coeff=tf.Variable(C,name='coef',trainable=False)
#        x2=tf.reshape(tf.stack([tf.ones_like(cause),cause],axis=1),[-1,1,2])
#        x1=tf.transpose(x2,[0,2,1])
#        poly=tf.reduce_sum(x1*x2*coeff,axis=[1,2])
#        out= tf.squeeze(poly)
#        return poly
#
#        #coeff=tf.Variable(trainable=False,expected_shape=[1,3])
#    #    X=tf.stack([cause,cause*cause,cause*cause*cause],axis=1)
#    #    return tf.reduce_sum(coeff*X,axis=1)/tf.reduce_max(coeff)
#
#def poly2d(cause,cause2,name='poly2d',reuse=None):
#    with tf.variable_scope(name,initializer=randunif,reuse=reuse):
#        #coeff=tf.Variable(np.random.randn(1,2,2,2).astype(np.float32),trainable=False)
#        #x3=tf.reshape(tf.stack([cause,cause2],axis=0),[-1,1,1,2])
#        #x2=tf.transpose(x3,[0,2,3,1])
#        #x1=tf.transpose(x2,[0,2,3,1])
#
#        C=np.random.rand(1,3,3,3).astype(np.float32)
#        C[:,0,0,0]=0.#constant
#        C[:,0,2,0]=1.#x^3,y^3 coeff
#        C[:,0,0,2]=1.
#        coeff=tf.Variable(C, trainable=False)
#        x3=tf.reshape(tf.stack([tf.ones_like(cause),cause,cause2],axis=1),[-1,1,1,3])
#        x2=tf.transpose(x3,[0,2,3,1])
#        x1=tf.transpose(x2,[0,2,3,1])
#
#        poly=tf.reduce_sum(x1*x2*x3*coeff,axis=[1,2,3])
#
#        #out = tf.squeeze(poly)/tf.reduce_max(coeff)
#        out= tf.squeeze(poly)
#        return out

