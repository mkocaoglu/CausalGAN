from __future__ import division
from figure_scripts.pairwise import crosstab
from tqdm import trange
from figure_scripts.sample import intervention2d,get_joint
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pandas as pd
import sys
import scipy.stats as stats

from ops import *
from utils import *
from tensorflow.core.framework import summary_pb2
from tensorflow.contrib import slim

from Causal_controller import CausalController
from figure_scripts import pairwise

import time

def lrelu(x,leak=0.2,name='lrelu'):
  with tf.variable_scope(name):
      f1=0.5 * (1+leak)#halves memory footprint
      f2=0.5 * (1-leak)#relative to tf.maximum(leak*x,x)
      return f1*x + f2*tf.abs(x)

##Use this graph if no graph argument is passed in
#from causal_graph import standard_graph #This is the Male->Smiling<-Young
#from causal_graph import male_causes_beard
#from causal_graph import get_causal_graph


Graphs={
    'indep':[
        ['Young',[]],
        ['Male',[]],
        ['Smiling',[]],
        ['Narrow_Eyes',[]],
        ],

    'ScY':[
        ['Young',['Smiling']],
        ['Male',[]],
        ['Smiling',[]],
        ['Narrow_Eyes',[]],
        ],

    'YcS':[
        ['Young',[]],
        ['Male',[]],
        ['Smiling',['Young']],
        ['Narrow_Eyes',[]],
        ],

    'YcScNE':[
        ['Young',[]],
        ['Male',[]],
        ['Smiling',['Young']],
        ['Narrow_Eyes',['Smiling']],
        ],

    'NEcScY':[
        ['Young',['Smiling']],
        ['Male',[]],
        ['Smiling',['Narrow_Eyes']],
        ['Narrow_Eyes',[]],
        ],

#    'MxYcScNE':[
#        ['Young',[]],
#        ['Male',[]],
#        ['Smiling',['Male','Young']],
#        ['Narrow_Eyes',['Smiling']],
#        ],

    'MxYcS_SxMxYcNE':[
        ['Young',[]],
        ['Male',[]],
        ['Smiling',['Male','Young']],
        ['Narrow_Eyes',['Male','Young','Smiling']],
        ],

    'NEcSxMxY_ScYxM':[
        ['Young',['Narrow_Eyes','Smiling']],
        ['Male',['Narrow_Eyes','Smiling']],
        ['Smiling',['Narrow_Eyes']],
        ['Narrow_Eyes',[]],
        ],

}





hidden_size=10
batch_size=64

def discriminator_CC( labels, reuse=False):
    with tf.variable_scope("disc_CC") as scope:
      if reuse:
        scope.reuse_variables()
      # add minibatch features here to get fake labels with high variation
      def add_minibatch_features_for_labels(labels,batch_size):
        n_kernels = 50
        dim_per_kernel = 20
        shape = labels.get_shape().as_list()
        dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
        input_ = tf.reshape(labels, [-1, dim])           # -1 means "all"  
        x = linear(input_, n_kernels * dim_per_kernel,'dCC_mbLabelLinear')
        activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))
        big = np.zeros((batch_size, batch_size), dtype='float32')
        big += np.eye(batch_size)
        big = tf.expand_dims(big, 1)

        # the next step is very complicated. My best understanding is that 
        # the expanded dimension is automatically replicated to 64 to make subtraction possible
        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask
        f1 = tf.reduce_sum(masked, 2) / tf.reduce_sum(mask)
        # sums over the third dimension, which is of size 64 (captures cross distance to other images in batch)
        #f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

        minibatch_features = tf.concat([labels, f1],1)
        return minibatch_features

      h0 = slim.fully_connected(labels,hidden_size,activation_fn=lrelu,scope='dCC_0')
      h1 = slim.fully_connected(h0,hidden_size,activation_fn=lrelu,scope='dCC_1')
      h1_aug = lrelu(add_minibatch_features_for_labels(h1,batch_size),name = 'disc_CC_lrelu')
      h2 = slim.fully_connected(h1_aug,hidden_size,activation_fn=lrelu,scope='dCC_2')
      h3 = slim.fully_connected(h2,1,activation_fn=None,scope='dCC_3')
      return tf.nn.sigmoid(h3),h3


def sxe(logits,labels):
    #use zeros or ones if pass in scalar
    if not isinstance(labels,tf.Tensor):
        labels=labels*tf.ones_like(logits)
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,labels=labels)


if __name__=='__main__':
    tf.reset_default_graph()
    print 'Resetting tf graph!'



    ccs={}
    c_vars,d_vars=[],[]

    #label_names,_ = zip(*[Graphs.values()[0]])
    label_names,_ = zip(*Graphs.values()[0])
    label_names=list(label_names)
    print 'label_names:',label_names
    n_labels=len(label_names)
    realLabels = tf.placeholder(tf.float32,[None, n_labels],name='real_labels')

    attr=0.5*(1+pd.read_csv('./data/list_attr_celeba.txt',delim_whitespace=True))
    attr=attr[label_names]

    ####Calculate Total Variation####
    df2=attr.drop_duplicates()
    df2 = df2.reset_index(drop = True).reset_index()
    df2=df2.rename(columns = {'index':'ID'})
    real_data_id=pd.merge(attr,df2)
    real_counts = pd.value_counts(real_data_id['ID'])
    real_pdf=real_counts/len(attr)
    def calc_tvd(data):
        data=np.round(data)
        df_dat=pd.DataFrame(columns=label_names,data=data)
        dat_id=pd.merge(df_dat,df2,on=label_names,how='left')
        dat_counts=pd.value_counts(dat_id['ID'])
        dat_pdf = dat_counts / dat_counts.sum()

        #diff=real_pdf-dat_pdf
        diff=real_pdf.subtract(dat_pdf, fill_value=0)
        tvd=0.5*diff.abs().sum()

        return tvd




    c_vars=[]
    total_c_loss,total_d_loss=0.,0.
    for graph_name,graph in Graphs.items():
        print 'graph_name:',graph_name
        with tf.variable_scope(graph_name):
            cc=CausalController(graph,batch_size)

            fake_labels= tf.concat( cc.list_labels(),-1 )

            DCC_real, DCC_real_logits = discriminator_CC(realLabels)
            DCC_fake, DCC_fake_logits = discriminator_CC(fake_labels, reuse=True)

            dcc_loss_real = tf.reduce_mean(sxe(DCC_real_logits,1))
            dcc_loss_fake = tf.reduce_mean(sxe(DCC_fake_logits,0))
            c_loss        = tf.reduce_mean(sxe(DCC_fake_logits,1))

            d_loss=dcc_loss_real + dcc_loss_fake

            ccs[graph_name]={}
            ccs[graph_name]['fake_labels']=fake_labels
            ccs[graph_name]['cc']=cc
            ccs[graph_name]['c_loss']=c_loss
            ccs[graph_name]['d_loss']=d_loss
            ccs[graph_name]['D_on_fake']=tf.reduce_mean(DCC_fake)

            c_vars.extend(cc.var)
            total_c_loss+=c_loss
            total_d_loss+=d_loss


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'disc_CC' in var.name ]


    c_optim = tf.train.AdamOptimizer(0.00008).minimize(total_c_loss, var_list=c_vars)
    d_optim = tf.train.AdamOptimizer(0.00008).minimize(total_d_loss, var_list=d_vars)


    for key in ['c_loss','d_loss','D_on_fake']:
        with tf.name_scope(key):
            for graph_name in ccs.keys():
                tf.summary.scalar(graph_name, ccs[graph_name][key])


    sess=tf.Session()
    sess.run(tf.global_variables_initializer())


    summary_op=tf.summary.merge_all()
    model_dir="./checkpoint/pretrainlogs"
    writer = SummaryWriter(model_dir, sess.graph)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1)


    #if load( checkpoint_dir ):
    #  print(" [*] Load SUCCESS")
    #else:
    #  print(" [!] Load failed...")



    def noise(u):
      u = 0.5+np.array(u)*0.2#ranges from 0.3 to 0.7
      lower, upper, scale = 0, 0.2, 1/2.0
      t = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
      s = t.rvs(1)
      u = u + ((0.5-u)/0.2)*s
      return u

    def label_mapper(u,name):
        return noise(u)

    def make_summary(name, val):
        return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

    counter = 0
    start_time = time.time()


    #Make sure real labels are in the correct order
    name_list = ccs.values()[0]['cc'].node_names
    print name_list
    attributes=pd.read_csv('./data/list_attr_celeba.txt',delim_whitespace=True)
    for epoch in xrange(400):
        data = glob(os.path.join(
          "./data", 'celebA', "*.jpg"))
          #"./data", config.dataset, input_fname_pattern))
        #batch_idxs = min(len(data), config.train_size) // config.batch_size
        batch_idxs = min(len(data), np.inf) // batch_size
        #batch_idxs = min(len(data), config.train_size) // config.batch_size


        for idx in xrange(0, batch_idxs):
            batch_files = data[idx*batch_size:(idx+1)*batch_size]
            fileNames = [i[-10:] for i in batch_files]

            np_realLabels = np.array([np.hstack(\
                tuple([label_mapper(attributes.loc[i].loc[label_name],
                                    label_name) for label_name in name_list]))\
                                    for i in fileNames])

            fd={realLabels:np_realLabels}
            fetch={'c_opt':c_optim,
                   'd_opt':d_optim}

            T_summary=30
            T_tvd=100

            if counter%T_summary==0:
                fetch.update({'summary':summary_op,
                              'c_loss' :total_c_loss,
                              'd_loss' :total_d_loss,
                             })

            result=sess.run(fetch, feed_dict=fd)

            if counter%T_summary==0:
                summary_str=result['summary']
                writer.add_summary(summary_str, counter)
                writer.flush()

                c_loss = result['c_loss']
                d_loss = result['d_loss']
                print("[{}/{}] Loss_C: {:.6f} Loss_DCC: {:.6f}"\
                      .format(counter, 50000, c_loss, d_loss))


            if counter%T_tvd==0:
                t0=time.time()
                ckptsave=os.path.join(model_dir,'implicitgan.ckpt')
                saver.save(sess,ckptsave, global_step=counter)

                for graph_name in ccs.keys():
                    print 'tvd on graph:',graph_name,
                    tf_fake_labels=ccs[graph_name]['fake_labels']

                    #Get enough data
                    N=20
                    fake_out=[]
                    for i in range(N):
                        fake_out.append(sess.run(tf_fake_labels))
                    dat=np.vstack(fake_out)

                    tvd=calc_tvd(dat)
                    print 'tvd=:',tvd
                    sum_tvd=make_summary('tvd/'+graph_name, tvd)
                    writer.add_summary(sum_tvd,counter)

                print 'tvd took:',time.time()-t0,'(s)'


            counter+=1

