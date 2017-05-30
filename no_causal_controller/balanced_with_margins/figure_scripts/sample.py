import tensorflow as tf
import numpy as np
import os
import scipy.misc
import numpy as np
from tqdm import trange

import pandas as pd
from itertools import combinations, product
import sys

from utils import save_figure_images#makes grid image plots


def take_product(do_dict):
    '''
    this function takes some dictionary like:
        {key1:1, key2:[a,b], key3:[c,d]}
    and returns the dictionary:
        {key1:[1,1,1], key2[a,a,b,b,],key3[c,d,c,d]}
    computing the product of values
    '''
    values=[]
    for v in do_dict.values():
        if hasattr(v,'__iter__'):
            values.append(v)
        else:
            values.append([v])#allows scalar to be passed

    prod_values=np.vstack(product(*values))
    return {k:np.array(v) for k,v in zip(do_dict.keys(),zip(*prod_values))}


def chunks(input_dict, chunk_size):
    """
    Yield successive n-sized chunks.
    Takes a dictionary of iterables and makes an
    iterable of dictionaries
    """
    if len(input_dict)==0:
        return [{}]

    n=chunk_size
    batches=[]

    L=len(input_dict.values()[0])
    for i in xrange(0, L, n):
        fd={}
        n=n- max(0, (i+n) - L )#incase doesn't evenly divide
        for key,value in input_dict.items():
            fd[key]=value[i:i+n]

        batches.append(fd)
    return batches


def do2feed( do_dict, model, on_logits=True):
    '''
    this contains logit for parsing "do_dict"
    into a feed dict that can actually be worked with
    '''
    feed_dict={}
    for key,value in do_dict.items():
        if isinstance(key,tf.Tensor):
            feed_dict[key]=value
        elif isinstance(key,str):
            if key in model.cc.node_names:
                node=model.cc.node_dict[key]
                if on_logits:# intervene on logits by default
                    feed_dict[node.label_logit]=value
                else:
                    feed_dict[node.label]=value
            elif hasattr(model,key):
                feed_dict[getattr(model,key)]=value
            else:
                raise ValueError('string keys must be attributes of either\
                                 model.cc or model. Got string:',key)
        else:
            raise ValueError('keys must be tensors or strings but got',type(key))

    #Make sure [64,] isn't passed to [64,1] for example
    for tensor,value in feed_dict.items():
        #Make last dims line up:
        tf_shape=tensor.get_shape().as_list()
        shape=[len(value)]+tf_shape[1:]
        try:
            feed_dict[tensor]=np.reshape(value,shape)
        except Exception,e:
            print 'Unexpected difficulty reshaping inputs:',tensor, tf_shape, np.size(value)
            raise e

    return feed_dict



def once_sample(model, fetch, do_dict=None, step=None):
    pass


def sample(model, fetch=None, do_dict=None, on_logits=True):
    '''
    fetch should be a list of tensors to do sess.run on
    do_dict is a list of strings or tensors of the form:
    {'Male':1, model.z_gen:[0,1], model.cc.Smiling:[0.1,0.9]}
    '''

    if fetch==None:
        #assume images
        fetch=model.G

    p_do_dict=take_product(do_dict)

    ##Need divisible batch_size for most models
    if len(p_do_dict)>0:
        L=len(p_do_dict.values()[0])
    else:
        L=0
    print "L is " + str(L)
    print p_do_dict
    if L>=model.batch_size:
        if not L % model.batch_size == 0:
            raise ValueError('do_dict must be dividable by batch_size\
                             but instead product of inputs was of length',L)
        feed_dict = do2feed(p_do_dict, model, on_logits=on_logits)
    elif model.batch_size % L == 0:
        p_do_dict = {key:np.repeat(value,model.batch_size/L,axis=0) for key,value in p_do_dict.items()}
        feed_dict = do2feed(p_do_dict, model, on_logits=on_logits)
    else:
        raise ValueError('No. of intervened values must divide batch_size.')

    fds=chunks(feed_dict,model.batch_size)

    outputs=[]
    for fd in fds:
        out=model.sess.run(fetch, fd)
        outputs.append(out)

    return np.vstack(outputs), feed_dict

def intervention2d(model, fetch=None, do_dict=None, do_dict_name=None, on_logits=True, step=''):
    '''
    This function is a wrapper around the more general function "sample".
    In this function, the do_dict is assumed to have only two varying
    parameters on which a 2d interventions plot can be made.
    '''
    image_dim = np.sqrt(model.batch_size).astype(int)
    if not on_logits:
        raise ValueError('on_logits=False not implemented')

    #Interpret defaults:
    #n_defaults=len( filter(lambda l:l == 'model_default', do_dict.values() ))
    #accept any string for now
    n_defaults=len( filter(lambda l: isinstance(l,str), do_dict.values() ))

    if n_defaults>0:
        print n_defaults,' default values given..using 8 for each of them'

    try:
        for key,value in do_dict.items():
            if value == 'model_default':
                itv_min,itv_max=model.intervention_range[key]
                do_dict[key]=np.linspace(itv_min,itv_max,8)
            else:
                #otherwise pass a number, list, or array
                assert(not isinstance(value,str))

    except Exception, e:
        raise(e,'Difficulty accessing default model interventions')


    str_step=str(step)

    lengths = [ len(v) for v in do_dict.values() if hasattr(v,'__len__') ]
    #print('lengths',lengths)
    print 'lengths',lengths

    gt_one = filter(lambda l:l>1,lengths)

    if not 0<=len(gt_one)<=2:
        raise ValueError('for visualizing intervention, must have < 3 parameters varying')
    if len(gt_one) == 0:
        size = [image_dim,image_dim]
    if len(gt_one)==1 and lengths[0]>=model.batch_size:
        size=[gt_one[0],1]
    elif len(gt_one)==1 and lengths[0]<model.batch_size:
        size = [image_dim,image_dim]
    elif len(gt_one)==2:
        size=[gt_one[0],gt_one[1]]


    #Terminology
    if model.model_name=='began':
        result_dir=model.model_dir
        if str_step=='':
            str_step=str( model.sess.run(model.step) )+'_'
    elif model.model_name=='dcgan':
        print 'DCGAN'
        result_dir=model.checkpoint_dir

    sample_dir=os.path.join(result_dir,'sample_figures')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    #print 'do_dict DEBUG:',do_dict
    images, feed_dict= sample(model,fetch=model.G, do_dict=do_dict,on_logits=on_logits)

    #print 'DEBUG,shape:',images.shape
    #images, feed_dict= sample(model,fetch=model.G, do_dict={},on_logits=on_logits)
    ckpt = tf.train.get_checkpoint_state(model.checkpoint_dir)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    itv_file=os.path.join(sample_dir, str_step+str(do_dict_name)+str(ckpt_name[-5:])+'.png')
    #if os.path.exists(itv_file):
    #    itv_file='new'+itv_file #don't overwrite

    print '[*] saving intervention2d:',itv_file
    save_figure_images(model.model_name,images,itv_file,size=size)

def intervention_noCC(model, config, fetch=None, do_dict=None, do_dict_name=None, on_logits=True, step=''):
    image_dim = np.sqrt(model.batch_size).astype(int)
    size = [image_dim,image_dim]
    result_dir=config.checkpoint_dir + 'celebA_64_64_64/'
    print result_dir
    sample_dir=os.path.join(result_dir,'sample_figures')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    z_gen_fixed = np.random.uniform(-1, 1, size=(32, model.z_gen_dim)) # first 32 z_gen is the same as next 32
    z_gen_fixed = np.tile(z_gen_fixed,[2,1])
    name_list = [i[0] for i in model.graph]
    sub = model.attributes
    l = sub.shape[0]
    pointer = np.random.random_integers(0,l-33,1)[0]
    fake_labels_fixed = np.ones((model.batch_size,len(name_list)))
    name_id2 = 0
    for name2 in name_list:
        # print name2
        # print sub[name2].shape
        fake_labels_fixed[0:32,name_id2] = 0.5 + 0.2*sub[name2][pointer:pointer+32].values.reshape((32,))
        name_id2 = name_id2 + 1
    fake_labels_fixed[32:64,:] = fake_labels_fixed[0:32,:]
    name_id2 = 0
    for name2 in name_list:
        if name2 == do_dict_name:
            fake_labels_fixed[:,name_id2] = 0.5+0.2*np.vstack((np.ones((32,1)), -np.ones((32,1)) )).reshape((64,))
        name_id2 = name_id2 + 1
    str_step=str(step)
    images = model.sess.run(model.G, feed_dict={model.z_gen:z_gen_fixed, model.fake_labels:fake_labels_fixed})
    ckpt = tf.train.get_checkpoint_state(result_dir)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    itv_file=os.path.join(sample_dir, str_step+str(do_dict_name)+str(ckpt_name[-5:])+'.png')
    save_figure_images(model.model_name,images,itv_file,size=size)





