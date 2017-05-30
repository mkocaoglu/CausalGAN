import tensorflow as tf
import numpy as np
import os
import scipy.misc
import numpy as np
from tqdm import trange,tqdm

import pandas as pd
from itertools import combinations, product
import sys

from utils import save_figure_images#makes grid image plots


from IPython.core import debugger
debug = debugger.Pdb().set_trace



def get_joint(model, do_dict=None, N=50,return_discrete=True,step=''):
    '''
    do_dict is intended to be a distribution of interventions. Pass a
    distribution to get the average (discrete) g/fake label joint over that
    distribution

    Ex: if intervention=+1 corresponds to logits uniform in [0,0.6], pass
    np.linspace(0,0.6,n)

    N is number of batches to sample at each location in logitspace (num_labels
    dimensional)
    '''
    str_step=str(step)

    if do_dict is not None:

        print 'do_dict:',do_dict
        p_do_dict=interpret_dict( do_dict, model, n_times=N,on_logits=True)
        print 'interpreted p_do_dict:',p_do_dict

        lengths = [ len(v) for v in p_do_dict.values() if hasattr(v,'__len__') ]
        print 'lengths',lengths

        #n_batches=len(p_do_dict[token]/N)
        print 'len product_do_dict',len(p_do_dict[token])

        feed_dict = do2feed(p_do_dict, model)#{tensor:array}
        fds=chunks(feed_dict,model.batch_size)
    else:
        def gen_fd():
            while True:
                yield None
        fds=gen_fd()


    #print('WARNING! using only N=100 samples:DEBUG mode')
    #N=100#to go quicker

    print 'Calculating joint distribution'
    n_batches=N

    labels, d_fake_labels= [],[]
    #Terminology
    if model.model_name=='began':
        fake_labels=model.fake_labels
        D_fake_labels=model.D_fake_labels
    elif model.model_name=='dcgan':
        fake_labels=model.fake_labels
        D_fake_labels=model.D_labels_for_fake

#        outputs=[]
#        for fd in fds:
#            out=model.sess.run(fetch_dict, fd)
#            outputs.append(out['G'])
#        return np.vstack(outputs), feed_dict

    #for n in range(n_batches):

    for step in trange(n_batches):
        fd=next(fds)
        if step==0:
            if fd is not None:
                print 'FD',fd

        lab,dfl=model.sess.run([fake_labels,D_fake_labels],feed_dict=fd)

        labels.append(lab)
        d_fake_labels.append(dfl)

    n_labels=len(model.cc.node_names)
    list_labels=np.split( np.vstack(labels),n_labels, axis=1)
    list_d_fake_labels=np.split( np.vstack(d_fake_labels),n_labels, axis=1)

    joint={}
    for name, lab, dfl in zip(model.cc.node_names,list_labels,list_d_fake_labels):
        #Create dictionary:
            #node_name -> 
                        #'g_fake_label'->val
                        #'d_fake_label'->val

        if return_discrete:
            lab_val=(lab>0.5).astype('int')
            dfl_val=(dfl>0.5).astype('int')

        joint[name]={
                'g_fake_label':lab_val,
                'd_fake_label':dfl_val
                }

    return joint



#__________

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

def cond2fetch( cond_dict=None, model=None, on_logits=True):
    '''
    this contains logit for parsing "cond_dict"
    into a fetch dict that can actually be worked with.
    A fetch dict can be passed into the first argument
    of session.run and therefore has values that are all tensors
    '''
    cond_dict=cond_dict or {}

    fetch_dict={}
    for key,value in cond_dict.items():
        if isinstance(value,tf.Tensor):
            fetch_dict[key]=value#Nothing to be done
        elif isinstance(key,tf.Tensor):
            fetch_dict[key]=key#strange scenario, but possible
        elif isinstance(key,str):
            if key in model.cc.node_names:
                node=model.cc.node_dict[key]
                if on_logits:# intervene on logits by default
                    fetch_dict[key]=node.label_logit
                else:
                    fetch_dict[key]=node.label
            elif hasattr(model,key):
                fetch_dict[key]=getattr(model,key)
            else:
                raise ValueError('string keys must be attributes of either\
                                 model.cc or model. Got string:',key)
        else:
            raise ValueError('keys must be tensors or strings but got',type(key))

    return fetch_dict


def once_sample(model, fetch, do_dict=None, step=None):
    pass


def interpret_dict( a_dict, model,n_times=1, on_logits=True):
    '''
    pass either a do_dict or a cond_dict.
    The rules for converting arguments to numpy arrays to pass
    to tensorflow are identical
    '''
    if a_dict is None:
        return {}
    elif len(a_dict)==0:
        return {}

    if n_times>1:
        token=tf.placeholder_with_default(2.22)
        a_dict[token]=-2.22

    p_a_dict=take_product(a_dict)

    ##Need divisible batch_size for most models
    if len(p_a_dict)>0:
        L=len(p_a_dict.values()[0])
    else:
        L=0
    print "L is " + str(L)
    print p_a_dict

    ##Check compatability batch_size and L
    if L>=model.batch_size:
        if not L % model.batch_size == 0:
            raise ValueError('a_dict must be dividable by batch_size\
                             but instead product of inputs was of length',L)
    elif model.batch_size % L == 0:
        p_a_dict = {key:np.repeat(value,model.batch_size/L,axis=0) for key,value in p_a_dict.items()}
    else:
        raise ValueError('No. of intervened values must divide batch_size.')
    return p_a_dict



def slice_dict(feed_dict, rows):
    '''
    conditional sampling requires doing only certain indicies depending
    on the result of the previous iteration.
    This function takes a feed_dict and "slices" it,
    returning a dictionary with the same keys, but with values[rows,:]
    '''
    fd_out={}
    for key,value in feed_dict.iteritems():
        fd_out[key]=value[rows]
    return fd_out


#def get_remaining(rows, batch_size):
#    '''
#    this function takes a list/array of rows and returns
#    some subset of them of size batch_size
#
#    '''
def did_succeed( output_dict, cond_dict ):
    '''
    Used in rejection sampling:
    for each row, determine if cond is satisfied
    for every cond in cond_dict

    success is hardcoded as being more extreme
    than the condition specified
    '''
    test_key=cond_dict.keys()[0]
    print('output_dict:',np.squeeze(output_dict[test_key]))
    print('cond_dict:',cond_dict[test_key])


    print('WARNING:using hardcoded success condition')
    #definition success:
    def is_win(key):
        cond=np.squeeze(cond_dict[key])
        val=np.squeeze(output_dict[key])
        cond1=np.sign(val)==np.sign(cond)
        cond2=np.abs(val)>np.abs(cond)

        cond1=val<0.8
        cond2=val>0.6

        #cond1=val<0.6
        #cond2=val>0.0

        return cond1*cond2


    scoreboard=[is_win(key) for key in cond_dict]
    print('scoreboard', scoreboard)
    all_victories_bool=np.logical_and.reduce(scoreboard)
    return all_victories_bool.flatten()


def sample(model, cond_dict=None, do_dict=None, fetch=None, on_logits=True):
    '''
    #don't use fetch
    fetch should be a list of tensors to do sess.run on
    do_dict is a list of strings or tensors of the form:
    {'Male':1, model.z_gen:[0,1], model.cc.Smiling:[0.1,0.9]}
    '''
    if fetch:
        raise ValueError('manual fetch not supported')

    if cond_dict and do_dict:
        raise ValueError('simultaneous condition and\
                         intervention not supported')

    print('sampler recieved dictionary:',cond_dict or do_dict)


    #check_tensors_dict = interpret_dict( cond_dict, model, on_logits=on_logits)

    print('given cond_dict', cond_dict )#None


    #expand dict with products of sets of interventions/conditions
    do_dict = interpret_dict( do_dict, model, on_logits=on_logits)
    cond_dict = interpret_dict( cond_dict, model,on_logits=on_logits)#{string:array}

    print('actual cond_dict', cond_dict )#{}

    #replace strings with tensors appropriately
    feed_dict = do2feed(do_dict, model, on_logits=on_logits)#{tensor:array}
    fetch_dict= cond2fetch(cond_dict,model,on_logits=on_logits) #{string:tensor}
    fetch_dict.update(fetch or {'G':model.G})

    print('feed_dict',feed_dict)
    print('fetch_dict',fetch_dict)

    if not cond_dict and do_dict:
        #Simply do intervention w/o loop
        print('sampler mode:Interventional')

        fds=chunks(feed_dict,model.batch_size)

        outputs=[]
        for fd in fds:
            out=model.sess.run(fetch_dict, fd)
            outputs.append(out['G'])
        return np.vstack(outputs), feed_dict

    elif cond_dict and not do_dict:
    #Could also pass do_dict here to be interesting
        ##Implements rejection sampling
        print('sampler mode:Conditional')


        rows=np.arange( len(cond_dict.values()[0]))#what idx do we need
        assert(len(rows)>=model.batch_size)#should already be true.

        print('nrows:',len(rows))

        #init
        print('WARNING: max_fail too high')
        #max_fail=100
        max_fail=10000
        n_fails=np.zeros_like(rows)
        remaining_rows=rows.copy()
        completed_rows=[]

        #null=lambda :[-1 for r in rows]
        outputs={key:[-1 for r in rows]for key in fetch_dict}
        print('n keys in outputs:',len(outputs.keys()))

        #debug()

        ii=0
        while( len(remaining_rows)>0 ):
            #debug()
            ii+=1
            print('Iter:',ii)
            #loop
            iter_rows=remaining_rows[:model.batch_size]
            print('iter_rows:',len(iter_rows),':',iter_rows)
            n_pad = model.batch_size - len(iter_rows)
            print('n_pad:',n_pad)
            #iter_rows.extend( [iter_rows[-1]]*n_pad )#just duplicate
            pad_iter_rows=list(iter_rows)+ ( [iter_rows[-1]]*n_pad )

            iter_rows=np.array(iter_rows)
            pad_iter_rows=np.array(pad_iter_rows)

            fed=slice_dict( feed_dict, pad_iter_rows )
            cond=slice_dict( cond_dict, pad_iter_rows )

            out=model.sess.run(fetch_dict, fed)

            bool_pass = did_succeed(out,cond)[:len(iter_rows)]
            print('bool_pass:',len(bool_pass),':',bool_pass)
            pass_idx=iter_rows[bool_pass]
            fail_idx=iter_rows[~bool_pass]


            good_rows=set( iter_rows[bool_pass] )
            print('good_rows',good_rows)
            bad_rows=set( rows[ n_fails>=max_fail ] )
            print('bad_rows',bad_rows)

            #yuck
            for key in out:
                for i,row_pass in enumerate(bool_pass):
                    idx=iter_rows[i]
                    if row_pass:
                        outputs[key][idx]=out[key][i]
                    else:
                        n_fails[idx]+=1
                for idx_giveup in bad_rows:
                    shape=fetch_dict[key].get_shape().as_list()[1:]
                    outputs[key][idx_giveup]=np.zeros(shape)
                    print('key:',key,' shape giveup:',shape)


            ##Remove rows
            remaining_rows=list( set(remaining_rows)-good_rows-bad_rows )

            #debug()

            print('n_fails:10',n_fails[:10])

        #Tempory since for now we are only interested in 'G'
        return np.stack(outputs['G']),cond_dict

    else:
        raise Exception('This should not happen')




def condition2d( model, cond_dict,cond_dict_name,step='', on_logits=True):
    '''
    Function largely copied from intervention2d with minor changes.

    This function is a wrapper around the more general function "sample".
    In this function, the cond_dict is assumed to have only two varying
    parameters on which a 2d interventions plot can be made.
    '''
    #TODO: Unify function with intervention2d

    if not on_logits:
        raise ValueError('on_logits=False not implemented')

    #Interpret defaults:
    #n_defaults=len( filter(lambda l:l == 'model_default', cond_dict.values() ))
    #accept any string for now
    n_defaults=len( filter(lambda l: isinstance(l,str), cond_dict.values() ))

    if n_defaults>0:
        print n_defaults,' default values given..using 8 for each of them'

    try:
        for key,value in cond_dict.items():
            if value == 'model_default':
                print('Warning! using 1/2*model.intervention_range\
                      to specify the conditioning defaults')
                cond_min,cond_max=model.intervention_range[key]
                #cond_dict[key]=np.linspace(cond_min,cond_max,8)
                cond_dict[key]=[0.5*cond_min,0.5*cond_max]
                print('Condition dict used:',cond_dict)

            elif value=='percentile':
                #fetch=cond2fetch(cond_dict)
                print('...calculating percentile')
                data=[]
                for _ in range(30):
                    data.append(model.sess.run(model.cc.node_dict[key].label_logit))
                D=np.vstack(data)
                print('dat',D.flatten())
                cond_dict[key]=np.repeat([np.percentile(D,95),np.percentile(D,5)],64)
                print('percentiles for',key,'are',[np.percentile(D,5),np.percentile(D,95)])


            else:
                #otherwise pass a number, list, or array
                assert(not isinstance(value,str))

    except Exception, e:
        raise(e,'Difficulty accessing default model interventions')


    str_step=str(step)

    lengths = [ len(v) for v in cond_dict.values() if hasattr(v,'__len__') ]
    #print('lengths',lengths)
    print 'lengths',lengths

    gt_one = filter(lambda l:l>1,lengths)

    if not 0<=len(gt_one)<=2:
        raise ValueError('for visualizing intervention, must have < 3 parameters varying')
    if len(gt_one) == 0:
        image_dim = np.sqrt(model.batch_size).astype(int)
        size = [image_dim,image_dim]
#    if len(gt_one)==1 and lengths[0]>=model.batch_size:
#        size=[gt_one[0],1]
#    elif len(gt_one)==1 and lengths[0]<model.batch_size:
#        image_dim = np.sqrt(model.batch_size).astype(int)
#        size = [image_dim,image_dim]
#    elif len(gt_one)==2:
#        size=[gt_one[0],gt_one[1]]
#

    elif len(gt_one)==2:
        size=[gt_one[0],gt_one[1]]

    else:
        N=np.prod(lengths)
        if N%8==0:
            size=[N/8,8]
        else:
            size=[8,8]



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

    images, _= sample(model, cond_dict=cond_dict,on_logits=on_logits)

    print 'Images shape:',images.shape


    #cond_file=os.path.join(sample_dir, str_step+str(cond_dict_name)+'_cond'+'.png')
    cond_file=os.path.join(sample_dir,str_step+str(cond_dict_name)+'_cond'+'.pdf')

    #if os.path.exists(cond_file):
    #    cond_file='new'+cond_file #don't overwrite

    print '[*] saving condition2d:',cond_file
    save_figure_images(model.model_name,images,cond_file,size=size)



def intervention2d(model, fetch=None, do_dict=None, do_dict_name=None, on_logits=True, step=''):
    '''
    This function is a wrapper around the more general function "sample".
    In this function, the do_dict is assumed to have only two varying
    parameters on which a 2d interventions plot can be made.
    '''
    #TODO: Unify function with condition2d

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

            elif value=='percentile':
                print('...calculating percentile')
                data=[]
                for _ in range(30):
                    data.append(model.sess.run(model.cc.node_dict[key].label_logit))
                D=np.vstack(data)
                print('dat',D.flatten())
                do_dict[key]=np.repeat([np.percentile(D,95),np.percentile(D,5)],64)
                print('percentiles for',key,'are',[np.percentile(D,5),np.percentile(D,95)])
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
        #image_dim = np.sqrt(model.batch_size).astype(int)
        image_dim = np.sqrt(64).astype(int)
        size = [image_dim,image_dim]

    #if len(gt_one)==1 and lengths[0]>=model.batch_size:
    #    size=[gt_one[0],1]
    #elif len(gt_one)==1 and lengths[0]<model.batch_size:
    #    #image_dim = np.sqrt(model.batch_size).astype(int)
    #    image_dim = np.sqrt(64).astype(int)
    #    size = [image_dim,image_dim]
    elif len(gt_one)==2:
        size=[gt_one[0],gt_one[1]]

    else:
        N=np.prod(lengths)
        if N%8==0:
            size=[N/8,8]
        else:
            size=[8,8]


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
    images, feed_dict= sample(model, do_dict=do_dict,on_logits=on_logits)


    itv_file=os.path.join(sample_dir, str_step+str(do_dict_name)+'_intv'+'.pdf')
    #itv_file=os.path.join(sample_dir, str_step+str(do_dict_name)+'_intv'+'.png')

    #if os.path.exists(itv_file):
    #    itv_file='new'+itv_file #don't overwrite

    print '[*] saving intervention2d:',itv_file
    save_figure_images(model.model_name,images,itv_file,size=size)




####
#def intervention2d(*args,**kwargs):
#    print('Warning: use image_panel_2d instead of intervention2d in future')
#    return image_panel_2d(*args,**kwargs)



