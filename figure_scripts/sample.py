from __future__ import print_function
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

#convenience functions
from utils import make_sample_dir,guess_model_step,infer_grid_image_shape


from IPython.core import debugger
debug = debugger.Pdb().set_trace


def find_logit_percentile(model, key, per):
    data=[]
    for _ in range(30):
        data.append(model.sess.run(model.cc.node_dict[key].label_logit))
    D=np.vstack(data)
    pos_logits,neg_logits=D[D>0], D[D<0]
    pos_tile = np.percentile(pos_logits,per)
    neg_tile = np.percentile(neg_logits,100-per)
    return pos_tile,neg_tile

def fixed_label_diversity(model, config,step=''):
    sample_dir=make_sample_dir(model)
    str_step=str(step) or guess_model_step(model)

    N=64#per image
    n_combo=5#n label combinations

    #0,1 label combinations
    fixed_labels=model.attr.sample(n_combo)[model.cc.node_names]
    size=infer_grid_image_shape(N)

    for j, fx_label in enumerate(fixed_labels.values):
        fx_label=np.reshape(fx_label,[1,-1])
        fx_label=np.tile(fx_label,[N,1])
        do_dict={model.cc.labels: fx_label}

        images, feed_dict= sample(model, do_dict=do_dict)
        fx_file=os.path.join(sample_dir, str_step+'fxlab'+str(j)+'.pdf')
        save_figure_images(model.model_type,images['G'],fx_file,size=size)

    #which image is what label
    fixed_labels=fixed_labels.reset_index(drop=True)
    fixed_labels.to_csv(os.path.join(sample_dir,str_step+'fxlab'+'.csv'))


def get_joint(model, int_do_dict=None,int_cond_dict=None, N=6400,return_discrete=True):
    '''
    Returns a dictionary of dataframes of samples.
    Each dataframe correponds to a different tensor i.e. cc labels, d_labeler
    labels etc.

    int_do_dict and int_cond_dict indicate that just a simple +1 or 0 should be
    passed in
    ex: int_do_dict={'Wearing_Lipstick':+1}


    Ex: if intervention=+1 corresponds to logits uniform in [0,0.6], pass
    np.linspace(0,0.6,n)

    N is number of batches to sample at each location in logitspace (num_labels
    dimensional)
    '''

    #values are either +1 or -1 in cond and do dict

    do_dict,cond_dict={},{}
    if int_do_dict is not None:
        for key,value in int_do_dict.items():
            #Intervene in the middle of where the model is used to operating
            print('calculating percentile...')
            data=[]
            for _ in range(30):
                data.append(model.sess.run(model.cc.node_dict[key].label_logit))
            D=np.vstack(data)
            pos_logits,neg_logits=D[D>0], D[D<0]
            if value == 1:
                intv = np.percentile(pos_logits,50)
            elif value == 0:
                intv = np.percentile(neg_logits,50)
            else:
                raise ValueError('pass either +1 or 0')
            do_dict[key]=np.repeat([intv],N)


    if int_cond_dict is not None:
        for key,value in int_cond_dict.items():
            eps=3.
            if value == 1:
                cond_dict[key]=np.repeat([+eps],N)
            elif value == 0:
                cond_dict[key]=np.repeat([-eps],N)
            else:
                raise ValueError('pass either +1 or 0')

    #print 'getjoint: cond_dict:',cond_dict
    #print 'getjoint: do_dict:',do_dict

    #Terminology
    if model.model_type=='began':
        fake_labels=model.fake_labels
        D_fake_labels=model.D_fake_labels
        D_real_labels=model.D_real_labels
    elif model.model_type=='dcgan':
        fake_labels=model.fake_labels
        D_fake_labels=model.D_labels_for_fake
        D_real_labels=model.D_labels_for_real

    #fetch_dict={'cc_labels':model.cc.labels}
    fetch_dict={'d_fake_labels':D_fake_labels,
                'cc_labels':model.cc.labels}

    if model.model_type=='began':#dcgan not fully connected
        if not cond_dict and not do_dict:
            #Havent coded conditioning on real data
            fetch_dict.update({'d_real_labels':D_real_labels})


    print('Calculating joint distribution')
    result,_=sample(model, cond_dict=cond_dict, do_dict=do_dict,N=N,
                    fetch=fetch_dict,return_failures=False)
    print('fetd keys:',fetch_dict.keys())
    result={k:result[k] for k in fetch_dict.keys()}

    n_labels=len(model.cc.node_names)
    #list_labels=np.split( result['cfl'],n_labels, axis=1)
    #list_d_fake_labels=np.split(result['dfl'],n_labels, axis=1)
    #list_d_real_labels=np.split(result['drl'],n_labels, axis=1)

    for k in result.keys():
        print('valshape',result[k].shape)
        print('result',result[k])
    list_result={k:np.split(val,n_labels, axis=1) for k,val in result.items()}

    pd_joint={}
    for key,r in list_result.items():
        joint={}
        for name,val in zip(model.cc.node_names,r):
            int_val=(val>0.5).astype('int')
            joint[name]=int_val.ravel()
        pd_joint[key]=pd.DataFrame.from_dict(joint)

    return pd_joint


    for name, lab, dfl in zip(model.cc.node_names,list_labels,list_d_fake_labels):
        if return_discrete:
            cfl_val=(lab>0.5).astype('int')
            dfl_val=(dfl>0.5).astype('int')

        joint['dfl'][name]=dfl_val
        joint['cfl'][name]=cfl_val


    cfl=pd.DataFrame.from_dict( {k:val.ravel() for k,val in joint['cfl'].items()} )
    dfl=pd.DataFrame.from_dict( {k:val.ravel() for k,val in joint['cfl'].items()} )

    print('get_joint successful')
    return cfl,dfl



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
            print('Unexpected difficulty reshaping inputs:',tensor.name, tf_shape, len(value), np.size(value))
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
    print("L is " + str(L))
    print(p_a_dict)

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


def did_succeed( output_dict, cond_dict ):
    '''
    Used in rejection sampling:
    for each row, determine if cond is satisfied
    for every cond in cond_dict

    success is hardcoded as being more extreme
    than the condition specified
    '''
    test_key=cond_dict.keys()[0]
    #print('output_dict:',np.squeeze(output_dict[test_key]))
    #print('cond_dict:',cond_dict[test_key])


    #definition success:
    def is_win(key):
        cond=np.squeeze(cond_dict[key])
        val=np.squeeze(output_dict[key])
        cond1=np.sign(val)==np.sign(cond)
        cond2=np.abs(val)>np.abs(cond)
        return cond1*cond2


    scoreboard=[is_win(key) for key in cond_dict]
    #print('scoreboard', scoreboard)
    all_victories_bool=np.logical_and.reduce(scoreboard)
    return all_victories_bool.flatten()


def sample(model, cond_dict=None, do_dict=None, fetch_dict=None,N=None,
           on_logits=True,return_failures=True):
    '''
    fetch_dict should be a dict of tensors to do sess.run on
    do_dict is a list of strings or tensors of the form:
    {'Male':1, model.z_gen:[0,1], model.cc.Smiling:[0.1,0.9]}

    N is used only if cond_dict and do_dict are None
    '''

    do_dict= do_dict or {}
    cond_dict= cond_dict or {}
    fetch_dict=fetch_dict or {'G':model.G}

    ##Handle the case where len querry doesn't divide batch_size
    #a_dict=cond_dict or do_dict
    #if a_dict:
    #    nsamples=len(a_dict.values()[0])
    #elif N:
    #    nsamples=N
    #else:
    #    raise ValueError('either pass a dictionary or N')


    ##Pad to be batch_size divisible
    #npad=(64-nsamples)%64
    #if npad>0:
    #    print("Warn. nsamples doesnt divide batch_size, pad=",npad)
    ##N+=npad

    #if npad>0:
    #    if do_dict:
    #        for k in do_dict.keys():
    #            keypad=np.tile(do_dict[k][0],[npad])
    #            do_dict[k]=np.concatenate([do_dict[k],keypad])

    #    if cond_dict:
    #        for k in cond_dict.keys():
    #            keypad=np.tile(cond_dict[k][0],[npad])
    #            cond_dict[k]=np.concatenate([cond_dict[k],keypad])

    verbose=False
    #verbose=True



    feed_dict = do2feed(do_dict, model, on_logits=on_logits)#{tensor:array}
    cond_fetch_dict= cond2fetch(cond_dict,model,on_logits=on_logits) #{string:tensor}
    fetch_dict.update(cond_fetch_dict)


    #print('actual cond_dict', cond_dict )#{}
    #print('actual do_dict', do_dict )#{}

    if verbose:
        print('feed_dict',feed_dict)
        print('fetch_dict',fetch_dict)

    if not cond_dict and do_dict:
        #Simply do intervention w/o loop
        if verbose:
            print('sampler mode:Interventional')

        #fds=chunks(feed_dict,model.batch_size)
        fds=chunks(feed_dict,model.default_batch_size)

        outputs={k:[] for k in fetch_dict.keys()}
        for fd in fds:
            out=model.sess.run(fetch_dict, fd)
            #outputs.append(out['G'])
            for k,val in out.items():
                outputs[k].append(val)

        for k in outputs.keys():
            outputs[k]=np.vstack(outputs[k])[:nsamples]
        return outputs,feed_dict
        #return np.vstack(outputs), feed_dict

    elif not cond_dict and not do_dict:
        #neither passed, but get N samples
        assert(N>0)
        if verbose:
            print('sampling model N=',N,' times')

        ##Should be variable batch_size allowed
        outputs=model.sess.run(fetch_dict,{model.batch_size:N})

        ##fds=chunks({'idx':range(npad+N)},model.batch_size)
        #fds=chunks({'idx':range(npad+N)},model.default_batch_size)

        #outputs={k:[] for k in fetch_dict.keys()}
        #for fd in fds:
        #    out=model.sess.run(fetch_dict)
        #    for k,val in out.items():
        #        outputs[k].append(val)
        #for k in outputs.keys():
        #    outputs[k]=np.vstack(outputs[k])[:nsamples]
        #return outputs, feed_dict

        return outputs


    #elif cond_dict and not do_dict:
    elif cond_dict:
    #Could also pass do_dict here to be interesting
        ##Implements rejection sampling
        if verbose:
            print('sampler mode:Conditional')
            print('conddict',cond_dict)

        rows=np.arange( len(cond_dict.values()[0]))#what idx do we need
        assert(len(rows)>=model.batch_size)#should already be true.

        if verbose:
            print('nrows:',len(rows))

        #init
        max_fail=4000
        #max_fail=10000
        n_fails=np.zeros_like(rows)
        remaining_rows=rows.copy()
        completed_rows=[]
        bad_rows=set()

        #null=lambda :[-1 for r in rows]
        if verbose:
            print('cond fetch_dict',fetch_dict)
        outputs={key:[np.zeros(fetch_dict[key].get_shape().as_list()[1:]) for r in rows] for key in fetch_dict}
        if verbose:
            print('n keys in outputs:',len(outputs.keys()))

        #debug()

        ii=0
        while( len(remaining_rows)>0 ):
            #debug()
            ii+=1
            #loop
            if not return_failures:
                if len(completed_rows)>=nsamples:
                    if verbose:
                        print('Have enough for now; breaking')
                    break
            iter_rows=remaining_rows[:model.batch_size]
            n_pad = model.batch_size - len(iter_rows)
            if verbose:
                print('Iter:',ii, 'to go:',len(iter_rows))
                #print('iter_rows:',len(iter_rows),':',iter_rows)
            #iter_rows.extend( [iter_rows[-1]]*n_pad )#just duplicate
            pad_iter_rows=list(iter_rows)+ ( [iter_rows[-1]]*n_pad )

            iter_rows=np.array(iter_rows)
            pad_iter_rows=np.array(pad_iter_rows)

            fed=slice_dict( feed_dict, pad_iter_rows )
            cond=slice_dict( cond_dict, pad_iter_rows )

            out=model.sess.run(fetch_dict, fed)

            bool_pass = did_succeed(out,cond)[:len(iter_rows)]
            if verbose:
                print('bool_pass:',len(bool_pass),':',bool_pass)
            pass_idx=iter_rows[bool_pass]
            fail_idx=iter_rows[~bool_pass]


            #yuck
            for key in out:
                for i,row_pass in enumerate(bool_pass):
                    idx=iter_rows[i]
                    if row_pass:
                        outputs[key][idx]=out[key][i]
                    else:
                        n_fails[idx]+=1

            good_rows=set( iter_rows[bool_pass] )
            completed_rows.extend(list(good_rows))
            #print('good_rows',good_rows)
            bad_rows=set( rows[ n_fails>=max_fail ] )
            #print('bad_rows',bad_rows)

            for key in out:
                for idx_giveup in bad_rows:
                    shape=fetch_dict[key].get_shape().as_list()[1:]
                    outputs[key][idx_giveup]=np.zeros(shape)
                    if verbose:
                        print('key:',key,' shape giveup:',shape)


            ##Remove rows
            remaining_rows=list( set(remaining_rows)-good_rows-bad_rows )

            #debug()

        if verbose:
            print('conditioning took',ii,' tries')
            n_fails.sort()
            print('10 most fail counts(limit=',max_fail,'):',n_fails[-10:])

        if verbose:
            print('means:')
            for k in outputs.keys():
                for v in outputs[k]:
                    print(np.mean(v))


        if not return_failures:
            #useful for pdf calculations.
            #not useful for image grids
            if verbose:
                print('Not returning failures!..',)
            for k in outputs.keys():
                outputs[k]=[outputs[k][i] for i in completed_rows]
                if verbose:
                    print('..Returning', len(completed_rows),'/',len(cond_dict.values()[0]))
        else:
            for k in outputs.keys():
                outputs[k]=outputs[k][:nsamples]

        for k in outputs.keys():
            if verbose:
                print('tobestacked:',len(outputs[k]))
                print('tobestacked:',isinstance(outputs[k][0],np.ndarray))

            values=outputs[k][:nsamples]
            if verbose:
                for v in values:
                    try:
                        print(v.shape)
                    except:
                        print(type(v))

            if len(fetch_dict[k].get_shape().as_list())>1:
                outputs[k]=np.stack(outputs[k])
            else:
                outputs[k]=np.concatenate(outputs[k])


        return outputs,cond_dict

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
        print(n_defaults,' default values given..using 8 for each of them')

    try:
        for key,value in cond_dict.items():
            if value == 'model_default':
                print('Warning! using 1/2*model.intervention_range\
                      to specify the conditioning defaults')
                cond_min,cond_max=model.intervention_range[key]
                #cond_dict[key]=np.linspace(cond_min,cond_max,8)
                cond_dict[key]=[0.5*cond_min,0.5*cond_max]
                print('Condition dict used:',cond_dict)

            elif value=='int':
                #for integer pretrained models
                #eps=0.1 #usually logits are around 4-20
                eps=3 #usually logits are around 4-10
                #sigmoid(3) ~ 0.95
                cond_dict[key]=np.repeat([+eps,-eps],64) #logit on either size of 0
            elif value=='percentile':
                ##I'm changing this to do 50th percentile
                #of positive or of negative class
                print('calculating percentile...')
                data=[]
                for _ in range(30):
                    data.append(model.sess.run(model.cc.node_dict[key].label_logit))
                D=np.vstack(data)
                pos_logits,neg_logits=D[D>0], D[D<0]
                print("Conditioning on 5th percentile")
                pos_intv = np.percentile(pos_logits,5)
                neg_intv = np.percentile(neg_logits,95)
                cond_dict[key]=np.repeat([pos_intv,neg_intv],64)
                print('percentile5 for',key,'is',np.percentile(D,5))
                print('percentile25 for',key,'is',np.percentile(D,25))
                print('percentile50 for',key,'is',np.percentile(D,50))
                print('percentile75 for',key,'is',np.percentile(D,75))
                print('percentile95 for',key,'is',np.percentile(D,95))

                #OLD:
                ##fetch=cond2fetch(cond_dict)
                #print('...calculating percentile')
                #data=[]
                #for _ in range(30):
                #    data.append(model.sess.run(model.cc.node_dict[key].label_logit))
                #D=np.vstack(data)
                #print('dat',D.flatten())
                #cond_dict[key]=np.repeat([np.percentile(D,95),np.percentile(D,5)],64)
                #print('percentiles for',key,'are',[np.percentile(D,5),np.percentile(D,95)])


            else:
                #otherwise pass a number, list, or array
                assert(not isinstance(value,str))

    except Exception, e:
        raise(e,'Difficulty accessing default model interventions')


    str_step=str(step)

    lengths = [ len(v) for v in cond_dict.values() if hasattr(v,'__len__') ]
    #print('lengths',lengths)
    print('lengths',lengths)

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
            #size=[N/8,8]
            size=[8,N/8]
        else:
            size=[8,8]



    #Terminology
    if model.model_type=='began':
        result_dir=model.model_dir
        if str_step=='':
            str_step=str( model.sess.run(model.step) )+'_'
    elif model.model_type=='dcgan':
        print('DCGAN')
        result_dir=model.checkpoint_dir

    sample_dir=os.path.join(result_dir,'sample_figures')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    out, _= sample(model, cond_dict=cond_dict,on_logits=on_logits)
    images=out['G']

    #print('Images shape:',images.shape)


    #cond_file=os.path.join(sample_dir, str_step+str(cond_dict_name)+'_cond'+'.png')
    cond_file=os.path.join(sample_dir,str_step+str(cond_dict_name)+'_cond'+'.pdf')

    #if os.path.exists(cond_file):
    #    cond_file='new'+cond_file #don't overwrite

    save_figure_images(model.model_type,images,cond_file,size=size)


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
        print(n_defaults,' default values given..using 8 for each of them')

    try:
        for key,value in do_dict.items():
            if value == 'model_default':
                itv_min,itv_max=model.intervention_range[key]
                do_dict[key]=np.linspace(itv_min,itv_max,8)

            elif value=='int':
                #for integer pretrained models
                #eps=0.1 #usually logits are around 4-20
                eps=3 #usually logits are around 4-10
                #sigmoid(3) ~ 0.95
                do_dict[key]=np.repeat([-eps,+eps],64) #logit on either size of 0

            elif value=='percentile':
                ##I'm changing this to do 50th percentile
                #of positive or of negative class
                print('calculating percentile...')
                data=[]
                for _ in range(30):
                    data.append(model.sess.run(model.cc.node_dict[key].label_logit))
                D=np.vstack(data)
                pos_logits,neg_logits=D[D>0], D[D<0]
                pos_intv = np.percentile(pos_logits,50)
                neg_intv = np.percentile(neg_logits,50)
                do_dict[key]=np.repeat([pos_intv,neg_intv],64)
                print('percentile5 for',key,'is',np.percentile(D,5))
                print('percentile25 for',key,'is',np.percentile(D,25))
                print('percentile50 for',key,'is',np.percentile(D,50))
                print('percentile75 for',key,'is',np.percentile(D,75))
                print('percentile95 for',key,'is',np.percentile(D,95))
            else:
                #otherwise pass a number, list, or array
                assert(not isinstance(value,str))

    except Exception, e:
        raise(e,'Difficulty accessing default model interventions')


    str_step=str(step)

    lengths = [ len(v) for v in do_dict.values() if hasattr(v,'__len__') ]
    #print('lengths',lengths)
    print('lengths',lengths)

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
            #size=[N/8,8]
            size=[8,N/8]
        else:
            size=[8,8]

    #Terminology
    if model.model_type=='began':
        result_dir=model.model_dir
        if str_step=='':
            str_step=str( model.sess.run(model.step) )+'_'
    elif model.model_type=='dcgan':
        print('DCGAN')
        result_dir=model.checkpoint_dir

    sample_dir=os.path.join(result_dir,'sample_figures')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    #print('do_dict DEBUG:',do_dict)
    out, feed_dict= sample(model, do_dict=do_dict,on_logits=on_logits)
    images=out['G']


    itv_file=os.path.join(sample_dir, str_step+str(do_dict_name)+'_intv'+'.pdf')
    #itv_file=os.path.join(sample_dir, str_step+str(do_dict_name)+'_intv'+'.png')

    #if os.path.exists(itv_file):
    #    itv_file='new'+itv_file #don't overwrite

    save_figure_images(model.model_type,images,itv_file,size=size)






