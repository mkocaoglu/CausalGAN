import tensorflow as tf
import os
import scipy.misc
import numpy as np
from tqdm import trange

import pandas as pd
from itertools import combinations
import sys



def crosstab(model,step=None):
    '''
    This is a script for outputing [0,1/2], [1/2,1] binned pdfs
    including the marginals and the pairwise comparisons

    '''
    if step is None:
        str_step=''
    else:
        str_step=str(step)+'_'

    n_labels=len(model.cc.nodes)

    N=50*n_labels*64 #N may need to scale higher than this
    #N=100#to go quicker

    print 'Calculating joint distribution with',
    print 'N=',N,' samples'

    n_batches=N/model.batch_size
    labels, d_fake_labels= [],[]
    #Terminology
    if model.model_name=='began':
        fake_labels=model.fake_labels
        #D_fake_labels=model.D_fake_labels
        result_dir=model.model_dir
        if str_step=='':
            str_step=str( model.sess.run(model.step) )+'_'
    elif model.model_name=='dcgan':
        fake_labels=model.fake_labels
        #D_fake_labels=model.D_labels_for_fake
        result_dir=model.checkpoint_dir

    if not os.path.exists(result_dir):
        raise ValueError('result_dir:',result_dir,' does not exist')

    #dfl_xtab_fn = os.path.join(result_dir,str_step+'d_fake_label_crosstab.txt')
    lab_xtab_fn = os.path.join(result_dir,str_step+'glabel_crosstab.txt')
    #gvsd_xtab_fn = os.path.join(result_dir,str_step+'glabel_vs_dfake_crosstab.txt')


    #for n in range(n_batches):
    for step in trange(n_batches):
        lab=model.sess.run(fake_labels)
        #lab,dfl=model.sess.run([fake_labels,D_fake_labels])
        labels.append(lab)
        #d_fake_labels.append(dfl)

    list_labels=np.split( np.vstack(labels),n_labels, axis=1)
    #list_d_fake_labels=np.split( np.vstack(d_fake_labels),n_labels, axis=1)


    joint={}
    label_joint={}
    dfl_joint={}#d_fake_label
    #for name, lab, dfl in zip(model.cc.node_names,list_labels,list_d_fake_labels):
    for name, lab in zip(model.cc.node_names,list_labels):
        #Create dictionary:
            #node_name -> 
                        #'g_fake_label'
                        #'d_fake_label'
        joint[name]={
                'g_fake_label':lab,
                #'d_fake_label':dfl
                }

    print 'Writing to files:',
    #print dfl_xtab_fn,
    print lab_xtab_fn,
    #print gvsd_xtab_fn

    #Make a cross table for every pair of labels and save that to csv
    #with open(dfl_xtab_fn,'w') as dlf_f, open(lab_xtab_fn,'w') as lab_f, open(gvsd_xtab_fn,'w') as gldf_f:
    with open(lab_xtab_fn,'w') as lab_f:
        #dlf_f.write('Marginals:\n')
        lab_f.write('Marginals:\n')
        #gldf_f.write('Pairwise g_label vs d_fake\n')

        #Marginals and Pairwise-1-node
        for name in joint.keys():
            #dlf_f.write('Node: '+name+'\n')
            lab_f.write('Node: '+name+'\n')
            #gldf_f.write('Node: '+name+'\n')

            lab_marg=(joint[name]['g_fake_label'] > 0.5).astype('int')
            #dlf_marg=(joint[name]['d_fake_label'] > 0.5).astype('int')

            lab_f.write('  mean='+str(np.mean(lab_marg))+'\n')
            #dlf_f.write('  mean='+str(np.mean(dlf_marg))+'\n')

            #gldf_df=pd.DataFrame(data=np.hstack([lab_marg,dlf_marg]),columns=['glabel','dfake'])
            #gldf_ct=pd.crosstab(index=gldf_df['glabel'],columns=gldf_df['dfake'],margins=True)
            #gldf_ct/=N

            #gldf_f.write( gldf_ct.__repr__() )

            #dlf_f.write('\n')
            lab_f.write('\n')
            #gldf_f.write('\n\n')


        #dlf_f.write('\nPairwise:\n')
        lab_f.write('\nPairwise:\n')

        for node1,node2 in combinations(joint.keys(),r=2):

            lab_node1=(joint[node1]['g_fake_label']>0.5).astype('int')
            lab_node2=(joint[node2]['g_fake_label']>0.5).astype('int')
            lab_df=pd.DataFrame(data=np.hstack([lab_node1,lab_node2]),columns=[node1,node2])
            lab_ct=pd.crosstab(index=lab_df[node1],columns=lab_df[node2],margins=True)
            lab_ct/=N

            lab_ct.to_csv(lab_xtab_fn,mode='a')
            lab_f.write( lab_ct.__repr__() )
            lab_f.write('\n\n')


            #dlf_node1=(joint[node1]['d_fake_label']>0.5).astype('int')
            #dlf_node2=(joint[node2]['d_fake_label']>0.5).astype('int')
            #dlf_df=pd.DataFrame(data=np.hstack([dlf_node1,dlf_node2]),columns=[node1,node2])
            #dlf_ct=pd.crosstab(index=dlf_df[node1],columns=dlf_df[node2],margins=True)
            #dlf_ct/=N

            #dlf_ct.to_csv(dlf_xtab_fn,mode='a')#not aligned
            #dlf_f.write( dlf_ct.__repr__() )
            #dlf_f.write('\n\n')


