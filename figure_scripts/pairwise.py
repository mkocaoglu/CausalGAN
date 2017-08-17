from __future__ import print_function
import time
import tensorflow as tf
import os
import scipy.misc
import numpy as np
from tqdm import trange

import pandas as pd
from itertools import combinations
import sys
from sample import sample




def calc_tvd(label_dict,attr):
    '''
    attr should be a 0,1 pandas dataframe with
    columns corresponding to label names

    for example:
    names=zip(*self.graph)[0]
    calc_tvd(label_dict,attr[names])

    label_dict should be a dictionary key:1d-array of samples
    '''
    ####Calculate Total Variation####
    if np.min(attr.values)<0:
        raise ValueError('calc_tvd received \
                 attr that may not have been in {0,1}')

    label_names=label_dict.keys()
    attr=attr[label_names]

    df2=attr.drop_duplicates()
    df2 = df2.reset_index(drop = True).reset_index()
    df2=df2.rename(columns = {'index':'ID'})
    real_data_id=pd.merge(attr,df2)
    real_counts = pd.value_counts(real_data_id['ID'])
    real_pdf=real_counts/len(attr)

    label_list_dict={k:np.round(v.ravel()) for k,v in label_dict.items()}
    df_dat=pd.DataFrame.from_dict(label_list_dict)
    dat_id=pd.merge(df_dat,df2,on=label_names,how='left')
    dat_counts=pd.value_counts(dat_id['ID'])
    dat_pdf = dat_counts / dat_counts.sum()
    diff=real_pdf.subtract(dat_pdf, fill_value=0)
    tvd=0.5*diff.abs().sum()
    return tvd


def crosstab(model,result_dir=None,report_tvd=True,no_save=False,N=500000):
    '''
    This is a script for outputing [0,1/2], [1/2,1] binned pdfs
    including the marginals and the pairwise comparisons

    report_tvd is given as optional because it is somewhat time consuming

    result_dir is where to save the distribution text files. defaults to
    model.cc.model_dir

    '''
    result_dir=result_dir or model.cc.model_dir
    result={}

    n_labels=len(model.cc.nodes)

    #Not really sure how this should scale
    #N=1000*n_labels
    #N=500*n_labels**2#open to ideas that avoid a while loop
    #N=12000

    #tvd will not be reported as low unless N is large
    #N=500000 #default

    print('Calculating joint distribution with',)

    t0=time.time()
    label_dict=sample(model,fetch_dict=model.cc.label_dict,N=N)
    print('sampling model N=',N,' times took ',time.time()-t0,'sec')


    #fake_labels=model.cc.fake_labels

    str_step=str( model.sess.run(model.cc.step) )+'_'

    attr=model.data.attr
    attr=attr[model.cc.node_names]

    lab_xtab_fn = os.path.join(result_dir,str_step+'glabel_crosstab.txt')
    print('Writing to files:',lab_xtab_fn)

    if report_tvd:
        t0=time.time()
        tvd=calc_tvd(label_dict,attr)
        result['tvd']=tvd
        print('calculating tvd from samples took ',time.time()-t0,'sec')

        if no_save:
            return result

    t0=time.time()

    joint={}
    label_joint={}
    #for name, lab in zip(model.cc.node_names,list_labels):
    for name, lab in label_dict.items():
        joint[name]={ 'g_fake_label':lab }


    #with open(dfl_xtab_fn,'w') as dlf_f, open(lab_xtab_fn,'w') as lab_f, open(gvsd_xtab_fn,'w') as gldf_f:
    with open(lab_xtab_fn,'w') as lab_f:
        if report_tvd:
            lab_f.write('TVD:'+str(tvd)+'\n\n')
        lab_f.write('Marginals:\n')

        #Marginals
        for name in joint.keys():
            lab_f.write('Node: '+name+'\n')

            true_marg=np.mean((attr[name]>0.5).values)
            lab_marg=(joint[name]['g_fake_label'] > 0.5).astype('int')

            lab_f.write('  mean='+str(np.mean(lab_marg))+'\t'+\
                        'true mean='+str(true_marg)+'\n')

            lab_f.write('\n')


        #Pairs of labels
        lab_f.write('\nPairwise:\n')

        for node1,node2 in combinations(joint.keys(),r=2):

            lab_node1=(joint[node1]['g_fake_label']>0.5).astype('int')
            lab_node2=(joint[node2]['g_fake_label']>0.5).astype('int')
            lab_df=pd.DataFrame(data=np.hstack([lab_node1,lab_node2]),columns=[node1,node2])
            lab_ct=pd.crosstab(index=lab_df[node1],columns=lab_df[node2],margins=True,normalize=True)

            true_ct=pd.crosstab(index=attr[node1],columns=attr[node2],margins=True,normalize=True)


            lab_f.write('\n\tFake:\n')
            lab_ct.to_csv(lab_xtab_fn,mode='a')
            lab_f.write( lab_ct.__repr__() )
            lab_f.write('\n\tReal:\n')
            lab_f.write( true_ct.__repr__() )

            lab_f.write('\n\n')

    print('calculating pairwise crosstabs and saving results took ',time.time()-t0,'sec')
    return result










