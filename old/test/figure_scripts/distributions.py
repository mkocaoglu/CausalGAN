import tensorflow as tf
import numpy as np
import os
import scipy.misc
import numpy as np
import pandas as pd
from tqdm import trange,tqdm
import pandas as pd
from itertools import combinations, product
import sys
from utils import save_figure_images,make_sample_dir,guess_model_step
from sample import get_joint,sample



def get_pdf(model, do_dict=None,cond_dict=None,name='',N=6400,return_discrete=True,step=''):
    str_step=str(step) or guess_model_step(model)

    joint=get_joint(model,int_do_dict=do_dict,int_cond_dict=cond_dict,N=N,return_discrete=return_discrete)

    sample_dir=make_sample_dir(model)

    if name:
        name+='_'
    f_pdf=os.path.join(sample_dir,str_step+name+'dist'+'.csv')

    pdf=pd.DataFrame.from_dict({k:val.mean() for k,val in joint.items()})

    #print 'get pdf cond_dict:',cond_dict
    if not do_dict and not cond_dict:
        data=model.attr.mean()
        pdf['data']=data
    if not do_dict and cond_dict:
        bool_cond=np.logical_and.reduce([model.attr[k]==v for k,v in cond_dict.items()])
        attr=model.attr[bool_cond]
        pdf['data']=attr.mean()

    print 'Writing to file',f_pdf
    pdf.to_csv(f_pdf)

    return pdf


TINY=1e-6
def get_interv_table(model,intrv=True):

    n_batches=25
    table_outputs=[]
    d_vals=np.linspace(TINY,0.6,n_batches)
    for name in model.cc.node_names:
        outputs=[]
        for d_val in d_vals:
            do_dict={model.cc.node_dict[name].label_logit : d_val*np.ones((model.batch_size,1))}
            outputs.append(model.sess.run(model.fake_labels,do_dict))

        out=np.vstack(outputs)
        table_outputs.append(out)

    table=np.stack(table_outputs,axis=2)

    np.mean(np.round(table),axis=0)

    return table

#dT=pd.DataFrame(index=p_names, data=T, columns=do_names)
#T=np.mean(np.round(table),axis=0)
#table=get_interv_table(model)



def record_interventional(model,step=''):
    '''
    designed for truncated exponential noise.
    For each node that could be intervened on,
    sample interventions from the continuous
    distribution that discrete intervention
    corresponds to. Collect the joint and output
    to a csv file
    '''
    make_sample_dir(model)

    str_step=str(step)
    if str_step=='':
        if hasattr(model,'step'):
            str_step=str( model.sess.run(model.step) )+'_'

    m=20
    do =lambda val: np.linspace(0,val*0.8,m)
    for name in model.cc.node_names:
        for int_val,intv in enumerate([do(-1), do(+1)]):
            do_dict={name:intv}

            joint=get_joint(model, do_dict=None, N=5,return_discrete=True,step='')

            lab_df=pd.DataFrame(data=joint['g_fake_label'])
            dfl_df=pd.DataFrame(data=joint['d_fake_label'])

            lab_fname=str_step+str(name)+str(int_val)+'.csv'
            dfl_fname=str_step+str(name)+str(int_val)+'.csv'

            lab_df.to_csv(lab_fname)
            dfl_df.to_csv(dfl_fname)

    #with open(dfl_xtab_fn,'w') as dlf_f, open(lab_xtab_fn,'w') as lab_f:






