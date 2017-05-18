import tensorflow as tf
import numpy as np
import os
import scipy.misc
import numpy as np
from tqdm import trange,tqdm
import pandas as pd
from itertools import combinations, product
import sys
from utils import save_figure_images,make_sample_dir

from sample import get_joint








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

            lab_fname=str_step+str(name)+str(int_val)
            dfl_fname=str_step+str(name)+str(int_val)

with open(dfl_xtab_fn,'w') as dlf_f






