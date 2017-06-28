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
from sample import get_joint,sample,find_logit_percentile



'''
This is a file where each function creates a particular figure. No real need
for this to be configurable. Just make a new function for each figure

This uses functions in sample.py and distribution.py, which are intended to
be lower level functions that can be used more generally.

'''




def fig1(model, output_folder):
    '''
    This function makes two 2x10 images
    showing the difference between conditioning
    and intervening
    '''

    str_step=guess_model_step(model)
    fname=os.path.join(output_folder,str_step+model.model_type)

    for key in ['Mustache','Bald']:
        p50,n50=find_logit_percentile(model,key,50)

        do_dict={key:np.repeat([p50,n50],10)}
        eps=3
        cond_dict={key:np.repeat([+eps,-eps],10)}

        out,_=sample(model,do_dict=do_dict)
        intv_images=out['G']

        out,_=sample(model,cond_dict=cond_dict)
        cond_images=out['G']

        itv_file=fname+'_'+key+'intv.pdf'
        cond_file=fname+'_'+key+'cond.pdf'

        save_figure_images(model.model_type,intv_images,itv_file,size=[2,10])
        save_figure_images(model.model_type,cond_images,cond_file,size=[2,10])



