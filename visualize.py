import tensorflow as tf
import os
import scipy.misc
import numpy as np
from tqdm import trange

import pandas as pd
from itertools import combinations
import sys

from model_loader import get_model

from figure_scripts.pairwise import crosstab




if __name__=='__main__':
    '''
    ###If you're running this directly from the commandline or from ipython
    Usage for this section:
    (first word is the preferred model)

    %run visualize.py 'dcgan' --dataset 'celebA' --input_height=108
    --is_train=False --is_crop True --graph 'big_causal_graph' --checkpoint_dir
    './checkpoint/big_causal1'

#Works:
In [6]: %run visualize 'dcgan' --dataset 'celebA' --input_height=108 --is_train False
        --is_crop True --graph 'big_causal_graph' --checkpoint_dir './checkpoint/big_causal1'

    #or

    %run visualize.py 'began' --dataset 'celebA' --num_gpu=1 --num_worker=30
    --load_path 'celebA_0507_222545' --noisy_labels=True
    --indep_causal=False --separate_labeler=True --causal_model='big_causal_graph'

#Works:
In [17]: %run visualize.py 'began' --dataset 'celebA' --input_height=108
        --is_train Fals e --is_crop True --causal_model 'big_causal_graph' --num_gpu=1
        --num_worker=30 --load_path 'celebA_0507_222545' --separate_labeler=True
        --indep_causal=False --noisy_labels=True

    '''

    model_name= sys.argv[1]
    model=get_model(model_name)
    model.model_name=model_name

    crosstab(model)


