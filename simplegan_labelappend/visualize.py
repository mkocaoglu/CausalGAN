import tensorflow as tf
import os
import scipy.misc
import numpy as np
from tqdm import trange
import argparse
import pandas as pd
from itertools import combinations
import sys

from model_loader import get_model
import json
from figure_scripts.pairwise import crosstab
from figure_scripts.sample import intervention2d,condition2d
from causal_intervention import get_do_dict
from causal_conditioning import get_cond_dict

def str2bool(v):
    #return (v is True) or (v.lower() in ('true', '1'))
    return v is True or v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

#sample_arg = add_argument_group('Sample')
#sample_arg.add_argument('--do_dict', type=str, help='pass as \'{"key1":"value')

visualize_arg = add_argument_group('visualize')
visualize_arg.add_argument('--model_type', type=str,default=None)

#Which visualizations to do?
visualize_arg.add_argument('--cross_tab',type=str2bool,default=False,\
                          help='Tabulates pairwise marginal distributions\
                          and saves them to text files')
visualize_arg.add_argument('--sample_model', type=str2bool,default=False,\
                          help='Tells program to do sampling for do_dict and\
                           other arguments provided. Run this to do\
                           intervention2d for example')


visualize_arg.add_argument('--do_dict_name',type=str, default=None)
visualize_arg.add_argument('--cond_dict_name',type=str, default=None)

#I'm strongly worried this line will override flags in main.py so I commented it out
#visualize_arg.add_argument('--checkpoint_dir',type=str, default=None)
#Ref: flags:
#flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

if __name__=='__main__':
    '''
    Current instructions:

    In addition to whatever flags you would normally use to run the model, use
    flags to specify model_type, do_dict (if applicable), and whatever figures
    you would like to produce (followed by True)

    Examples:

    To run intervention2d, the following works:

    #Tested
    python visualize.py --model_type dcgan --sample_model True
    --do_dict_name second_example_dict --dataset celebA --input_height 108
    --is_train False --is_crop True --graph male_causes_mustache --checkpoint_dir
    ./checkpoint/male_c_mustache


    to run crosstab, in addition provide --cross_tab True:
    python visualize.py --model_type dcgan --sample_model True --cross_tab True
    --do_dict_name second_example_dict --dataset celebA --input_height 108
    --is_train False --is_crop True --graph male_causes_mustache --checkpoint_dir
    ./checkpoint/male_c_mustache



    This can also be run from ipython:
    ipython:
    %run visualize.py --model_type 'dcgan' --sample_model True --dataset 'celebA' --input_height=108
    --is_train=False --is_crop True --graph 'big_causal_graph' --checkpoint_dir
    './checkpoint/big_causal1'


    #Tested
    %run visualize.py --model_type 'began' --sample_model True  --do_dict_name
    'gender_lipstick_default' --causal_graph 'big_causal_graph' --is_train False
    --load_path 'celebA_0507_222545'

    '''


    config, unparsed = parser.parse_known_args()
    print 'The config you passed to visualize:',config

    #Get model
    model_type= config.model_type
    model=get_model(model_type)
    model.model_type=model_type
    if config.cross_tab:
        crosstab(model)

    if config.sample_model:
        if config.cond_dict_name and config.do_dict_name:
            raise ValueError('simultaneous condition and intervention not supported')
        if config.do_dict_name:
            do_dict=get_do_dict( config.do_dict_name )
            intervention2d( model, do_dict=do_dict, do_dict_name=config.do_dict_name, on_logits=True)

        elif config.cond_dict_name:
            cond_dict=get_cond_dict( config.cond_dict_name )
            condition2d( model, cond_dict=cond_dict, cond_dict_name=config.cond_dict_name, on_logits=True)

        else:
            raise ValueError('need do_dict_name xor cond_dict_name')





