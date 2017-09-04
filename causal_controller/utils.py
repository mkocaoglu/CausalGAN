from __future__ import print_function
import numpy as np
import tensorflow as tf

def summary_stats(name,tensor,collections=None,hist=False):
    collections=collections or [tf.GraphKeys.SUMMARIES]
    ave=tf.reduce_mean(tensor)
    std=tf.sqrt(tf.reduce_mean(tf.square(ave-tensor)))
    tf.summary.scalar(name+'_ave',ave,collections)
    tf.summary.scalar(name+'_std',std,collections)
    if hist:
        tf.summary.histogram(name+'_hist',tensor,collections)

def did_succeed( output_dict, cond_dict ):
    '''
    Used in rejection sampling:
    for each row, determine if cond is satisfied
    for every cond in cond_dict

    success is hardcoded as round(label) being exactly equal
    to the integer in cond_dict
    '''

    #definition success:
    def is_win(key):
        #cond=np.squeeze(cond_dict[key])
        cond=np.squeeze(cond_dict[key])
        val=np.squeeze(output_dict[key])
        condition= np.round(val)==cond
        return condition

    scoreboard=[is_win(key) for key in cond_dict]
    #print('scoreboard', scoreboard)
    all_victories_bool=np.logical_and.reduce(scoreboard)
    return all_victories_bool.flatten()

