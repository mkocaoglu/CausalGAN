from __future__ import print_function
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from skimage import io
import shutil
import sys
import math
import time
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from tensorflow.core.framework import summary_pb2
import matplotlib.pyplot as plt

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def summary_losses(sess,model,N=1000):
    step,loss_g,loss_d=sess.run([model.step,model.loss_g,model.loss_d],{model.data.N:N,model.gen.N:N})
    lgsum=make_summary(model.data.name+'_gloss',loss_g)
    ldsum=make_summary(model.data.name+'_dloss',loss_d)
    return step,lgsum, ldsum

def calc_tvd(sess,Generator,Data,N=50000,nbins=10):
    Xd=sess.run(Data.X,{Data.N:N})
    step,Xg=sess.run([Generator.step,Generator.X],{Generator.N:N})

    p_gen,_ = np.histogramdd(Xg,bins=nbins,range=[[0,1],[0,1],[0,1]],normed=True)
    p_dat,_ = np.histogramdd(Xd,bins=nbins,range=[[0,1],[0,1],[0,1]],normed=True)
    p_gen/=nbins**3
    p_dat/=nbins**3
    tvd=0.5*np.sum(np.abs( p_gen-p_dat ))
    mvd=np.max(np.abs( p_gen-p_dat ))

    return step,tvd, mvd

    s_tvd=make_summary(Data.name+'_tvd',tvd)
    s_mvd=make_summary(Data.name+'_mvd',mvd)

    return step,s_tvd,s_mvd
    #return make_summary('tvd/'+Generator.name,tvd)


def summary_stats(name,tensor,hist=False):
    ave=tf.reduce_mean(tensor)
    std=tf.sqrt(tf.reduce_mean(tf.square(ave-tensor)))
    tf.summary.scalar(name+'_ave',ave)
    tf.summary.scalar(name+'_std',std)
    if hist:
        tf.summary.histogram(name+'_hist',tensor)

def summary_scatterplots(X1,X2,X3):
    with tf.name_scope('scatter'):
        img1=summary_scatter2d(X1,X2,'X1X2',xlabel='X1',ylabel='X2')
        img2=summary_scatter2d(X1,X3,'X1X3',xlabel='X1',ylabel='X3')
        img3=summary_scatter2d(X2,X3,'X2X3',xlabel='X2',ylabel='X3')
        plt.close()
    return img1,img2,img3



def summary_scatter2d(x,y,title='2dscatterplot',xlabel=None,ylabel=None):
    fig=scatter2d(x,y,title,xlabel=xlabel,ylabel=ylabel)

    fig.canvas.draw()
    rgb=fig.canvas.tostring_rgb()
    buf=np.fromstring(rgb,dtype=np.uint8)

    w,h = fig.canvas.get_width_height()
    img=buf.reshape(1,h,w,3)
    #summary=tf.summary.image(title,img)
    plt.close(fig)
    #fig.clf()
    return img

def scatter2d(x,y,title='2dscatterplot',xlabel=None,ylabel=None):
    fig=plt.figure()
    plt.scatter(x,y)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if not 0<=np.min(x)<=np.max(x)<=1:
        raise ValueError('summary_scatter2d title:',title,' input x exceeded [0,1] range.\
                         min:',np.min(x),' max:',np.max(x))
    if not 0<=np.min(y)<=np.max(y)<=1:
        raise ValueError('summary_scatter2d title:',title,' input y exceeded [0,1] range.\
                         min:',np.min(y),' max:',np.max(y))

    plt.xlim([0,1])
    plt.ylim([0,1])
    return fig


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    if config.is_train:
        config.log_code_dir=os.path.join(config.model_dir,'code')
        for path in [config.log_dir, config.data_dir,
                     config.model_dir, config.log_code_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        #Copy python code in directory into model_dir/code for future reference:
        code_dir=os.path.dirname(os.path.realpath(sys.argv[0]))
        model_files = [f for f in listdir(code_dir) if isfile(join(code_dir, f))]
        for f in model_files:
            if f.endswith('.py'):
                shutil.copy2(f,config.log_code_dir)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)



class Timer(object):
    def __init__(self):
        self.total_section_time=0.
        self.iter=0
    def on(self):
        self.t0=time.time()
    def off(self):
        self.total_section_time+=time.time()-self.t0
        self.iter+=1
    def __str__(self):
        n_min=self.total_section_time/60.
        return '%.2fmin'%n_min
