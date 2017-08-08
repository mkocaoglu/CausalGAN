from __future__ import print_function
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import shutil
import sys
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from tensorflow.core.framework import summary_pb2

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def summary_stats(name,tensor,collections=None,hist=False):
    collections=collections or [tf.GraphKeys.SUMMARIES]
    ave=tf.reduce_mean(tensor)
    std=tf.sqrt(tf.reduce_mean(tf.square(ave-tensor)))
    tf.summary.scalar(name+'_ave',ave,collections)
    tf.summary.scalar(name+'_std',std,collections)
    if hist:
        tf.summary.histogram(name+'_hist',tensor,collections)


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

    if not config.load_path:
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

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type=='GPU']

def distribute_input_data(data_loader,num_gpu):
    '''
    data_loader is a dictionary of tensors that are fed into our model

    This function takes that dictionary of n*batch_size dimension tensors
    and breaks it up into n dictionaries with the same key of tensors with
    dimension batch_size. One is given to each gpu
    '''
    if num_gpu==0:
        return {'/cpu:0':data_loader}

    gpus=get_available_gpus()
    if num_gpu > len(gpus):
        raise ValueError('number of gpus specified={}, more than gpus available={}'.format(num_gpu,len(gpus)))

    gpus=gpus[:num_gpu]


    data_by_gpu={g:{} for g in gpus}
    for key,value in data_loader.items():
        spl_vals=tf.split(value,num_gpu)
        for gpu,val in zip(gpus,spl_vals):
            data_by_gpu[gpu][key]=val

    return data_by_gpu


def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
