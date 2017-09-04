from __future__ import print_function
import numpy as np
import os
import tensorflow as tf

from trainer import Trainer
from causal_graph import get_causal_graph
from utils import prepare_dirs_and_logger, save_configs

#Generic configuration arguments
from config import get_config
#Submodel specific configurations
from causal_controller.config import get_config as get_cc_config
from causal_dcgan.config import get_config as get_dcgan_config
from causal_began.config import get_config as get_began_config

from causal_began import CausalBEGAN
from causal_dcgan import CausalGAN

from IPython.core import debugger
debug = debugger.Pdb().set_trace

def get_trainer():
    print('tf: resetting default graph!')
    tf.reset_default_graph()#for repeated calls in ipython


    ####GET CONFIGURATION####
    #TODO:load configurations from previous model when loading previous model
    ##if load_path:
        #load config files from dir
    #except if pt_load_path, get cc_config from before
    #overwrite is_train, is_pretrain with current args--sort of a mess

    ##else:
    config,_=get_config()
    cc_config,_=get_cc_config()
    dcgan_config,_=get_dcgan_config()
    began_config,_=get_began_config()

    ###SEEDS###
    np.random.seed(config.seed)
    #tf.set_random_seed(config.seed) # Not working right now.

    prepare_dirs_and_logger(config)
    if not config.load_path:
        print('saving config because load path not given')
        save_configs(config,cc_config,dcgan_config,began_config)

    #Resolve model differences and batch_size
    if config.model_type:
        if config.model_type=='dcgan':
            config.batch_size=dcgan_config.batch_size
            cc_config.batch_size=dcgan_config.batch_size # make sure the batch size of cc is the same as the image model
            config.Model=CausalGAN.CausalGAN
            model_config=dcgan_config
        if config.model_type=='began':
            config.batch_size=began_config.batch_size
            cc_config.batch_size=began_config.batch_size # make sure the batch size of cc is the same as the image model
            config.Model=CausalBEGAN.CausalBEGAN
            model_config=began_config

    else:#no image model
        model_config=None
        config.batch_size=cc_config.batch_size

        if began_config.is_train or dcgan_config.is_train:
            raise ValueError('need to specify model_type for is_train=True')

    #Interpret causal_model keyword
    cc_config.graph=get_causal_graph(config.causal_model)

    #Builds and loads specified models:
    trainer=Trainer(config,cc_config,model_config)
    return trainer

def main(trainer):
    #Do pretraining
    if trainer.cc_config.is_pretrain:
        trainer.pretrain_loop()

    if trainer.model_config:
        if trainer.model_config.is_train:
            trainer.train_loop()

if __name__ == "__main__":
    trainer=get_trainer()

    #make ipython easier
    sess=trainer.sess
    cc=trainer.cc
    if hasattr(trainer,'model'):
        model=trainer.model

    main(trainer)

    tf.logging.set_verbosity(tf.logging.ERROR)