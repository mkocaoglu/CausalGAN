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


'''
    Sometimes I leave notes here as a way to see what the motivation was for the
    previous models and to highlight what code changes were made. File is copied
    into log directory.
'''


'''
TODO:
    load config from json when load model
    it seems like to make multi gpu functional again, causal_controller has to be created twice

    CausalGAN: to test out:
        config.label_type='discrete'
        type_input_to_generator='labels'
        stab_proj variants

    Setup image saving for label_mode collapse
'''


'''
z reconstruction was removed and results seem to be improved.

recent change to how batch_size is handled
    cc_config.batch_size=dcgan_config.batch_size


'''


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
    tf.set_random_seed(config.seed)


    prepare_dirs_and_logger(config)
    if not config.load_path:
        print('saving config because load path not given')
        save_configs(config,cc_config,dcgan_config,began_config)


    #Resolve model differences and batch_size
    if config.model_type:
        if config.model_type=='dcgan':
            config.batch_size=dcgan_config.batch_size
            cc_config.batch_size=dcgan_config.batch_size
            config.Model=CausalGAN.CausalGAN
            model_config=dcgan_config
        if config.model_type=='began':
            config.batch_size=began_config.batch_size
            cc_config.batch_size=began_config.batch_size
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

    #so ipython isn't interupted annoyingly
    #I think each queue creates a step/sec INFO warning
    tf.logging.set_verbosity(tf.logging.ERROR)


    #I wish there were a way to tell supervisor to allow further graph
    #modifications so that if running this in ipython, more tensors could be created after




