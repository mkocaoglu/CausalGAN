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
    decide lrelu vs tanh for CC
    load config from json when load model
    pt_factorized=True doesn't work
        decide if pt_factorized is worse than without

    intervention and conditioning code
    writing conditioning right into causal controller
        #That'll be faster because don't need to generate image for rejected samples

    it seems like to make multi gpu functional again, causal_controller has to be created twice

CausalGAN: to test out:

    config.label_type='discrete'
    type_input_to_generator='labels'
'''

'''

Possible Sources of error:
    had to rewrite minibatch_features(CausalGAN)
    began data_format muckery




'''

'''

>fixed by adding noise to real_labels
Nan culprit:
    self.d_on_z_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss_on_z, var_list=self.dz_vars)
              #.minimize(self.g_loss_on_z + self.rec_loss_coeff*self.real_reconstruction_loss, var_list=self.dz_vars)
even though self.rec_loss_coeff=0.0, droping this term got rid of nan loss


Running pretraining on 13 label graph

'''


def get_trainer():
    print('tf: resetting default graph!')
    tf.reset_default_graph()#for repeated calls in ipython


    #TODO:
    ##if load_path:
        #load config files from dir
    #except if pt_load_path, get cc_config from before

    ##else:
    config,_=get_config()
    cc_config,_=get_cc_config()
    dcgan_config,_=get_dcgan_config()
    began_config,_=get_began_config()

    print('factorized:',cc_config.pt_factorized)

    prepare_dirs_and_logger(config)
    if not config.load_path:
        print('saving config because load path not given')
        save_configs(config,cc_config,dcgan_config,began_config)


    #Resolve model differences and batch_size
    if config.model_type:
        if config.model_type=='dcgan':
            config.batch_size=dcgan_config.batch_size
            config.Model=CausalGAN.CausalGAN
            model_config=dcgan_config
        if config.model_type=='began':
            config.batch_size=began_config.batch_size
            config.Model=CausalBEGAN.CausalBEGAN
            model_config=began_config
    else:#no image model
        model_config=None
        config.batch_size=cc_config.batch_size

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



    #I wish there were a way to tell supervisor to allow further graph
    #modifications so that if running this in ipython, more tensors could be created after

