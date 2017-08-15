from __future__ import print_function
import numpy as np
import tensorflow as tf

from trainer import Trainer
from causal_graph import get_causal_graph
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

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

    Try leaky relu activation inside causal_controller instead of tanh
    save .py files in subfolders as well
    Each config file should be saved to its own json, not just the main one!


    OLD:
        Get rid of Supervisor. It's creating backwards compatability issues
        Allow config to default to json during loading
        Allow causal controller to train on its own without began (lower gpu mem)

        Allow batch_size PlaceHolder for causal controller
            faster tvd calculation
            larger batch might help pretraining(limited at 16 right now)

        speedup crosstab
        allow only creation of causal controller graph (should also come with pt speedup)

        #This should be switched for node.label
            tf_parents=[self.z]+[node.label_logit for node in self.parents]



'''

'''
Actually feeding labels works well: doens't need to be rounded and factorized.

Feeding round(labels) instead of label_logits within cc made a huge difference

Feeding real parents was a bit of a disaster. Not sure why.
Try not doing that but with passing label instead of label_logit
        print 'WARNING: cc passes labels and rounds them before use'
        tf_parents=[self.z]+[tf.round(node.label) for node in self.parents]

'''


def main():
    print('tf: resetting default graph!')
    tf.reset_default_graph()#for repeated calls in ipython
    prepare_dirs_and_logger(config)

    config=get_config()
    cc_config,_=get_cc_config()
    dcgan_config,_=get_dcgan_config()
    began_config,_=get_began_config()

    attr_file= glob("{}/*.{}".format(config.data_path, 'txt'))[0]
    setattr(cc_config,'attr_file',attr_file)


    #Resolve model differences and batch_size
    if config.model_type:
        if config.model_type=='dcgan':
            self.config.batch_size=dcgan_config.batch_size
            config.Model=CausalGAN.CausalGAN
            model_config=dcgan_config
        if config.model_type=='began':
            self.config.batch_size=began_config.batch_size
            config.Model=CausalBEGAN.CausalBEGAN
            model_config=began_config
    else:#no image model
        model_config=None
        self.config.batch_size=cc_config.batch_size


    #Builds and loads specified models:
    trainer=Trainer(config,cc_config,model_config)


    if config.dry_run:
        #debug()
        return trainer

    #save combined model
    #Run pretrain and stuff
    #Run train

#    #if config.is_pretrain or config.is_train:
#    if not config.load_path:
#        print('saving config because load path not given')
#        save_config(config)
#
#    if config.is_pretrain:
#        trainer.pretrain()
#    if config.is_train:
#        trainer.train()
#    else:
#        if not config.load_path:
#            raise Exception("[!] You should specify `load_path` to load a pretrained model")
#
#        trainer.intervention()

    return trainer


if __name__ == "__main__":
    trainer=main()
