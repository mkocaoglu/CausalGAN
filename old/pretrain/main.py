import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from causal_graph import get_causal_graph
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

from IPython.core import debugger
debug = debugger.Pdb().set_trace


'''
just trying to get pretraining off the ground

'''

def get_trainer(config):
    print 'tf: resetting default graph!'
    tf.reset_default_graph()

    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    #tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        #setattr(config, 'batch_size', 64)
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        #batch_size = config.sample_per_image
        batch_size = config.batch_size
        do_shuffle = False

    data_loader, label_stats= get_loader(config,
            data_path,config.batch_size,config.input_scale_size,
            config.data_format,config.split,
            do_shuffle,config.num_worker,config.is_crop)

    config.graph=get_causal_graph(config.causal_model)

    print 'Config:'
    print config

    trainer = Trainer(config, data_loader, label_stats)
    return trainer


def main(trainer,config):
    if config.dry_run:
        #debug()
        return


    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")

        trainer.intervention()
        #trainer.test()

def get_model(config=None):
    if not None:
        config, unparsed = get_config()
    return get_trainer(config)

if __name__ == "__main__":
    config, unparsed = get_config()
    trainer=get_trainer(config)
    main(trainer,config)
    ##debug mode: below is main() code

#    prepare_dirs_and_logger(config)
#
#    rng = np.random.RandomState(config.random_seed)
#    tf.set_random_seed(config.random_seed)
#
#    if config.is_train:
#        data_path = config.data_path
#        batch_size = config.batch_size
#        do_shuffle = True
#    else:
#        setattr(config, 'batch_size', 64)
#        if config.test_data_path is None:
#            data_path = config.data_path
#        else:
#            data_path = config.test_data_path
#        batch_size = config.sample_per_image
#        do_shuffle = False
#
#    data_loader = get_loader(config,
#            data_path,config.batch_size,config.input_scale_size,
#            config.data_format,config.split,
#            do_shuffle,config.num_worker,config.is_crop)
#
#    config.graph=get_causal_graph()
#
#    trainer = Trainer(config, data_loader)
#
#    if config.dry_run:
#        return
#
#    if config.is_train:
#        save_config(config)
#        trainer.train()
#    else:
#        if not config.load_path:
#            raise Exception("[!] You should specify `load_path` to load a pretrained model")
#        trainer.test()
#
