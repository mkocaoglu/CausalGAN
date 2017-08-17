import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
import os

from IPython.core import debugger
debug = debugger.Pdb().set_trace


'''main code for synthetic experiments

'''


def get_trainer(config):
    print 'tf: resetting default graph!'
    tf.reset_default_graph()

    #tf.set_random_seed(config.random_seed)
    #np.random.seed(22)

    print 'Using data_type ',config.data_type
    trainer=Trainer(config,config.data_type)
    print 'built trainer successfully'

    tf.logging.set_verbosity(tf.logging.ERROR)

    return trainer


def main(trainer,config):

    if config.is_train:
        trainer.train()



def get_model(config=None):
    if not None:
        config, unparsed = get_config()
    return get_trainer(config)

if __name__ == "__main__":
    config, unparsed = get_config()
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    trainer=get_trainer(config)
    main(trainer,config)


