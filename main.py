import numpy as np
import tensorflow as tf

from trainer import Trainer
from causal_graph import get_causal_graph
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

#Generic configuration arguments
from config import get_config
#Submodel specific configurations
from causal_controller.config import get_config as get_cc_config()
from causal_dcgan.config import get_config as get_dcgan_config()
from causal_began.config import get_config as get_began_config()


from IPython.core import debugger
debug = debugger.Pdb().set_trace


'''
    Sometimes I leave notes here as a way to see what the motivation was for the
    previous models and to highlight what code changes were made. File is copied
    into log directory.
'''


'''
TODO:
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


    save .py files in subfolders as well

'''

'''
Actually feeding labels works well: doens't need to be rounded and factorized.

Feeding round(labels) instead of label_logits within cc made a huge difference

Feeding real parents was a bit of a disaster. Not sure why.
Try not doing that but with passing label instead of label_logit
        print 'WARNING: cc passes labels and rounds them before use'
        tf_parents=[self.z]+[tf.round(node.label) for node in self.parents]

'''


def get_trainer(config):

    #rng = np.random.RandomState(config.random_seed)
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
    print 'tf: resetting default graph!'
    tf.reset_default_graph()#Repeated calls in ipython
    prepare_dirs_and_logger(config)

    trainer=Trainer(config,cc_config,dcgan_config,began_config)



    config=get_config()
    cc_config,_=get_cc_config()





    if config.dry_run:
        #debug()
        return

    #if config.is_pretrain or config.is_train:
    if not config.load_path:
        print('saving config because load path not given')
        save_config(config)

    if config.is_pretrain:
        trainer.pretrain()
    if config.is_train:
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")

        trainer.intervention()

def get_model(config=None):
    if not None:
        config, unparsed = get_config()
    return get_trainer(config)

if __name__ == "__main__":
    config, unparsed = get_config()
    trainer=get_trainer(config)
    main(trainer,config)
    ##debug mode: below is main() code
