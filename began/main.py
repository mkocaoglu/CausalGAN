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
TODO:
    Get rid of Supervisor. It's creating backwards compatability issues
    Update config defaults
    Allow config to default to json during loading
    Make data symlinks so that we can keep models on other partitions
'''


'''
!!!!!Rounding cc outputs was a huge success!!!
Also I made some crucial adjustments to conditioning and intervening.
This calls for resetting began defaults.

    def break_pretrain(stats,counter):
        c1=counter>=self.config.min_pretrain_iter
        c2= (stats['tvd']<self.config.min_tvd)
        return (c1 and c2)

pretrain_arg.add_argument('--min_pretrain_iter',type=int,default=5000,
                          help=pretrain for at least this long. This is to
                          avoid being able to get a low tvd without labels
                          being clustered near integers)
pretrain_arg.add_argument('--pretrain_type',type=str,default='wasserstein',choices=['wasserstein','gan'])
pretrain_arg.add_argument('--pt_factorized',type=str2bool,default=True,
net_arg.add_argument('--cc_n_layers',type=int, default=6,
train_arg.add_argument('--label_loss',type=str,default='squarediff',choices=['xe','absdiff','squarediff'])
train_arg.add_argument('--round_fake_labels',type=str2bool,default=True,
train_arg.add_argument('--noisy_labels', type=str2bool, default=False)


pretrain_arg.add_argument('--pretrain_type',type=str,default='gan',choices=['wasserstein','gan'])
pretrain_arg.add_argument('--pt_factorized',type=str2bool,default=False,
net_arg.add_argument('--cc_n_layers',type=int, default=3,
train_arg.add_argument('--label_loss',type=str,default='absdiff',choices=['xe','absdiff','squarediff'])
train_arg.add_argument('--round_fake_labels',type=str2bool,default=False,
train_arg.add_argument('--noisy_labels', type=str2bool, default=True)




Trying to round labels before passed to generator as well..
        if self.config.round_fake_labels:
            self.z= tf.concat( [tf.round(self.fake_labels), self.z_gen],axis=-1,name='z')
        else:
            self.z= tf.concat( [self.fake_labels, self.z_gen],axis=-1,name='z')



Tried to load some previous models. Running into terrible errors
Basically this line:

        self.saver = tf.train.Saver(var_list=self.var)

makes it so that only trainable variables are saved by Saver. Then when
supervisor tries to restore model, there are unintialized Adam and step
parameters




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
