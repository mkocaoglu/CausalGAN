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
Added wasserstein as an option during pretraining:
    #Pretrain network
    pretrain_arg=add_argument_group('Pretrain')
    pretrain_arg.add_argument('pretrain_type',type=str,default='gan',choices=['wasserstein','gan'])
    pretrain_arg.add_argument('lambda_W',type=float,default=0.1,
                              help='penalty for gradient of W critic')
    pretrain_arg.add_argument('n_critic',type=int,default=25,
                              help='number of critic iterations between gen update')
    pretrain_arg.add_argument('critic_hidden_size',type=int,default=10,
                             help='hidden_size for critic of discriminator')
    pretrain_arg.add_argument('min_tvd',type=float,
                              help='if tvd<min_tvd then stop pretrain')
    pretrain_arg.add_argument('pretrain_iter',type=int,default=10000,
                              help='if iter>pretrain_iter then stop pretrain')


Pass attr through config later on
    setattr(config,'attr',attributes[label_names])

def pretrain is written

Some uncertianty in my mind how cc_step and self.step should play together
self.step should probably be a property which is a sum

Also added code to do intervention2d and condition2d during train

Also added tvd code

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
    elif config.is_train:
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
