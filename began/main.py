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
Some notes on each version of the model:

smiling->MSO did not work so inc pretrain
            if step < 15000:#PRETRAIN CC


!!!BUG!!! Previous models all ran setup_tensor twice, which effectvely left them
independent in causal structure.

also the variable scopes were nesting due to lazy evaluation behind the scenes

    def __init__(self,name=None,n_hidden=10):
        self.name=name
        self.n_hidden=n_hidden#also is z_dim

        #Use tf.random_uniform instead of placeholder for noise
        n=self.batch_size*self.n_hidden
        #print 'CN n',n
        with tf.variable_scope(self.name):
            self.z = tf.random_uniform(
                    (self.batch_size, self.n_hidden), minval=-1.0, maxval=1.0)
    def setup_tensor(self):
        if self._label is not None:#already setup
            return
        tf_parents=[self.z]+[node.label_logit for node in self.parents]
        with tf.variable_scope(self.name):
            vec_parents=tf.concat(tf_parents,-1)
            h0=slim.fully_connected(vec_parents,self.n_hidden,activation_fn=tf.nn.tanh,scope='layer0')
            h1=slim.fully_connected(h0,self.n_hidden,activation_fn=tf.nn.tanh,scope='layer1')
            self._label_logit = slim.fully_connected(h1,1,activation_fn=None,scope='proj')
            self._label=tf.nn.sigmoid( self._label_logit )


Also has a HACK, made a symlink to figure_scripts so I could use that.

during pretraining:
    if step+1 %1000==0:
        crosstab(self)


'''

def get_trainer(config):
    print 'tf: resetting default graph!'
    tf.reset_default_graph()

    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

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
