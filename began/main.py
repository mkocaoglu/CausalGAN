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


#Maybe should encorporate this kind of code to reduce gpu memory footprint:
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True


---

Bugfix: by intervening on labels: I neglected the causal graph, which passes
logits from node to node

------------
Previous model did not have balance_l functional but still pretty much worked.

d_loss_real label > d_loss_fake_label but not by much. Increase gamma_l

train_arg.add_argument('--gamma_label', type=float, default=1.0)
#train_arg.add_argument('--gamma_label', type=float, default=0.5)


also increase pretrain iter to 5000

            #if step < 3000:#PRETRAIN CC
            if step < 5000:#PRETRAIN CC


#Change real noise method again:
        #simple [0.2,1/2] or [1/2,0.8]
        noise=tf.random_uniform([len(p)],0.2,0.5)
        neg_noise=noise
        pos_noise=1.0-noise
        label= (1-label)*neg_noise + label*pos_noise
        ##Original Murat Noise model
        #noise=tf.random_uniform([len(p)],-.25,.25,)
        #P = label+p-2*label*p #p or (1-p):for label=0,1
        #L = 1-2*label# +1 or -1   :for label=0,1
        #label=0.5 + L*.25*P + P*noise

#Also added ability to model labels separately with separate labeler

        net_arg.add_argument('--separate_labeler', type=str2bool, default=False)

        if not self.separate_labeler:
            self.D_fake_labels_logits=tf.slice(self.D_encode_G,[0,0],[-1,n_labels])
            self.D_real_labels_logits=tf.slice(self.D_encode_x,[0,0],[-1,n_labels])
        else:

            label_logits,self.DL_var=Discriminator_labeler(
                    tf.concat([G, x], 0), len(self.cc.nodes), self.repeat_num,
                    self.conv_hidden_num, self.data_format)
            self.D_fake_labels_logits,self.D_real_labels_logits=tf.split(label_logits,2)
            self.D_var += self.DL_var

#changed default graph
        #data_arg.add_argument('--causal_model', type=str, default='male.young.smiling')
        data_arg.add_argument('--causal_model', type=str, default='big_causal_graph')

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
        batch_size = config.sample_per_image
        do_shuffle = False

    data_loader = get_loader(config,
            data_path,config.batch_size,config.input_scale_size,
            config.data_format,config.split,
            do_shuffle,config.num_worker,config.is_crop)

    config.graph=get_causal_graph(config.causal_model)

    print 'Config:'
    print config

    trainer = Trainer(config, data_loader)
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
