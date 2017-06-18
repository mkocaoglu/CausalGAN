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
Less frequenty logs
            #if step % (self.log_step * 10) == 0:
            if step % (self.log_step * 50) == 0:


Using squared loss seemed to work for wasserstein but interestingly not for
mustache. One thing left to try is to use rounding on the labels.

    (from before)
        self.d_squarediff_real_label=tf.square(self.D_real_labels  - self.real_labels)
        self.d_squarediff_fake_label=tf.square(self.D_fake_labels  - self.fake_labels)
        self.g_squarediff_label=tf.square(self.fake_labels  -  self.D_fake_labels)

Also:
    (instead of 1000 logit for integer)
            'min_logit':-5.*ones
            'max_logit':5*ones


train_arg.add_argument('--label_loss',type=str,default='absdiff',choices=['xe','absdiff','squarediff'])
train_arg.add_argument('--round_fake_labels',type=str2bool,default=False,
                       help=Whether the label outputs of the causal
                       controller should be rounded first before calculating
                       the loss of generator or d-labeler



If anything, decreasing gamma_label to 0.5 made things worse. Even though
k_l < 0 during almost the whole training, still celebA_0531_035710 had a
Disc_labeler that didn't work on real images. This is strange given that is was
the only objective for those variables. Source of bug unknown.

Going to try with continuous labels, and allowing disc_labeler to train during
pretraining. see what happens

    pretrain_arg.add_argument('--pretrain_labeler',type=str2bool,default=False,
                              help=whether to train the labeler on real images
                              during pretraining)

    self.dl_optim=self.dlr_optimizer.minimize(trainer.d_loss_real_label)
    if self.config.pretrain_labeler:
        self.pretrain_op = tf.group(self.c_optim,self.dcc_optim,self.dl_optim)
    else:
        self.pretrain_op = tf.group(self.c_optim,self.dcc_optim)


-----

gamma_label was 1.0 instead of 0.5 and this caused d_labeler to never learn the
labels

#train_arg.add_argument('--gamma_label', type=float, default=1.0)
train_arg.add_argument('--gamma_label', type=float, default=0.5)

I think I advice gamma_label~1.0 for imperfect continuous pretraining, but 0.5 for
idealized training e.g.discrete. Also
TODO: compare the diameter between training labels and real labels. Ratio
should be normalization number for lambda.


Tensorflow's bilinear resizing seemed to be messing up "glasses"
scipy.misc.resize had a number of methods that performed better.
Using tensorflow "Area" seemed to do better, but may reduce sharpness
    resize_method=getattr(tf.image.ResizeMethod,config.resize_method)
    image=tf.image.resize_images(image,[scale_size,scale_size],
            method=resize_method)



tvd for male_mustache_lipstick reached 0.02 tvd in 4k iter
However, mustache label was not heavily centered around 0,1
Need to add percentile code to check that out

Also need to change variable name for g_step from 'step' to g_step
Also consider code that loads a model based on the code in that models directory


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
