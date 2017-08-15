#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    #return (v is True) or (v.lower() in ('true', '1'))
    return v is True or v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

#Pretrain network
pretrain_arg=add_argument_group('Pretrain')
pretrain_arg.add_argument('--pt_load_path', type=str, default='')
pretrain_arg.add_argument('--is_pretrain',type=str2bool,default=True,
                         help='to do pretraining')
pretrain_arg.add_argument('--only_pretrain', action='store_true',
                         help='simply complete pretrain and exit')
pretrain_arg.add_argument('--pretrain_type',type=str,default='wasserstein',choices=['wasserstein','gan'])
pretrain_arg.add_argument('--pt_cc_lr',type=float,default=0.00008,#
                          help='learning rate for causal controller')
pretrain_arg.add_argument('--pt_dcc_lr',type=float,default=0.00008,#
                          help='learning rate for causal controller')
pretrain_arg.add_argument('--lambda_W',type=float,default=0.1,#
                          help='penalty for gradient of W critic')
pretrain_arg.add_argument('--n_critic',type=int,default=25,#5 for speed
                          help='number of critic iterations between gen update')
pretrain_arg.add_argument('--critic_layers',type=int,default=6,#4 usual.8 might help
                          help='number of layers in the Wasserstein discriminator')
pretrain_arg.add_argument('--critic_hidden_size',type=int,default=15,#10,15
                         help='hidden_size for critic of discriminator')
pretrain_arg.add_argument('--min_tvd',type=float,default=0.02,
                          help='if tvd<min_tvd then stop pretrain')
pretrain_arg.add_argument('--min_pretrain_iter',type=int,default=5000,
                          help='''pretrain for at least this long before
                          stopping early due to tvd convergence. This is to
                          avoid being able to get a low tvd without labels
                          being clustered near integers''')
pretrain_arg.add_argument('--pretrain_iter',type=int,default=10000,
                          help='if iter>pretrain_iter then stop pretrain')
#pretrain_arg.add_argument('--pretrain_labeler',type=str2bool,default=False,
#                          help='''whether to train the labeler on real images
#                          during pretraining''')
pretrain_arg.add_argument('--pt_factorized',type=str2bool,default=True,
                          help='''whether the discriminator should be
                          factorized according to the structure of the graph
                          to speed convergence''')
pretrain_arg.add_argument('--pt_round_node_labels',type=str2bool,default=True,
                          help='''whether the labels internal in the causal
                          controller should be rounded before calcaulting the
                          labels for the child nodes
                          Should probably be False when pt_factorized is False''')

#pretrain_arg.add_argument('--pt_penalize_each_grad',type=str2bool,default=True,
#                          help='''whether to enforce that the gradient penalty
#                          for each component is close to 1, rather than
#                          enforcing that their average is close to 1''')

#Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_scale_size', type=int, default=64,
                     help='input image will be resized with the given value as width and height')
#net_arg.add_argument('--graph',type=str,default='big_causal_graph')
net_arg.add_argument('--conv_hidden_num', type=int, default=128,
                     choices=[64, 128],help='n in the paper')
net_arg.add_argument('--separate_labeler', type=str2bool, default=True)
net_arg.add_argument('--z_num', type=int, default=64, choices=[64, 128])
net_arg.add_argument('--cc_n_layers',type=int, default=6,
                     help='''this is the number of neural network fc layers
                     between the causes of a neuron and the neuron itsef.''')
net_arg.add_argument('--cc_n_hidden',type=int, default=10,
                     help='''number of neurons per layer in causal controller''')

# Data
data_arg = add_argument_group('Data')
#data_arg.add_argument('--causal_model', type=str, default='male.young.smiling')
data_arg.add_argument('--causal_model', type=str)
data_arg.add_argument('--dataset', type=str, default='celebA')
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
#data_arg.add_argument('--num_worker', type=int, default=4)
data_arg.add_argument('--num_worker', type=int, default=24,
                     help='number of threads to use for loading and preprocessing data')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=False)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=500000)
train_arg.add_argument('--noisy_labels', type=str2bool, default=False)
train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
train_arg.add_argument('--d_lr', type=float, default=0.00008)
train_arg.add_argument('--g_lr', type=float, default=0.00008)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
#train_arg.add_argument('--gamma_label', type=float, default=1.0)
train_arg.add_argument('--gamma_label', type=float, default=0.5)
train_arg.add_argument('--zeta', type=float, default=0.5)
train_arg.add_argument('--lambda_k', type=float, default=0.001)
train_arg.add_argument('--lambda_l', type=float, default=0.00008)
train_arg.add_argument('--lambda_z', type=float, default=0.01)
train_arg.add_argument('--no_third_margin', type=str2bool, default=False)
train_arg.add_argument('--indep_causal', type=str2bool, default=False)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--num_gpu', type=int, default=1,
                      help='specify 0 for cpu. If k specified, will default to\
                      first k of n detected. If use_gpu=True but num_gpu not\
                      specified will default to 1')

train_arg.add_argument('--label_loss',type=str,default='squarediff',choices=['xe','absdiff','squarediff'])
train_arg.add_argument('--round_fake_labels',type=str2bool,default=True,
                       help='''Whether the label outputs of the causal
                       controller should be rounded first before calculating
                       the loss of generator or d-labeler''')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--build_all', type=str2bool, default=False,
                     help='''normally specifying is_pretrain=False will cause
                     the pretraining components not to be built and likewise
                      with is_train=False only the pretrain compoenent will
                      (possibly) be built. This is here as a debug helper to
                      enable building out the whole model without doing any
                      training''')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--dry_run', action='store_true')
#misc_arg.add_argument('--dry_run', type=str2bool, default='False')
misc_arg.add_argument('--is_crop', type=str2bool, default='True')
misc_arg.add_argument('--resize_method',type=str,default='AREA',choices=['AREA','BILINEAR','BICUBIC','NEAREST_NEIGHBOR'],
                     help='''methods to resize image to 64x64. AREA seems to work
                     best, possibly some scipy methods could work better''')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
#misc_arg.add_argument('--sample_per_image', type=int, default=64,
#                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)

#Doesn't do anything atm
#misc_arg.add_argument('--visualize', action='store_true')


def gpu_logic(config):

    #consistency between use_gpu and num_gpu
    if config.num_gpu>0:
        config.use_gpu=True
    else:
        config.use_gpu=False
#        if config.use_gpu and config.num_gpu==0:
#            config.num_gpu=1
    return config


def get_config():
    config, unparsed = parser.parse_known_args()
    config=gpu_logic(config)

    #this has to respect gpu/cpu
    #data_format = 'NCHW'
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)

    if config.only_pretrain and config.is_train:
        print('Warning.. is_train=True conflicts with only_pretrain'),
        print('..setting is_train=False')
        setattr(config,'is_train',False)

    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

