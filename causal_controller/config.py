'''

These are the command line parameters that pertain exlusively to the
CausalController.

'''

from __future__ import print_function
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
pretrain_arg.add_argument('--is_pretrain',type=str2bool,default=False,
                         help='to do pretraining')
#pretrain_arg.add_argument('--only_pretrain', action='store_true',
#                         help='simply complete pretrain and exit')

#Used to be an option, but now is solved
#pretrain_arg.add_argument('--pretrain_type',type=str,default='wasserstein',choices=['wasserstein','gan'])

pretrain_arg.add_argument('--pt_cc_lr',type=float,default=0.00008,#
                          help='learning rate for causal controller')
pretrain_arg.add_argument('--pt_dcc_lr',type=float,default=0.00008,#
                          help='learning rate for causal controller')
pretrain_arg.add_argument('--lambda_W',type=float,default=0.1,#
                          help='penalty for gradient of W critic')
pretrain_arg.add_argument('--n_critic',type=int,default=20,#5 for speed
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

pretrain_arg.add_argument('--pt_factorized',type=str2bool,default=False,
                          help='''Interesting approach that seemed to stabalize
                          training, but is not needed in this application.
                          It turned out that we could get very good training without
                          this complication, so we did not include in the paper.
                          I've left it commented out here in the code.

                          Whether the discriminator should be
                          factorized according to the structure of the graph
                          to speed/stabalize convergence.
                          
                          This creates a separate discriminator for each node
                          that only looks at each causal nodes value and its
                          parents''')

#Network
net_arg = add_argument_group('Network')

net_arg.add_argument('--cc_n_layers',type=int, default=6,
                     help='''This is the number of neural network fc layers
                     between the causes of a node and the node itsef.''')
net_arg.add_argument('--cc_n_hidden',type=int, default=10,
                     help='''number of neurons per layer in causal controller.
                     Also functions as the dimensionality of the uniform noise
                     input to the controller''')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--causal_model', type=str)
data_arg.add_argument('--dataset', type=str, default='celebA')

data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--num_worker', type=int, default=24,
     help='number of threads to use for loading and preprocessing data')

# Training / test parameters
train_arg = add_argument_group('Training')




# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')


def get_config():
    config, unparsed = parser.parse_known_args()
    print('Loaded ./causal_controller/config.py')
    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

