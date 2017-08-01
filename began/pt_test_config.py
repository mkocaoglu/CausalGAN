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








def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

