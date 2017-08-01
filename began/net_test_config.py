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





def get_config():
    config, unparsed = parser.parse_known_args()
    return config,unparsed


if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

