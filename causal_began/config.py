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
net_arg.add_argument('--c_dim',type=int, default=3,
                     help='''number of color channels. I wouldn't really change
                     this from 3''')
net_arg.add_argument('--conv_hidden_num', type=int, default=128,
                     choices=[64, 128],help='n in the paper')
net_arg.add_argument('--separate_labeler', type=str2bool, default=True)
net_arg.add_argument('--z_dim', type=int, default=64, choices=[64, 128],
                    help='''dimension of the noise input to the generator along
                    with the labels''')
net_arg.add_argument('--z_num', type=int, default=64,
                    help='''dimension of the hidden space of the autoencoder''')


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='celebA')
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=16)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--d_lr', type=float, default=0.00008)
train_arg.add_argument('--g_lr', type=float, default=0.00008)
train_arg.add_argument('--label_loss',type=str,default='squarediff',choices=['xe','absdiff','squarediff'],
                      help='''what comparison should be made between the
                       labeler output and the actual labels''')
train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
train_arg.add_argument('--max_step', type=int, default=50000)
train_arg.add_argument('--num_iter',type=int,default=250000,
                       help='the number of training iterations to run the model for')
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--round_fake_labels',type=str2bool,default=True,
                       help='''Whether the label outputs of the causal
                       controller should be rounded first before calculating
                       the loss of generator or d-labeler''')
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--num_gpu', type=int, default=1,
                      help='specify 0 for cpu. If k specified, will default to\
                      first k of n gpus detected. If use_gpu=True but num_gpu not\
                      specified will default to 1')

margin_arg = add_argument_group('Margin')
margin_arg.add_argument('--gamma', type=float, default=0.5)
margin_arg.add_argument('--gamma_label', type=float, default=0.5)
margin_arg.add_argument('--lambda_k', type=float, default=0.001)
margin_arg.add_argument('--lambda_l', type=float, default=0.00008,
                       help='''As mentioned in the paper this is lower because
                       this margin can be responded to more quickly than the
                        other margins. Im not sure if it definitely needs to be lower''')
margin_arg.add_argument('--lambda_z', type=float, default=0.01)
margin_arg.add_argument('--no_third_margin', type=str2bool, default=False,
                       help='''Use True for appendix figure in paper. This is
                        used to neglect the third margin (c3,b3)''')
margin_arg.add_argument('--zeta', type=float, default=0.5,
                       help='''This is gamma_3 in the paper''')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_train',type=str2bool,default=False,
                      help='''whether to enter the image training loop''')
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
misc_arg.add_argument('--log_step', type=int, default=100,
                     help='''how often to log stuff. Sample images are created
                     every 10*log_step''')
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')



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


    print('Loaded ./causal_began/config.py')

    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

