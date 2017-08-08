from __future__ import print_function
import argparse

def str2bool(v):
    return v is True or v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=64,
                     help='''default batch_size when using this model and not
                      specifying the batch_size elsewhere''')



data_arg.add_argument('--label_specific_noise',type=str2bool,default=False,
                      help='whether to add noise dependent on the data mean')

data_arg.add_argument('--fakeLabels_distribution',type=str,choices=['real_joint','iid_uniform'],default='real_joint')
data_arg.add_argument('--label_type',type=str,choices=['discrete','continuous'],default='continuous')
data_arg.add_argument('--round_fake_labels',type=str2bool,default=True,
                    help='''whether to round the outputs of causal controller
                      before (possibly) adding noise to them or using them as
                      input to the image generator. I highly recommend as a
                      small improvement.''')


#Network
net_arg = add_argument_group('Network')

#TODO need help strings
net_arg.add_argument('--df_dim',type=int,    )
net_arg.add_argument('--gf_dim',type=int,    )

net_arg.add_argument('--z_dim',type=int,default=100,
                     help='''the number of dimensions for the noise input that
                     will be concatenated with labels and fed to the image
                     generator''')

net_arg.add_argument('--loss_function',type=int,default=100,
                     help='''which loss function to choose. See CausalGAN.py''')

net_arg.add_argument('--critic_hidden_size',type=int,default=10,
                    help='''number of neurons per fc layer in discriminator''')


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--num_iter',type=int,default=100000,
                       help='the number of training iterations to run the model for')
train_arg.add_argument('--learning_rate',type=float,default=0.0002,
                       help='Learning rate for adam [0.0002]')
train_arg.add_argument('--beta1',type=float,default=0.5,
                       help='Momentum term of adam [0.5]')


#TODO unclear on default for these two arguments
train_arg.add_argument('--pretrain_labelerR',type=str2bool,default=False)
train_arg.add_argument('--pretrain_LabelerR_no_of_epochs',type=int,default=5)


#TODO: add help strings describing params
train_arg.add_argument('--lambda_m',type=float,default=0.05,)
train_arg.add_argument('--lambda_k',type=float,default=0.05,)
train_arg.add_argument('--lambda_l',type=float,default=0.005,)
train_arg.add_argument('--gamma_m',type=float,default=4.0,)
train_arg.add_argument('--gamma_k',type=float,default=0.8,
                       help='''default initial value''')
train_arg.add_argument('--gamma_l',type=float,default=0.5,
                      )


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')


def get_config():
    config, unparsed = parser.parse_known_args()

    print('Loaded ./causal_dcgan/config.py')
    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

