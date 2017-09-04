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

#This flag doesn't function. Model is designed to take in CC.labels
data_arg.add_argument('--fakeLabels_distribution',type=str,choices=['real_joint','iid_uniform'],default='real_joint')


data_arg.add_argument('--label_type',type=str,choices=['discrete','continuous'],default='continuous')
data_arg.add_argument('--round_fake_labels',type=str2bool,default=True,
                    help='''whether to round the outputs of causal controller
                      before (possibly) adding noise to them or using them as
                      input to the image generator. I highly recommend as a
                      small improvement.''')

data_arg.add_argument('--type_input_to_generator',type=str,choices=['labels','logits'],
                      default='logits',help='''Whether to send labels or logits to the generator
                      to form images. Chris recommends labels''')

#Network
net_arg = add_argument_group('Network')

#TODO need help strings
net_arg.add_argument('--df_dim',type=int, default=64 )
net_arg.add_argument('--gf_dim',type=int, default=64,
                    help='''output dimensions [gf_dim,gf_dim] for generator''')
net_arg.add_argument('--c_dim',type=int, default=3,
                     help='''number of color channels. I wouldn't really change
                     this from 3''')

net_arg.add_argument('--z_dim',type=int,default=100,
                     help='''the number of dimensions for the noise input that
                     will be concatenated with labels and fed to the image
                     generator''')

net_arg.add_argument('--loss_function',type=int,default=1,
                     help='''which loss function to choose. See CausalGAN.py''')

net_arg.add_argument('--critic_hidden_size',type=int,default=10,
                    help='''number of neurons per fc layer in discriminator''')

net_arg.add_argument('--reconstr_loss',type=str2bool,default=False,
                     help='''whether to inclue g_loss_on_z in the generator
                     loss. This was True by default until recently which is where there are a lot of unneccsary networks''')


net_arg.add_argument('--stab_proj',type=str2bool,default=False,
                     help='''stabalizing projection method used for
                     discriminator. Stabalizing GAN Training with Multiple
                     Random Projections
                     https://arxiv.org/abs/1705.07831''')

net_arg.add_argument('--n_stab_proj',type=int,default=256,
                     help='''number of stabalizing projections. Need
                     stab_proj=True for this to have effect''')


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--num_iter',type=int,default=100000,
                       help='the number of training iterations to run the model for')
train_arg.add_argument('--learning_rate',type=float,default=0.0002,
                       help='Learning rate for adam [0.0002]')
train_arg.add_argument('--beta1',type=float,default=0.5,
                       help='Momentum term of adam [0.5]')

train_arg.add_argument('--off_label_losses',type=str2bool,default=False)

#TODO unclear on default for these two arguments
#Not yet setup. Use False
train_arg.add_argument('--pretrain_LabelerR',type=str2bool,default=False)

#counters over epochs preferred
#train_arg.add_argument('--pretrain_LabelerR_no_of_epochs',type=int,default=5)
train_arg.add_argument('--pretrain_LabelerR_no_of_iters',type=int,default=15000)


#TODO: add help strings describing params
train_arg.add_argument('--lambda_m',type=float,default=0.05,)#0.05
train_arg.add_argument('--lambda_k',type=float,default=0.05,)#0.05
train_arg.add_argument('--lambda_l',type=float,default=0.001,)#0.005
train_arg.add_argument('--gamma_m',type=float,default=-1.0,)# NOT USED!
train_arg.add_argument('--gamma_k',type=float,default=-1.0,#0.8#FLAGS.gamma_k not used
                       help='''default initial value''')
train_arg.add_argument('--gamma_l',type=float,default=-1.0,
                      )

train_arg.add_argument('--tau',type=float,default=3000,
                       help='''time constant. Every tau calls of k_t_update will
                       reduce k_t by a factor of 1/e.''')


#old config file differed from implementation:
#    FLAGS.gamma_k = -1.0
#    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
#    FLAGS.gamma_l = -1.0 # made more extreme
#    FLAGS.lambda_k = 0.05
#    FLAGS.lambda_m = 0.05
#    FLAGS.lambda_l = 0.001


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_train',type=str2bool,default=False,
                      help='''whether to enter the image training loop''')
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--log_step', type=int, default=100,
                     help='''how often to log stuff. Sample images are created
                     every 10*log_step''')


##REFERENCE
#  elif model_ID == 44:
#    FLAGS.is_train = True
#    #FLAGS.graph = "big_causal_graph"
#    FLAGS.graph = "complete_big_causal_graph"
#    FLAGS.loss_function = 1
#    FLAGS.pretrain_LabelerR = False
#    FLAGS.pretrain_LabelerR_no_of_epochs = 3
#    FLAGS.fakeLabels_distribution = "real_joint"
#    FLAGS.gamma_k = -1.0
#    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
#    FLAGS.gamma_l = -1.0 # made more extreme
#    FLAGS.lambda_k = 0.05
#    FLAGS.lambda_m = 0.05
#    FLAGS.lambda_l = 0.001
#    FLAGS.label_type = 'continuous'
#    return FLAGS



def get_config():
    config, unparsed = parser.parse_known_args()

    print('Loaded ./causal_dcgan/config.py')
    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

