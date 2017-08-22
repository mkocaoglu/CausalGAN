import argparse
from models import DataTypes
def str2bool(v):
    return v is True or v.lower() in ('true', '1')

dtypes=DataTypes.keys()


arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

#Pretrain network
data_arg=add_argument_group('Data')
gan_arg=add_argument_group('GAN')
misc_arg=add_argument_group('misc')
model_arg=add_argument_group('Model')

data_arg.add_argument('--data_type',type=str,choices=dtypes,
                      default='collider', help='''This is the graph structure
                      that generates the synthetic dataset through polynomials''')

gan_arg.add_argument('--gen_z_dim',type=int,default=10,
                     help='''dim of noise input for generator''')
gan_arg.add_argument('--gen_hidden_size',type=int,default=10,#3,
                     help='''hidden size used for layers of generator''')
gan_arg.add_argument('--disc_hidden_size',type=int,default=10,#6,
                     help='''hidden size used for layers of discriminator''')
gan_arg.add_argument('--lr_gen',type=float,default=0.0005,#0.005
                     help='''generator learning rate''')
gan_arg.add_argument('--lr_disc',type=float,default=0.0005,#0.0025
                     help='''discriminator learning rate''')

#broken
#misc_arg.add_argument('--save_pdfs',type=str2bool,default=False,
#                     help='''whether to save pdfs of scatterplots of x1x3 along
#                     with tensorboard summaries''')

misc_arg.add_argument('--model_dir',type=str,default='logs')
#misc_arg.add_argument('--np_random_seed', type=int, default=123)
#misc_arg.add_argument('--tf_random_seed', type=int, default=123)


model_arg.add_argument('--load_path',type=str,default='',
                       help='''Path to folder containing model to load. This
                       should be actual checkpoint to load. Example:
                       --load_path=./logs/0817_153755_collider/checkpoints/Model-50000''')
model_arg.add_argument('--is_train',type=str2bool,default=True,
                       help='''whether the model should train''')
model_arg.add_argument('--batch_size',type=int,default=64,
                      help='''batch_size for all generators and all
                       discriminators''')


def get_config():

    #setattr(config, 'data_dir', data_format)
    config, unparsed = parser.parse_known_args()
    return config, unparsed

