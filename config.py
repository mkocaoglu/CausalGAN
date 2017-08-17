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

# Data
data_arg = add_argument_group('Data')
#data_arg.add_argument('--batch_size', type=int, default=16)#default set elsewhere
data_arg.add_argument('--causal_model', type=str,
                     help='''Matches the argument with a key in ./causal_graph.py and sets the graph attribute of cc_config to be a list of lists defining the causal graph''')
data_arg.add_argument('--data_dir', type=str, default='data')
data_arg.add_argument('--dataset', type=str, default='celebA')
data_arg.add_argument('--do_shuffle', type=str2bool, default=True)#never used
data_arg.add_argument('--input_scale_size', type=int, default=64,
                     help='input image will be resized with the given value as width and height')
data_arg.add_argument('--is_crop', type=str2bool, default='True')
data_arg.add_argument('--grayscale', type=str2bool, default=False)#never used
data_arg.add_argument('--split', type=str, default='train')#never used
data_arg.add_argument('--num_worker', type=int, default=24,
                     help='number of threads to use for loading and preprocessing data')
data_arg.add_argument('--resize_method',type=str,default='AREA',choices=['AREA','BILINEAR','BICUBIC','NEAREST_NEIGHBOR'],
                     help='''methods to resize image to 64x64. AREA seems to work
                     best, possibly some scipy methods could work better. It
                     wasn't clear to me why the results should be so different''')


# Training / test parameters
train_arg = add_argument_group('Training')


train_arg.add_argument('--build_train', type=str2bool, default=False,
                      help='''You may want to build all the components for
                       training, without doing any training right away. This is
                      for that. This arg is effectively True when is_train=True''')
train_arg.add_argument('--build_pretrain', type=str2bool, default=False,
                      help='''You may want to build all the components for
                       training, without doing any training right away. This is
                      for that. This arg is effectively True when is_pretrain=True''')


train_arg.add_argument('--model_type',type=str,default='',choices=['dcgan','began'],
                      help='''Which model to use. If the argument is not
                       passed, only causal_controller is built. This overrides
                      is_train=True, since no image model to train''')
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--num_gpu', type=int, default=1,
                      help='specify 0 for cpu. If k specified, will default to\
                      first k of n detected. If use_gpu=True but num_gpu not\
                      specified will default to 1')

# Misc
misc_arg = add_argument_group('Misc')
#misc_arg.add_argument('--build_all', type=str2bool, default=False,
#                     help='''normally specifying is_pretrain=False will cause
#                     the pretraining components not to be built and likewise
#                      with is_train=False only the pretrain compoenent will
#                      (possibly) be built. This is here as a debug helper to
#                      enable building out the whole model without doing any
#                      training''')

misc_arg.add_argument('--descrip', type=str, default='',help='''
                      Only use this when creating a new model. New model folder names
                      are generated automatically by using the time-date. Then
                      you cant rename them while the model is running. If
                      provided, this is a short string that appends to the end
                      of a model folder name to help keep track of what the
                      contents of that folder were without getting into the
                      content of that folder. No weird characters''')

misc_arg.add_argument('--dry_run', action='store_true',help='''Build and load
                      the model and all the specified components, but don't actually do
                      any pretraining/training etc. This overrides
                      --is_pretrain, --is_train. This is mostly used for just
                      bringing the model into the workspace if you say wanted
                      to manipulated it in ipython''')

misc_arg.add_argument('--load_path', type=str, default='',
                     help='''This is a "global" load path. You can simply pass
                     the model_dir of the whatever run, and all the variables
                      (dcgan/began and causal_controller both). If you want to
                      just load one component: for example, the pretrained part
                      of a previous model, use pt_load_path from the
                      causal_controller.config section''')

misc_arg.add_argument('--log_step', type=int, default=100,
                     help='''this is used for generic summaries that are common
                     to both models. Use model specific config files for
                     logging done within train_step''')
#misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs', help='''where to store model and model results. Do not put a leading "./" out front''')

#misc_arg.add_argument('--sample_per_image', type=int, default=64,
#                      help='# of sample per image during test sample generation')

#misc_arg.add_argument('--random_seed', type=int, default=123)

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
    config.num_devices=max(1,config.num_gpu)#that are used in backprop


    #Just for BEGAN:
    ##this has to respect gpu/cpu
    ##data_format = 'NCHW'
    #if config.use_gpu:
    #    data_format = 'NCHW'
    #else:
    #    data_format = 'NHWC'
    #setattr(config, 'data_format', data_format)

    print('Loaded ./config.py')

    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

