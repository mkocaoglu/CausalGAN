import tensorflow as tf
import os
import scipy.misc
import numpy as np

from model import DCGAN
from Causal_model import DCGAN as CausalGAN

import pprint
pp = pprint.PrettyPrinter()

from utils import visualize, to_json#, show_all_variables
from main import FLAGS



from began.main import get_model as get_began_model



def get_model(name):
    '''
    simple handle function. pass in model name and model key word arguments
    as a dictionary


    This function exists so that in visualize, models can be called with
    commandline args if needed. Alternatively, if working within python,
    an existing model can just be passed as argument and this is not needed
    '''
    print 'Warning: resetting tensorflow for model load'
    tf.reset_default_graph()

    if not name in ['began','dcgan']:
        raise ValueError('Pass either began or dcgan to specify which model')


    if name == 'began':
        os.chdir('./began')
        try:
            model=get_began_model()
        except Exception as e:
            print 'exception caught in model_loader'
            raise
        #finally:
        #    os.chdir('..')

        return model


    elif name == 'dcgan':
        pp.pprint(FLAGS.__flags)

        if FLAGS.input_width is None:
            FLAGS.input_width = FLAGS.input_height
        if FLAGS.output_width is None:
            FLAGS.output_width = FLAGS.output_height

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.sample_dir):
            os.makedirs(FLAGS.sample_dir)

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth=True


        sess=tf.Session(config=run_config)
        dcgan = CausalGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            c_dim=FLAGS.c_dim,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            is_crop=FLAGS.is_crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            graph=FLAGS.graph)

        #show_all_variables()
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir):
                print ("Warning: [!] Train a model first, then run test mode")

        return dcgan




