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
from model_config import get_config


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
        #FLAGS = get_config(FLAGS, FLAGS.model_ID)

        print 'ModelID is',FLAGS.model_ID
        FLAGS.checkpoint_dir = "./checkpoint/" + str(FLAGS.model_ID)
        if FLAGS.model_ID == 44:
          #FLAGS.is_train = True
          #FLAGS.graph = "big_causal_graph"
          FLAGS.graph = "complete_big_causal_graph"
          FLAGS.loss_function = 1
          FLAGS.pretrain_LabelerR = False
          FLAGS.pretrain_LabelerR_no_of_epochs = 3
          FLAGS.fakeLabels_distribution = "real_joint"
          FLAGS.gamma_k = -1.0
          FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
          FLAGS.gamma_l = -1.0 # made more extreme
          FLAGS.lambda_k = 0.05
          FLAGS.lambda_m = 0.05
          FLAGS.lambda_l = 0.001
          FLAGS.label_type = 'continuous'
        elif FLAGS.model_ID == 46:
          FLAGS.graph = "male_smiling_lipstick_complete"
          FLAGS.loss_function = 1
          FLAGS.pretrain_LabelerR = False
          FLAGS.pretrain_LabelerR_no_of_epochs = 3
          FLAGS.fakeLabels_distribution = "real_joint"
          FLAGS.gamma_k = -1.0
          FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
          FLAGS.gamma_l = -1.0 # made more extreme
          FLAGS.lambda_k = 0.05
          FLAGS.lambda_m = 0.05
          FLAGS.lambda_l = 0.001
          FLAGS.label_type = 'continuous'

        pp.pprint(FLAGS.__flags)

        print "WhatSUP:"+ str(FLAGS.checkpoint_dir)
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

        print FLAGS.checkpoint_dir
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
            is_train = FLAGS.is_train,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            graph=FLAGS.graph,
            loss_function=FLAGS.loss_function,
            pretrain_LabelerR = FLAGS.pretrain_LabelerR,
            pretrain_LabelerR_no_of_epochs = FLAGS.pretrain_LabelerR_no_of_epochs,
            fakeLabels_distribution = FLAGS.fakeLabels_distribution,
            gamma_k = FLAGS.gamma_k, gamma_m = FLAGS.gamma_m, gamma_l = FLAGS.gamma_l, lambda_k = FLAGS.lambda_k, lambda_m = FLAGS.lambda_m, lambda_l = FLAGS.lambda_l,
            model_ID = FLAGS.model_ID,
            label_type = FLAGS.label_type
        )

        #show_all_variables()
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir):
                print ("Warning: [!] Train a model first, then run test mode")
            if FLAGS.cc_checkpoint:
                dcgan.cc.load(dcgan.sess,FLAGS.cc_checkpoint)


        return dcgan




