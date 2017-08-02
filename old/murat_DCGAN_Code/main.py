import os
import scipy.misc
import numpy as np

from model import DCGAN
from Causal_model import DCGAN as CausalGAN
from model_config import get_config

from utils import pp, visualize, to_json#, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("model", "causal", "dcgan to use the original model")
flags.DEFINE_string("graph", "Provide a graph argument", "causal graph defined in causal_graph.py to use")
flags.DEFINE_integer("epoch", 30, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("label_specific_noise", False, "If True, noise added to real labels are adjusted to require uniform logits")
flags.DEFINE_integer("loss_function", 0, "Type of loss to be used")
flags.DEFINE_float("label_loss_hyperparameter",1,"scaling of label loss")
flags.DEFINE_integer("model_ID", None, "ID for configuration file to be used")
flags.DEFINE_boolean("pretrain_LabelerR", False, "Should it pretrain labelerR")
flags.DEFINE_integer("pretrain_LabelerR_no_of_epochs", None, "For how many epochs should LabelerR be pretrained")
flags.DEFINE_string("fakeLabels_distribution", None, "Type of distribution to be used for fake labels")
flags.DEFINE_float("gamma_k", None, "g_lossLabels ~= g_lossLabels_GLabeler/gamma_k")
flags.DEFINE_float("gamma_m", None, "g_lossGAN ~= gamma_m*self.g_lossLabels")
flags.DEFINE_float("gamma_l", None, "g_lossGAN ~= gamma_l*(self.d_loss_real + self.d_loss_fake")
flags.DEFINE_float("lambda_k", None, "increment for k_t")
flags.DEFINE_float("lambda_m", None, "increment for m_t")
flags.DEFINE_float("lambda_l", None, "increment for l_t")
flags.DEFINE_string("label_type", None, "continuous vs discrete labels")
flags.DEFINE_string("cc_checkpoint", '', "checkpoint for causal_controller")
FLAGS = flags.FLAGS


def main(_):

  FLAGS = get_config(flags.FLAGS, flags.FLAGS.model_ID)
  pp.pprint(flags.FLAGS.__flags)

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

  if FLAGS.model == 'causal':
      GAN=CausalGAN
  elif FLAGS.model == 'dcgan':
      GAN=DCGAN


  with tf.Session(config=run_config) as sess:
    dcgan = GAN(
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
        gamma_k = FLAGS.gamma_k, gamma_m = FLAGS.gamma_m, gamma_l =
        FLAGS.gamma_l, lambda_k = FLAGS.lambda_k, lambda_m = FLAGS.lambda_m,
        lambda_l = FLAGS.lambda_l, model_ID = FLAGS.model_ID,
        label_type = FLAGS.label_type,
        cc_checkpoint=FLAGS.cc_checkpoint)

    #show_all_variables()
    if FLAGS.is_train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    # OPTION = 0 # male
    # visualize(sess, dcgan, FLAGS, OPTION)
    OPTION = 11 # male
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  #main(0)
  tf.app.run()
