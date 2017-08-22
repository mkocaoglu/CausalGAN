from __future__ import print_function
import tensorflow as tf
import logging
import numpy as np
import pandas as pd
import shutil
import json
import sys
import os
from datetime import datetime
from tqdm import trange
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile,join

from utils import calc_tvd,summary_scatterplots,Timer,summary_losses,make_summary
from models import GeneratorTypes,DataTypes,Discriminator,sxe

class GAN(object):
    def __init__(self,config,gan_type,data,parent_dir):
        self.config=config
        self.gan_type=gan_type
        self.data=data
        self.Xd=data.X
        self.parent_dir=parent_dir
        self.prepare_model_dir()
        self.prepare_logger()

        with tf.variable_scope(gan_type):
            self.step=tf.Variable(0,'step')
            self.inc_step=tf.assign(self.step,self.step+1)
            self.build_model()
        self.build_summaries()#This can be either in var_scope(name) or out

    def build_model(self):
        Gen=GeneratorTypes[self.gan_type]
        config=self.config
        self.gen=Gen(config.batch_size,config.gen_hidden_size,config.gen_z_dim)

        with tf.variable_scope('Disc') as scope:
            self.D1 = Discriminator(self.data.X, config.disc_hidden_size)
            scope.reuse_variables()
            self.D2 = Discriminator(self.gen.X, config.disc_hidden_size)
            d_var = tf.contrib.framework.get_variables(scope)

        d_loss_real=tf.reduce_mean( sxe(self.D1,1) )
        d_loss_fake=tf.reduce_mean( sxe(self.D2,0) )
        self.loss_d =  d_loss_real  +  d_loss_fake
        self.loss_g = tf.reduce_mean( sxe(self.D2,1) )

        optimizer=tf.train.AdamOptimizer
        g_optimizer=optimizer(self.config.lr_gen)
        d_optimizer=optimizer(self.config.lr_disc)
        self.opt_d = d_optimizer.minimize(self.loss_d,var_list= d_var)
        self.opt_g = g_optimizer.minimize(self.loss_g,var_list= self.gen.tr_var,
                               global_step=self.gen.step)

        with tf.control_dependencies([self.inc_step]):
            self.train_op=tf.group(self.opt_d,self.opt_g)

    def build_summaries(self):
        d_summ=tf.summary.scalar(self.data.name+'_dloss',self.loss_d)
        g_summ=tf.summary.scalar(self.data.name+'_gloss',self.loss_g)
        self.summaries=[d_summ,g_summ]
        self.summary_op=tf.summary.merge(self.summaries)
        self.tf_scatter=tf.placeholder(tf.uint8,[3,480,640,3])
        scatter_name='scatter_D'+self.data.name+'_G'+self.gen.name
        self.g_scatter_summary=tf.summary.image(scatter_name,self.tf_scatter,max_outputs=3)
        self.summary_writer=tf.summary.FileWriter(self.model_dir)

    def record_losses(self,sess):
        step, sum_loss_g, sum_loss_d = summary_losses(sess,self)
        self.summary_writer.add_summary(sum_loss_g,step)
        self.summary_writer.add_summary(sum_loss_d,step)
        self.summary_writer.flush()

    def record_tvd(self,sess):
        step,tvd,mvd = calc_tvd(sess,self.gen,self.data)
        self.log_tvd(step,tvd,mvd)
        summ_tvd=make_summary(self.data.name+'_tvd',tvd)
        summ_mvd=make_summary(self.data.name+'_mvd',mvd)
        self.summary_writer.add_summary(summ_tvd,step)
        self.summary_writer.add_summary(summ_mvd,step)
        self.summary_writer.flush()
    def record_scatter(self,sess):
        Xg=sess.run(self.gen.X,{self.gen.N:5000})
        X1,X2,X3=np.split(Xg,3,axis=1)
        x1x2,x1x3,x2x3 = summary_scatterplots(X1,X2,X3)
        step,Pg_summ=sess.run([self.step,self.g_scatter_summary],{self.tf_scatter:np.concatenate([x1x2,x1x3,x2x3])})
        self.summary_writer.add_summary(Pg_summ,step)
        self.summary_writer.flush()

#        if self.config.save_pdfs:
#            self.save_np_scatter(step,X1,X3)

#Maybe it's the supervisor creating the segfault??
#Try just one model at a time

#   #will cause segfault ;)
#    def save_np_scatter(self,step,x,y,save_dir=None,ext='.pdf'):
#        '''
#        This is a convenience that just saves the image as a pdf in addition to putting it on
#        tensorboard. only does x1x3 because that's what I needed at the moment
#
#        sorry I wrote this really quickly
#        TODO: make less bad.
#        '''
#        plt.scatter(x,y)
#        plt.title('X1X3')
#        plt.xlabel('X1')
#        plt.ylabel('X3')
#        plt.xlim([0,1])
#        plt.ylim([0,1])
#
#        scatter_dir=os.path.join(self.model_dir,'scatter')
#
#        save_dir=save_dir or scatter_dir
#        if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#
#        save_name=os.path.join(save_dir,'{}_scatter_x1x3_{}_{}'+ext)
#        save_path=save_name.format(step,self.config.data_type,self.gan_type)
#
#        plt.savefig(save_path)



    def prepare_model_dir(self):
        self.model_dir=os.path.join(self.parent_dir,self.gan_type)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        print('GAN Model directory is ',self.model_dir)
    def prepare_logger(self):
        self.logger=logging.getLogger(self.gan_type)
        pth=os.path.join(self.model_dir,'tvd.csv')
        file_handler=logging.FileHandler(pth)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info('iter tvd mvd')
    def log_tvd(self,step,tvd,mvd):
        log_str=' '.join([str(step),str(tvd),str(mvd)])
        self.logger.info(log_str)


class Trainer(object):
    def __init__(self,config,data_type):
        self.config=config
        self.data_type=data_type
        self.prepare_model_dir()



        #with tf.variable_scope('trainer'):#commented to get summaries on same plot
        self.step=tf.Variable(0,'step')
        self.inc_step=tf.assign(self.step,self.step+1)
        self.build_model()

        self.summary_writer=tf.summary.FileWriter(self.model_dir)

        self.saver=tf.train.Saver()

        #sv = tf.train.Supervisor(
        #                        logdir=self.save_model_dir,
        #                        is_chief=True,
        #                        saver=self.saver,
        #                        summary_op=None,
        #                        summary_writer=self.summary_writer,
        #                        save_model_secs=300,
        #                        global_step=self.step,
        #                        ready_for_local_init_op=None
        #                        )

        gpu_options = tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=0.333)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        #self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        self.sess = tf.Session(config=sess_config)


        init=tf.global_variables_initializer()
        self.sess.run(init)

        #if load_path, replace initialized values
        if self.config.load_path:
            print(" [*] Attempting to restore {}".format(self.config.load_path))
            self.saver.restore(self.sess,self.config.load_path)

            #print(" [*] Attempting to restore {}".format(ckpt))
            #self.saver.restore(self.sess,ckpt)
            #print(" [*] Success to read {}".format(ckpt))



        if not self.config.load_path:
            #once data scatterplot (doesn't change during training)
            self.data_scatterplot()


    def data_scatterplot(self):
        Xd=self.sess.run(self.data.X,{self.data.N:5000})
        X1,X2,X3=np.split(Xd,3,axis=1)
        x1x2,x1x3,x2x3 = summary_scatterplots(X1,X2,X3)
        step,Pg_summ=self.sess.run([self.step,self.d_scatter_summary],{self.tf_scatter:np.concatenate([x1x2,x1x3,x2x3])})
        self.summary_writer.add_summary(Pg_summ,step)
        self.summary_writer.flush()


    def build_model(self):
        self.data=DataTypes[self.data_type](self.config.batch_size)

        self.gans=[GAN(self.config,n,self.data,self.model_dir) for n in GeneratorTypes.keys()]

        with tf.control_dependencies([self.inc_step]):
            self.train_op=tf.group(*[gan.train_op for gan in self.gans])
            #self.train_op=tf.group(gan.train_op for gan in self.gans.values())

        #Used for generating image summaries of scatterplots
        self.tf_scatter=tf.placeholder(tf.uint8,[3,480,640,3])
        self.d_scatter_summary=tf.summary.image('scatter_Data_'+self.data_type,self.tf_scatter,max_outputs=3)


    def train(self):
        self.train_timer   =Timer()
        self.losses_timer  =Timer()
        self.tvd_timer     =Timer()
        self.scatter_timer =Timer()

        self.log_step=50
        self.max_step=50001
        #self.max_step=501
        for step in trange(self.max_step):

            if step % self.log_step == 0:
                for gan in self.gans:
                    self.losses_timer.on()
                    gan.record_losses(self.sess)
                    self.losses_timer.off()

                    self.tvd_timer.on()
                    gan.record_tvd(self.sess)
                    self.tvd_timer.off()

            if step % (10*self.log_step) == 0:
                for gan in self.gans:
                    self.scatter_timer.on()
                    gan.record_scatter(self.sess)

                    #DEBUG: reassure me nothing changes during optimization
                    #self.data_scatterplot()

                    self.scatter_timer.off()

            if step % (5000) == 0:
                self.saver.save(self.sess,self.save_model_name,step)

            self.train_timer.on()
            self.sess.run(self.train_op)
            self.train_timer.off()


        print("Timers:")
        print(self.train_timer)
        print(self.losses_timer)
        print(self.tvd_timer)
        print(self.scatter_timer)


    def prepare_model_dir(self):
        if self.config.load_path:
            self.model_dir=self.config.load_path
        else:
            pth=datetime.now().strftime("%m%d_%H%M%S")+'_'+self.data_type
            self.model_dir=os.path.join(self.config.model_dir,pth)


        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        print('Model directory is ',self.model_dir)

        self.save_model_dir=os.path.join(self.model_dir,'checkpoints')
        if not os.path.exists(self.save_model_dir):
            os.mkdir(self.save_model_dir)
        self.save_model_name=os.path.join(self.save_model_dir,'Model')


        param_path = os.path.join(self.model_dir, "params.json")
        print("[*] MODEL dir: %s" % self.model_dir)
        print("[*] PARAM path: %s" % param_path)
        with open(param_path, 'w') as fp:
            json.dump(self.config.__dict__, fp, indent=4, sort_keys=True)

        config=self.config
        if config.is_train and not config.load_path:
            config.log_code_dir=os.path.join(self.model_dir,'code')
            for path in [self.model_dir, config.log_code_dir]:
                if not os.path.exists(path):
                    os.makedirs(path)

            #Copy python code in directory into model_dir/code for future reference:
            code_dir=os.path.dirname(os.path.realpath(sys.argv[0]))
            model_files = [f for f in listdir(code_dir) if isfile(join(code_dir, f))]
            for f in model_files:
                if f.endswith('.py'):
                    shutil.copy2(f,config.log_code_dir)







