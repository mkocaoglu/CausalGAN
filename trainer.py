from __future__ import print_function
import tensorflow as tf
from causal_controller.CausalController import CausalController
from tqdm import trange
import os
import pandas as pd

from utils import make_summary
from data_loader import DataLoader
from figure_scripts.pairwise import crosstab





class Trainer(object):

    def __init__(self,config,cc_config,model_config=None):
        self.config=config
        self.cc_config=cc_config
        self.model_dir = config.model_dir
        self.cc_config.model_dir=config.model_dir

        self.model_config=model_config
        if self.model_config:
            self.model_config.model_dir=config.model_dir


        self.load_path = config.load_path
        self.use_gpu = config.use_gpu


        #This tensor controls batch_size for all models
        #Not expected to change during training, but during inference it can be
        #helpful to change it
        self.batch_size=tf.placeholder_with_default(self.config.batch_size,[],name='batch_size')

        loader_batch_size=config.num_devices*config.batch_size

        #Data
        print('setting up data')
        self.data=DataLoader(config)

        #Always need to build CC
        cc_batch_size=config.num_devices*self.batch_size#Tensor/placeholder
        self.cc=CausalController(cc_batch_size,cc_config)
        self.step=self.cc.step

        if self.cc_config.is_pretrain or self.cc_config.build_all:
            print('setup pretrain')
            #queue system to feed labels quickly. Does not queue images
            label_queue= self.data.get_label_queue(loader_batch_size)
            self.cc.build_pretrain(label_queue)


        #Build Model
        elif self.model_config:
            #Will build both gen and discrim
            self.model=self.config.Model(self.batch_size,self.model_config)

            #Trainer step is defined as cc.step+model.step
            #e.g. 10k iter pretrain and 100k iter image model
            #will have image summaries at 100k but trainer model saved at Model-110k
            self.step+=self.model.step


            data_queue=self.data.get_data_queue(loader_batch_size)

            self.real_data_by_gpu = distribute_input_data(data_queue,config.num_gpu)
            self.fake_data_by_gpu = distribute_input_data(self.cc.label_dict,config.num_gpu)

            with tf.variable_scope('tower'):
                for gpu in get_available_gpus():
                    gpu_idx+=1
                    print('using device:',gpu)

                    real_data=real_data_by_gpu[gpu]
                    fake_data=fake_data_by_gpu[gpu]
                    tower=gpu.replace('/','').replace(':','_')

                    with tf.device(gpu),tf.name_scope(tower):
                        #Build num_gpu copies of graph: inputs->gradient
                        #Updates self.tower_dict
                        self.model(real_data,fake_data)

                    #allow future gpu to use same variables
                    tf.get_variable_scope().reuse_variables()

        else:
            print('No training to occur')

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=4)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        print('trainer.model_dir:',self.model_dir)
        gpu_options = tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=0.333)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)


        sv = tf.train.Supervisor(
                                logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None
                                )
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        #ckpt = tf.train.get_checkpoint_state(self.model_dir)
        #if ckpt and ckpt.model_checkpoint_path:
        #    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        #    self.saver.restore(self.sess, os.path.join(self.model_dir, ckpt_name))
        #    print(" [*] Success to read {}".format(ckpt_name))


        if cc_config.pt_load_path:
            print('Attempting to load pretrain model:',cc_config.pt_load_path)
            self.cc.load(self.sess,cc_config.pt_load_path)

            print('Check tvd after restore')
            info=crosstab(self,report_tvd=True)
            print('tvd after load:',info['tvd'])



        #PREPARE training:
        #This guy is a dictionary of all possible z tensors
        #he has 1 for every causal label plus one called 'z_gen'
        #Use him to sample z and to feed z in
        #self.z_fd=self.cc.sample_z.copy()
        #self.z_fd.update({'z_gen':self.z_gen})




    def pretrain_loop(self,num_iter=None):
        '''
        num_iter : is the number of *additional* iterations to do
        baring one of the quit conditions (the model may already be
        trained for some number of iterations). Defaults to
        cc_config.pretrain_iter.

        '''
        #TODO: potentially should be moved into CausalController for consistency

        num_iter = num_iter or self.cc.config.pretrain_iter


        if hasattr(self,'model'):
            model_step=self.sess.run(self.model.step)
            assert model_step==0,'if pretraining, model should not be trained already'

        cc_step=self.sess.run(self.cc.step)
        if cc_step>0:
            print('Resuming training of already optimized CC model at\
                  step:',cc_step)

        label_stats=crosstab(self,report_tvd=True)

        def break_pretrain(label_stats,counter):
            c1=counter>=self.cc.config.min_pretrain_iter
            c2= (label_stats['tvd']<self.cc.config.min_tvd)
            return (c1 and c2)

        for counter in trange(cc_step,cc_step+num_iter):
            #Check for early exit
            if counter %(10*self.cc.config.log_step)==0:
                label_stats=crosstab(self,report_tvd=True)
                print('ptstep:',counter,'  TVD:',label_stats['tvd'])
                if break_pretrain(label_stats,counter):
                    print('Completed Pretrain by TVD Qualification')
                    break

            #Optimize critic
            self.cc.critic_update(self.sess)

            #one iter causal controller
            fetch_dict = {
                "pretrain_op": self.cc.train_op,
                'cc_step':self.cc.step,
                'step':self.step,
            }

            #update what to run
            if counter % self.cc.config.log_step == 0:
                fetch_dict.update({
                    "summary": self.cc.summary_op,
                    "c_loss": self.cc.c_loss,
                    "dcc_loss": self.cc.dcc_loss,
                })
            result = self.sess.run(fetch_dict)

            #update summaries
            if counter % self.cc.config.log_step == 0:
                if counter %(10*self.cc.config.log_step)==0:
                    sum_tvd=make_summary('misc/tvd', label_stats['tvd'])
                    self.summary_writer.add_summary(sum_tvd,result['cc_step'])

                self.summary_writer.add_summary(result['summary'],result['cc_step'])
                self.summary_writer.flush()

                c_loss = result['c_loss']
                dcc_loss = result['dcc_loss']
                print("[{}/{}] Loss_C: {:.6f} Loss_DCC: {:.6f}".\
                      format(counter, cc_step+ num_iter, c_loss, dcc_loss))

            if counter %(10*self.cc.config.log_step)==0:
                self.cc.saver.save(self.sess,self.cc.save_model_dir,result['cc_step'])

        else:
            label_stats=crosstab(self,report_tvd=True)
            self.cc.saver.save(self.sess,self.cc.save_model_dir,self.cc.step)
            print('Completed Pretrain by Exhausting all Pretrain Steps!')

        print('step:',result['cc_step'],'  TVD:',label_stats['tvd'])





