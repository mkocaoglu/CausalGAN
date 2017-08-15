from __future__ import print_function
import tensorflow as tf
from causal_controller.Causal_controller import CausalController
import os
import pandas as pd

from data_loader import DataLoader






class Trainer(object):

    def __init__(self,config,cc_config,model_config):
        self.config=config
        self.cc_config=cc_config
        self.model_config=began_config
        self.model_dir = config.model_dir
        self.load_path = config.load_path
        self.use_gpu = config.use_gpu


        #This tensor controls batch_size for all models
        #Not expected to change during training, but during inference it can be
        #helpful to change it
        self.batch_size=tf.placeholder_with_default(self.config.batch_size,name='batch_size')

        loader_batch_size=config.num_devices*config.batch_size

        #Data
        print('setting up data')
        self.data=DataLoader(config)

        #Always need to build CC
        cc_batch_size=config.num_devices*self.batch_size#Tensor/placeholder
        self.cc=CausalController(cc_batch_size,cc_config,label_loader)

        if self.cc_config.is_pretrain or self.config.build_all:
            print('setup pretrain')
            #queue system to feed labels quickly. Does not queue images
            label_queue= self.data.get_label_queue(loader_batch_size)
            self.cc.build_pretrain(label_queue)

            #Trainer step is defined as cc.step+model.step
            #e.g. 10k iter pretrain and 100k iter image model
            #will have image summaries at 100k but trainer model saved at Model-110k
            self.step=self.cc.step

        #TODO:Redirect references to self.data.attr
        ##For use elsewhere during inference.
        #attributes = pd.read_csv(cc_config.attr_file,delim_whitespace=True) #+-1
        #self.attr = 0.5*(attributes+1) #labels are 0,1


        #Build Model
        if self.config.model_type:
            #Will build both gen and discrim
            self.model=self.config.Model(self.batch_size,self.model_config)
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


        ####set self.data_loader to correspond to first gpu
        #if config.num_gpu>1:
        #    self.data_by_gpu=distribute_input_data(data_loader,config.num_gpu)
        #    #data_loader is used for evaluation/summaries
        #    self.data_loader=self.data_by_gpu.values()[0]
        #else:
        #    self.data_loader = data_loader

        #TODO clean


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
            info=crosstab(self,result_dir=self.cc.model_dir,report_tvd=True)
            print('tvd after load:',info['tvd'])






#        self.label_stats=label_stats
#        #Standardize encapsulation of intervention range
#        ml=self.label_stats['min_logit'].to_dict()
#        Ml=self.label_stats['max_logit'].to_dict()
#        self.intervention_range={name:[ml[name],Ml[name]] for name in ml.keys()}



