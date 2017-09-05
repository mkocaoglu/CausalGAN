from __future__ import print_function
import numpy as np
import tensorflow as tf
from causal_controller.CausalController import CausalController
from tqdm import trange
import os
import pandas as pd

from utils import make_summary,distribute_input_data,get_available_gpus
from utils import save_image

from data_loader import DataLoader
from figure_scripts.pairwise import crosstab

class Trainer(object):

    def __init__(self, config, cc_config, model_config=None):
        self.config=config
        self.cc_config=cc_config
        self.model_dir = config.model_dir
        self.cc_config.model_dir=config.model_dir

        self.model_config=model_config
        if self.model_config:
            self.model_config.model_dir=config.model_dir

        self.save_model_dir=os.path.join(self.model_dir,'checkpoints')
        if not os.path.exists(self.save_model_dir):
            os.mkdir(self.save_model_dir)

        self.summary_dir=os.path.join(self.model_dir,'summaries')
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)

        self.load_path = config.load_path
        self.use_gpu = config.use_gpu

        #This tensor controls batch_size for all models
        #Not expected to change during training, but during testing it can be
        #helpful to change it

        self.batch_size=tf.placeholder_with_default(self.config.batch_size,[],name='batch_size')

        loader_batch_size=config.num_devices*config.batch_size

        #Always need to build CC
        print('setting up CausalController')
        cc_batch_size=config.num_devices*self.batch_size#Tensor/placeholder
        self.cc=CausalController(cc_batch_size,cc_config)
        self.step=self.cc.step

        #Data
        print('setting up data')
        self.data=DataLoader(self.cc.label_names,config)

        if self.cc_config.is_pretrain or self.config.build_pretrain:
            print('setup pretrain')
            #queue system to feed labels quickly. This does not queue images
            label_queue= self.data.get_label_queue(loader_batch_size)
            self.cc.build_pretrain(label_queue)

        #Build Model
        if self.model_config:
            #Will build both gen and discrim
            self.model=self.config.Model(self.batch_size,self.model_config)

            #Trainer step is defined as cc.step+model.step
            #e.g. 10k iter pretrain and 100k iter image model
            #will have image summaries at 100k but trainer model saved at Model-110k
            self.step+=self.model.step

            # This queue holds (image,label) pairs, and is used for training conditional GANs
            data_queue=self.data.get_data_queue(loader_batch_size)

            self.real_data_by_gpu = distribute_input_data(data_queue,config.num_gpu)
            self.fake_data_by_gpu = distribute_input_data(self.cc.label_dict,config.num_gpu)

            with tf.variable_scope('tower'):
                for gpu in get_available_gpus():
                    print('using device:',gpu)

                    real_data=self.real_data_by_gpu[gpu]
                    fake_data=self.fake_data_by_gpu[gpu]
                    tower=gpu.replace('/','').replace(':','_')

                    with tf.device(gpu),tf.name_scope(tower):
                        #Build num_gpu copies of graph: inputs->gradient
                        #Updates self.tower_dict
                        self.model(real_data,fake_data)

                    #allow future gpu to use same variables
                    tf.get_variable_scope().reuse_variables()

            if self.model_config.is_train or self.config.build_train:
                self.model.build_train_op()
                self.model.build_summary_op()

        else:
            print('Image model not built')

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)

        print('trainer.model_dir:',self.model_dir)
        gpu_options = tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=0.333)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        sv = tf.train.Supervisor(
                                logdir=self.save_model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None
                                )
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if cc_config.pt_load_path:
            print('Attempting to load pretrain model:',cc_config.pt_load_path)
            self.cc.load(self.sess,cc_config.pt_load_path)

            print('Check tvd after restore')
            info=crosstab(self,report_tvd=True)
            print('tvd after load:',info['tvd'])

            #save copy of cc model in new dir
            cc_step=self.sess.run(self.cc.step)
            self.cc.saver.save(self.sess,self.cc.save_model_name,cc_step)

        if config.load_path:#Declare loading point
            pnt_str='Loaded variables at ccStep:{}'
            cc_step=self.sess.run(self.cc.step)
            pnt_str=pnt_str.format(cc_step)
            print('pntstr',pnt_str)
            if self.model_config:
                pnt_str+=' imagemodelStep:{}'
                model_step=self.sess.run
                pnt_str=pnt_str.format(model_step)
            print(pnt_str)

        #PREPARE training:
        #TODO save as Variables so they are restored to same values when load model
        fixed_batch_size=256 #get this many fixed z values

        self.fetch_fixed_z={n.z:n.z for n in self.cc.nodes}
        if model_config:
            self.fetch_fixed_z[self.model.z_gen]=self.model.z_gen

        #feed_dict that ensures constant inputs
        #add feed_fixed_z[self.cc.Male.label]=1*ones() to intervene
        self.feed_fixed_z=self.sess.run(self.fetch_fixed_z,{self.batch_size:fixed_batch_size})

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
                self.cc.saver.save(self.sess,self.cc.save_model_name,result['cc_step'])

        else:
            label_stats=crosstab(self,report_tvd=True)
            self.cc.saver.save(self.sess,self.cc.save_model_name,self.cc.step)
            print('Completed Pretrain by Exhausting all Pretrain Steps!')

        print('step:',result['cc_step'],'  TVD:',label_stats['tvd'])


    def train_loop(self,num_iter=None):
        '''
        This is a function for handling the training of either CausalBEGAN or
        CausalGAN models. The python function Model.train_step() is called
        num_iter times and some general image save features: intervening,
        conditioning, etc are done here too.
        '''
        num_iter=num_iter or self.model_config.num_iter

        #Train loop
        print('Entering train loop..')
        for counter in trange(num_iter):

            self.model.train_step(self.sess,counter)

            #scalar and histogram summaries
            if counter % self.config.log_step == 0:
                step,summ=self.sess.run([self.model.step,self.model.summary_op])
                self.summary_writer.add_summary(summ,step)
                self.summary_writer.flush()

            #expensive summaries
            if counter % (self.config.log_step * 50) == 0:
                self.causal_sampling([8,16])
                self.label_interpolation()
                self.sample_diversity()

            #more rare events
            if counter % (self.config.log_step * 100) == 0:
                self.causal_sampling([2,10])

    ##Wrapper methods
    def sample_label(self, cond_dict=None, do_dict=None,N=None):
        return self.cc.sample_label(self.sess,cond_dict=cond_dict,do_dict=do_dict,N=N)
    ##

    ##Sampling and figure methods
    def label_interpolation(self,inputs=None,save_dir=None,ext='.pdf'):
        '''
        Holding all other inputs the same, move a causal controller
        labels between 0 and 1. Recalculate the downstream effects to capture the causal effect.

        For each label, this makes an 8x8 image with each row being
        an instance of z_fixed with varying label
        '''

        interpolation_dir=os.path.join(self.model_dir,'label_interpolation')
        save_dir=save_dir or interpolation_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        inputs=inputs or {}

        #use the first 8 values
        #contrasting np.repeat and np.tile to get all combinations
        fixed_z=inputs or {k:np.repeat(v[:8],8,axis=0) for k,v in self.feed_fixed_z.items()}
        setval=np.tile(np.linspace(0,1,8),8).reshape([64,1])

        fixed_z.update({self.batch_size:64})
        save_name='{}/{}_G_interp_{}'+ext

        #make 8x8 image
        for node in self.cc.nodes:

            fd=fixed_z.copy()
            fd[node.label]=setval
            images,step=self.sess.run([self.model.G,self.model.step],fd)
            interp_path=save_name.format(save_dir,step,node.name)
            save_image(images,interp_path,nrow=8)

        out_str="[*] Interpolation Samples saved: "+save_name
        print(save_name.format(save_dir,step,'*'))

    def causal_sampling(self, img_shape ,ext='.pdf'):
        '''
        sampling new noise inputs each time, draw samples from
        interventional distributions.
        Recalculate downstream effects given a label value

        img_shape must have rows divisible by 2

        This function implements the following three sampling techniques: 
        1) Images where 
            Top half is sampled from the intervention do(label=1)
            Bottom half is sampled from the intervention do(label=0)
        2) Images where
            Top half is sampled from the intervention do(label=1/0)
            Bottom half is sampled conditioned on |label = 1/0
        3) Image where 
            Top half is sampled conditioned on |label = 1
            Bottom half is sampled conditioned on |label = 0
        '''

        assert len(img_shape)==2,'2d shape for output'
        assert img_shape[0]%2==0,'should have equal top and bot half'

        shape_str='_'+'x'.join(map(str,img_shape))

        #sample given(Label=1/0)
        conditioning_dir=os.path.join(self.model_dir,'label_conditioning')
        if not os.path.exists(conditioning_dir):
            os.mkdir(conditioning_dir)

        #sample do(Label=1/0)
        intervention_dir=os.path.join(self.model_dir,'label_intervention')
        if not os.path.exists(intervention_dir):
            os.mkdir(intervention_dir)

        #sample do(Label=1)/given(Label=1)
        #sample do(Label=0)/given(Label=0)
        intv_v_conditioning_dir=os.path.join(self.model_dir,'label_intv_v_conditioning')
        if not os.path.exists(intv_v_conditioning_dir):
            os.mkdir(intv_v_conditioning_dir)

        save_name_cond =os.path.join(conditioning_dir,'{}_condition_{}'+shape_str+ext)
        save_name_intv =os.path.join(intervention_dir,'{}_interv_{}'+shape_str+ext)
        save_name_intvcond=os.path.join(intv_v_conditioning_dir,'{}_intvcond_{}={}'+shape_str+ext)

        half_shape=[img_shape[0]//2, img_shape[1]]
        N=np.prod(half_shape)

        for name in self.cc.node_names:
            #First sample labels (two step more efficient)
            #ex:{'Male':1}
            c0=self.sample_label(cond_dict={name:0},N=N)
            c1=self.sample_label(cond_dict={name:1},N=N)
            d0=self.sample_label(do_dict=  {name:0},N=N)
            d1=self.sample_label(do_dict=  {name:1},N=N)

            feed_c0={self.cc.label_dict[k]:v for k,v in c0.iteritems()}
            feed_c1={self.cc.label_dict[k]:v for k,v in c1.iteritems()}
            feed_d0={self.cc.label_dict[k]:v for k,v in d0.iteritems()}
            feed_d1={self.cc.label_dict[k]:v for k,v in d1.iteritems()}

            feed_c0[self.batch_size]=N
            feed_c1[self.batch_size]=N
            feed_d0[self.batch_size]=N
            feed_d1[self.batch_size]=N

            step=self.sess.run(self.model.step)
            c0_images=self.sess.run(self.model.G,feed_c0)
            c1_images=self.sess.run(self.model.G,feed_c1)
            d0_images=self.sess.run(self.model.G,feed_d0)
            d1_images=self.sess.run(self.model.G,feed_d1)

            save_path_cond      = save_name_cond.format(step,name)
            save_path_intv      = save_name_intv.format(step,name)
            save_path_intvcond0 = save_name_intvcond.format(step,name,0)
            save_path_intvcond1 = save_name_intvcond.format(step,name,1)

            #saveimage fills row by row from top left
            save_image(np.concatenate([c1_images,c0_images]),save_path_cond,nrow=img_shape[0])
            save_image(np.concatenate([d1_images,d0_images]),save_path_intv,nrow=img_shape[0])
            save_image(np.concatenate([d0_images,c0_images]),save_path_intvcond0,nrow=img_shape[0])
            save_image(np.concatenate([d1_images,c1_images]),save_path_intvcond1,nrow=img_shape[0])

        print("[*] Conditioning Samples saved: "+conditioning_dir)
        print("[*] Intervention Samples saved: "+intervention_dir)
        print("[*] Intervention vs Condition Samples saved: "+intv_v_conditioning_dir)


    def sample_diversity(self,save_dir=None,ext='.pdf'):
        '''
        This is to make a 16x16 image from fixed inputs
        to examine the image diversity over time
        '''
        #Make 16x16 image
        nrow=16
        diversity_dir=os.path.join(self.model_dir,'image_diversity')
        save_dir=save_dir or diversity_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name=os.path.join(save_dir,'{}_G_diversity'+ext)

        feed_fixed={k:v[:256] for k,v in self.feed_fixed_z.items()}
        feed_fixed.update({self.batch_size:256})

        step,images = self.sess.run([self.model.step,self.model.G], feed_dict=feed_fixed)

        print('image shape',images.shape)

        save_path=save_name.format(step)
        save_image(images, save_path, nrow=nrow)
        print("[*] Diversity Sample saved: {}".format(save_path))

