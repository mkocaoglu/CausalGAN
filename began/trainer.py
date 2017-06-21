from __future__ import print_function
from utils import save_image,distribute_input_data,summary_stats,make_summary
import pandas as pd
import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from figure_scripts.pairwise import crosstab
from figure_scripts.sample import intervention2d,condition2d

from models import *
from Causal_controller import CausalController


from IPython.core import debugger
debug = debugger.Pdb().set_trace

#custom collections

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Trainer(object):
    model_type='began'
    def __init__(self, config, data_loader,label_stats):

        self.config = config
        self.dataset = config.dataset
        self.graph=config.graph
        self.label_stats=label_stats

        #Standardize encapsulation of intervention range
        ml=self.label_stats['min_logit'].to_dict()
        Ml=self.label_stats['max_logit'].to_dict()
        self.intervention_range={name:[ml[name],Ml[name]] for name in ml.keys()}

        label_names = zip(*self.graph)[0]
        label_names=list(label_names)
        attributes = pd.read_csv(config.attr_file,delim_whitespace=True) #+-1
        self.attr = 0.5*(attributes+1)[label_names]

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.separate_labeler=config.separate_labeler

        if self.config.pretrain_type=='wasserstein':
            self.DCC=DiscriminatorW
        elif self.config.pretrain_type=='gan':
            self.DCC=Discriminator_CC

        self.g_step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        self.lambda_k = config.lambda_k
        self.lambda_l = config.lambda_l
        self.lambda_z = config.lambda_z
        self.gamma = config.gamma
        self.gamma_label = config.gamma_label
        self.zeta=config.zeta

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        self.data_by_gpu = distribute_input_data(data_loader,config.num_gpu)
        ####set self.data_loader to correspond to first gpu
        #if config.num_gpu>1:
        #    self.data_by_gpu=distribute_input_data(data_loader,config.num_gpu)
        #    #data_loader is used for evaluation/summaries
        #    self.data_loader=self.data_by_gpu.values()[0]
        #else:
        #    self.data_loader = data_loader

        #TODO clean
        _, height, width, self.channel = \
                get_conv_shape(data_loader['x'], self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step#Not used
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train

        self.build_model()####multigpu stuff here

        self.pt_var=self.cc.var+self.dcc_var

        #Save dcc vars too??
        #self.pt_saver=tf.train.Saver(var_list=self.pt_var)
        self.pt_saver=tf.train.Saver(var_list=self.cc.var)

        self.pt_dir=os.path.join(self.model_dir,'pretrain')
        if not os.path.exists(self.pt_dir):
            os.mkdir(self.pt_dir)


        self.saver = tf.train.Saver(var_list=self.var)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        print('self.model_dir:',self.model_dir)
        gpu_options = tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=0.333)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)


        #if config.encode_image:
        #if True:
        if False:
            #Hack because I can have graph.finalize be called
            #doesn't use queuerunners but should be no problem?
            sm=tf.train.SessionManager()
            self.sess=sm.prepare_session(
                                    master='',
                                    saver=self.saver,
                                    checkpoint_dir=self.model_dir,
                                    config=sess_config,
                                   )

        else:
            #Main way to go
            sv = tf.train.Supervisor(logdir=self.model_dir,
                                    is_chief=True,
                                    saver=self.saver,
                                    summary_op=None,
                                    summary_writer=self.summary_writer,
                                    save_model_secs=300,
                                    global_step=self.step,
                                    ready_for_local_init_op=None)
            self.sess = sv.prepare_or_wait_for_session(config=sess_config)


        if config.pt_load_path:
            print('Attempting to load pretrain model:',config.pt_load_path)
            self.pt_saver.restore(self.sess,config.pt_load_path)

        #if not self.is_train:
        #    # dirty way to bypass graph finilization error
        #    g = tf.get_default_graph()
        #    g._finalized = False

        #    self.build_test_model()

    def pretrain(self):
        step,global_step=self.sess.run([self.cc.step,self.g_step])
        assert global_step==0,'if pretraining, model should not be trained already'

        stats=crosstab(self,result_dir=self.pt_dir,report_tvd=True)

        def break_pretrain(stats):
            return (stats['tvd']<self.config.min_tvd)

        for counter in trange(self.config.pretrain_iter):
            #Check for early exit
            if counter %(10*self.log_step)==0:
                stats=crosstab(self,result_dir=self.pt_dir,report_tvd=True)
                print('ptstep:',counter,'  TVD:',stats['tvd'])
                if break_pretrain(stats):
                    print('Completed Pretrain by TVD Qualification')
                    break

            #Optimize critic
            if self.config.pretrain_type=='wasserstein':
                fetch_dict = {"critic_op":self.dcc_optim }
                for i in range(self.config.n_critic):
                    result = self.sess.run(fetch_dict)

            #one iter causal controller
            fetch_dict = {
                "pretrain_op": self.pretrain_op,
                'cc_step':self.cc.step,
                'step':self.step,
            }
            if counter % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "c_loss": self.c_loss,
                    "dcc_loss": self.dcc_loss,
                })
            result = self.sess.run(fetch_dict)

            if counter % self.log_step == 0:
                if counter %(10*self.log_step)==0:
                    sum_tvd=make_summary('misc/tvd', stats['tvd'])
                    self.summary_writer.add_summary(sum_tvd,result['step'])

                self.summary_writer.add_summary(result['summary'],result['step'])
                self.summary_writer.flush()

                c_loss = result['c_loss']
                dcc_loss = result['dcc_loss']
                print("[{}/{}] Loss_C: {:.6f} Loss_DCC: {:.6f}".\
                      format(step, self.max_step, c_loss, dcc_loss))

            if counter %(10*self.log_step)==0:
                self.pt_saver.save(self.sess,self.pt_dir+'/CC_Model',result['step'])

        else:
            stats=crosstab(self,result_dir=self.pt_dir,report_tvd=True)
            print('Completed Pretrain by Exhaustin all Pretrain Steps!')

        print('step:',step,'  TVD:',stats['tvd'])


    def train(self):
        #dictionary of fixed z inputs(causal and gen)
        z_fixed = self.sess.run(self.z_fd)
        data_fixed=self.get_data_from_loader()
        x_fixed = data_fixed['x']

        #use these to feed fixed values into self.sess
        feed_fixed_z={self.z_fd[k]:val for k,val in z_fixed.items()}
        feed_fixed_data={self.data_fd[k]:val for k,val in data_fixed.items()}

        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)


        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "train_op": self.train_op,
                "measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "global_step": self.step,
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'],
                                                result['global_step'])
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, measure, k_t))

            #if step % (self.log_step * 10) == 0:
            if step % (self.log_step * 50) == 0:
                x_fake = self.generate(feed_fixed_z, self.model_dir, idx=step)
                #self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)
                self.autoencode(data_fixed, self.model_dir, idx=step, x_fake=x_fake)

                self.intervention( z_fixed )


            if step % (self.log_step * 100) == 0:
                self.big_generate()
                #For now on logits even if noisy_labels=False

                for name,node in self.cc.node_dict.items():
                    do_dict={name:'percentile'}
                    cond_dict={name:'percentile'}
                    do_dict_name='per'+name
                    cond_dict_name='per'+name
                    intervention2d( self, do_dict=do_dict, do_dict_name=do_dict_name, on_logits=True)
                    condition2d( self, cond_dict=cond_dict, cond_dict_name=cond_dict_name, on_logits=True)


            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                #cur_measure = np.mean(measure_history)
                #if cur_measure > prev_measure * 0.99:
                #prev_measure = cur_measure


    def build_tower(self,data_loader):
        self.data_loader=data_loader
        self.data_fd = data_loader

        #The keys of data_loader are all the labels union 'x'
        self.x = self.data_loader['x']#no
        x = norm_img(self.x)

        label_names=zip(*self.graph)[0]#the names of all the labels we are using
        n_labels=len(label_names)
        #print('shape of each label:',self.data_loader['Bald'].get_shape().as_list())
        self.real_labels_list=[self.data_loader[name] for name in label_names]
        self.real_labels=tf.concat(self.real_labels_list,-1)

#(self,graph,batch_size=1,indep_causal=False,n_layers=3,n_hidden=10,input_dict=None,reuse=None)
        self.cc=CausalController(self.graph,self.batch_size,
                        indep_causal=self.config.indep_causal,
                        n_layers=self.config.cc_n_layers,
                        n_hidden=self.config.cc_n_hidden)

        #maybe needs reshaped? [bs,] -> [bs,1]
        self.fake_labels= tf.concat( self.cc.list_labels(),-1 )
        self.fake_labels_logits= tf.concat( self.cc.list_label_logits(),-1 )
        #print('shape of fake_labels:',self.fake_labels.get_shape().as_list())

        n_hidden=self.config.critic_hidden_size
        #self.real_label_dict={n:self.data_loader[n] for n in self.cc.node_names}
        #self.fake_label_dict={n:self.cc.node_dict[n].label for n in self.cc.node_names}

        #Pretrain node by node by computing a new discriminator for each node
        #given the true value of its parents
        if self.config.pt_factorized:
            self.DCC=FactorizedNetwork(self.graph,self.DCC)

        self.dcc_real,self.dcc_real_logit,self.dcc_var=self.DCC(self.real_labels,self.batch_size,n_hidden=n_hidden)
        self.dcc_fake,self.dcc_fake_logit,self.dcc_var=self.DCC(self.fake_labels,self.batch_size,reuse=True,n_hidden=n_hidden)




        #z_num is 64 or 128 in paper
        self.z_gen = tf.random_uniform(
            (self.batch_size, self.z_num), minval=-1.0, maxval=1.0)

        #This guy is a dictionary of all possible z tensors
        #he has 1 for every causal label plus one called 'z_gen'
        #Use him to sample z and to feed z in
        self.z_fd=self.cc.sample_z.copy()
        self.z_fd.update({'z_gen':self.z_gen})

        #self.z= tf.concat( [self.fake_labels_logits, self.z_gen],axis=-1,name='z')
        self.z= tf.concat( [self.fake_labels, self.z_gen],axis=-1,name='z')


        G, self.G_var = GeneratorCNN(
                self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format)

        '''
        my approach was to just pretend 3 of the vars in the encoded space
        represented our causal variables
        z_num is only 64,I am using 3 for labels
        so if we use like 20 causal labels, make it larger

        TODO:actually I think it would have been more normal to pass labels through
        encoder and decoder. basically began but with (x,y) in place of x
        we can try this later (especially if there is label collapse)
        '''
        d_out, self.D_z, self.D_var = DiscriminatorCNN(
                tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format)
        AE_G, AE_x = tf.split(d_out, 2)

        self.D_encode_G, self.D_encode_x=tf.split(self.D_z, 2)#axis=0 by default

        if not self.separate_labeler:
            self.D_fake_labels_logits=tf.slice(self.D_encode_G,[0,0],[-1,n_labels])
            self.D_real_labels_logits=tf.slice(self.D_encode_x,[0,0],[-1,n_labels])
        else:

            label_logits,self.DL_var=Discriminator_labeler(
                    tf.concat([G, x], 0), len(self.cc.nodes), self.repeat_num,
                    self.conv_hidden_num, self.data_format)
            self.D_fake_labels_logits,self.D_real_labels_logits=tf.split(label_logits,2)

            self.D_var += self.DL_var


        self.D_real_labels=tf.sigmoid(self.D_real_labels_logits)
        self.D_fake_labels=tf.sigmoid(self.D_fake_labels_logits)
        self.D_real_labels_list=tf.split(self.D_real_labels,n_labels,axis=1)
        self.D_fake_labels_list=tf.split(self.D_fake_labels,n_labels,axis=1)


        #"sigmoid_cross_entropy_with_logits" is really long
        def sxe(logits,labels):
            #use zeros or ones if pass in scalar
            if not isinstance(labels,tf.Tensor):
                labels=labels*tf.ones_like(logits)
            return tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,labels=labels)


        #pretrain
        if self.config.pretrain_type=='gan':
            self.dcc_xe_real=tf.reduce_mean(sxe(self.dcc_real_logit,1))
            self.dcc_xe_fake=tf.reduce_mean(sxe(self.dcc_fake_logit,0))
            self.dcc_loss_real = tf.reduce_mean(self.dcc_xe_real)
            self.dcc_loss_fake = tf.reduce_mean(self.dcc_xe_fake)
            self.dcc_loss=self.dcc_loss_real+self.dcc_loss_fake
            self.c_xe_fake=sxe(self.dcc_fake_logit,1)
            self.c_loss=tf.reduce_mean(self.c_xe_fake)
        elif self.config.pretrain_type=='wasserstein':
            self.dcc_diff = self.dcc_fake_logit - self.dcc_real_logit
            self.dcc_gan_loss=tf.reduce_mean(self.dcc_diff)
            self.dcc_grad_loss,self.dcc_slopes=Grad_Penalty(self.real_labels,self.fake_labels,self.DCC,self.config)
            self.dcc_loss=self.dcc_gan_loss+self.dcc_grad_loss
            self.c_loss=-tf.reduce_mean(self.dcc_fake_logit)

        else:
            raise ValueError('shouldnt happen')

        #Should we round fake labels before calc loss?
        if self.config.round_fake_labels:
            fake_labels=tf.round(self.fake_labels)
        else:
            fake_labels=self.fake_labels

        self.d_xe_real_label=sxe(self.D_real_labels_logits,self.real_labels)
        self.d_xe_fake_label=sxe(self.D_fake_labels_logits,fake_labels)
        self.g_xe_label=sxe(self.fake_labels_logits, self.D_fake_labels)

        self.d_absdiff_real_label=tf.abs(self.D_real_labels  - self.real_labels)
        self.d_absdiff_fake_label=tf.abs(self.D_fake_labels  - fake_labels)
        self.g_absdiff_label=tf.abs(fake_labels  -  self.D_fake_labels)

        self.d_squarediff_real_label=tf.square(self.D_real_labels  - self.real_labels)
        self.d_squarediff_fake_label=tf.square(self.D_fake_labels  - fake_labels)
        self.g_squarediff_label=tf.square(fake_labels  -  self.D_fake_labels)


        if self.config.label_loss=='xe':
            self.d_loss_real_label = tf.reduce_mean(self.d_xe_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_xe_fake_label)
            self.g_loss_label=tf.reduce_mean(self.g_xe_label)
        elif self.config.label_loss=='absdiff':
            self.d_loss_real_label = tf.reduce_mean(self.d_absdiff_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_absdiff_fake_label)
            self.g_loss_label = tf.reduce_mean(self.g_absdiff_label)
        elif self.config.label_loss=='squarediff':
            self.d_loss_real_label = tf.reduce_mean(self.d_squarediff_real_label)
            self.d_loss_fake_label = tf.reduce_mean(self.d_squarediff_fake_label)
            self.g_loss_label = tf.reduce_mean(self.g_squarediff_label)


        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        u1=tf.abs(AE_x - x)
        u2=tf.abs(AE_G - G)
        m1=tf.reduce_mean(u1)
        m2=tf.reduce_mean(u2)
        c1=tf.reduce_mean(tf.square(u1-m1))
        c2=tf.reduce_mean(tf.square(u2-m2))
        self.eqn2 = tf.square(m1-m2)
        self.eqn1 = (c1+c2-2*tf.sqrt(c1*c2))/self.eqn2


        ##New label-margin loss:
        self.d_loss_real = tf.reduce_mean(u1)
        self.d_loss_fake = tf.reduce_mean(u2)
        self.g_loss_image = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss_image=self.d_loss_real       -   self.k_t*self.d_loss_fake
        self.d_loss_label=self.d_loss_real_label -   self.l_t*self.d_loss_fake_label
        self.d_loss=self.d_loss_image+self.d_loss_label
        self.g_loss = self.g_loss_image + self.z_t*self.g_loss_label

        #Careful on z_t sign!
        self.g_loss = self.g_loss_image + self.z_t*self.g_loss_label
        #end loss

        #pretrain:
        c_grad=self.c_optimizer.compute_gradients(self.c_loss,
                                                  var_list=self.cc.train_var)
        dcc_grad=self.dcc_optimizer.compute_gradients(self.dcc_loss,var_list=self.dcc_var)

        # Calculate the gradients for the batch of data,
        # on this particular gpu tower.
        g_grad=self.g_optimizer.compute_gradients(self.g_loss,var_list=self.G_var)
        #g_grad=self.g_optimizer.compute_gradients(self.g_loss,var_list=self.G_var+self.cc.var)
        d_grad=self.d_optimizer.compute_gradients(self.d_loss,var_list=self.D_var)


        self.tower_dict['c_tower_grads'].append(c_grad)
        self.tower_dict['dcc_tower_grads'].append(dcc_grad)

        self.tower_dict['g_tower_grads'].append(g_grad)
        self.tower_dict['d_tower_grads'].append(d_grad)
        self.tower_dict['tower_g_loss_image'].append(self.g_loss_image)
        self.tower_dict['tower_d_loss_real'].append(self.d_loss_real)
        self.tower_dict['tower_g_loss_label'].append(self.g_loss_label)
        self.tower_dict['tower_d_loss_real_label'].append(self.d_loss_real_label)
        self.tower_dict['tower_d_loss_fake_label'].append(self.d_loss_fake_label)

        self.var=self.G_var+self.D_var+self.dcc_var+self.cc.var+[self.g_step]

    def build_model(self):
        self.k_t = tf.get_variable(name='k_t',initializer=0.,trainable=False)
        self.l_t = tf.get_variable(name='l_t',initializer=0.,trainable=False)
        self.z_t = tf.get_variable(name='z_t',initializer=0.,trainable=False)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))
        self.g_optimizer, self.d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.c_optimizer, self.dcc_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)


        self.dlr_optimizer = optimizer(self.g_lr)


        #g_tower_grads=[]
        #d_tower_grads=[]
        #tower_g_loss_image=[]
        #tower_d_loss_real=[]
        #tower_g_loss_label=[]
        #tower_d_loss_real_label=[]

        self.tower_dict=dict(
                    c_tower_grads=[],
                    dcc_tower_grads=[],
                    g_tower_grads=[],
                    d_tower_grads=[],
                    tower_g_loss_image=[],
                    tower_d_loss_real=[],
                    tower_g_loss_label=[],
                    tower_d_loss_real_label=[],
                    tower_d_loss_fake_label=[],
            )
        #iterate in rev order, makes self.data_loader corresp to gpu0 
        gpu_idx=0
        num_gpus=len(self.data_by_gpu)
        assert num_gpus == self.config.num_gpu or self.config.num_gpu==0

        with tf.variable_scope('tower'):
            for gpu,data_loader in self.data_by_gpu.items()[::-1]:
                gpu_idx+=1
                print('using device:',gpu)
                tower=gpu.replace('/','').replace(':','_')
                with tf.device(gpu),tf.name_scope(tower):

                    #Build num_gpu copies of graph: inputs->gradient
                    #Updates self.tower_dict
                    self.build_tower(data_loader)

                #allow future gpu to use same variables
                tf.get_variable_scope().reuse_variables()

        #Now outside gpu loop

        d_loss_real       =tf.reduce_mean(self.tower_dict['tower_d_loss_real'])
        g_loss_image      =tf.reduce_mean(self.tower_dict['tower_g_loss_image'])
        d_loss_real_label =tf.reduce_mean(self.tower_dict['tower_d_loss_real_label'])
        d_loss_fake_label =tf.reduce_mean(self.tower_dict['tower_d_loss_fake_label'])
        g_loss_label      =tf.reduce_mean(self.tower_dict['tower_g_loss_label'])

        self.balance_k = self.gamma * d_loss_real - g_loss_image
        #self.balance_l = self.gamma_label * d_loss_real_label - g_loss_label
        self.balance_l = self.gamma_label * d_loss_real_label - d_loss_fake_label
        #switch order because g minimizes
        self.balance_z = self.zeta*tf.nn.relu(self.balance_k) - tf.nn.relu(self.balance_l)


        self.measure = d_loss_real + tf.abs(self.balance_k)
        self.measure_complete = d_loss_real + d_loss_real_label + \
            tf.abs(self.balance_k)+tf.abs(self.balance_l)+tf.abs(self.balance_z)


        k_update = tf.assign(
            self.k_t, tf.clip_by_value(self.k_t + self.lambda_k*self.balance_k, 0, 1))
        l_update = tf.assign(
            self.l_t, tf.clip_by_value(self.l_t + self.lambda_l*self.balance_l, 0, 1))
        z_update = tf.assign(
            self.z_t, tf.clip_by_value(self.z_t + self.lambda_z*self.balance_z, 0, 1))


        c_grads=average_gradients(self.tower_dict['c_tower_grads'])
        dcc_grads=average_gradients(self.tower_dict['dcc_tower_grads'])

        g_grads=average_gradients(self.tower_dict['g_tower_grads'])
        d_grads=average_gradients(self.tower_dict['d_tower_grads'])

        self.c_optim =self.c_optimizer.apply_gradients(c_grads,global_step=self.cc.step)
        self.dcc_optim = self.dcc_optimizer.apply_gradients(dcc_grads)

        self.dl_optim=self.dlr_optimizer.minimize(self.d_loss_real_label,var_list=self.DL_var)
        if self.config.pretrain_labeler:
            self.pretrain_op = tf.group(self.c_optim,self.dcc_optim,self.dl_optim)
        else:
            self.pretrain_op = tf.group(self.c_optim,self.dcc_optim)

        g_optim = self.g_optimizer.apply_gradients(g_grads, global_step=self.g_step)
        d_optim = self.d_optimizer.apply_gradients(d_grads)
        with tf.control_dependencies([k_update,l_update,z_update]):
            self.train_op=tf.group(g_optim, d_optim)


        #used as global indicator
        self.step= tf.add(self.g_step,self.cc.step,name='model_step')

        ##*#* Interesting but pass this time around
        ## Track the moving averages of all trainable
        ## variables.
        #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        #variables_averages_op = variable_averages.apply(tf.trainable_variables())
        ## Group all updates to into a single
        ## train op.
        #train_op = tf.group(apply_gradient_op, variables_averages_op)



        #Label summaries
        LabelList=[self.cc.nodes,self.real_labels_list,
                   self.D_fake_labels_list,self.D_real_labels_list]
        for node,rlabel,d_fake_label,d_real_label in zip(*LabelList):
            with tf.name_scope(node.name):
                #TODO:replace with summary_stats

                ##CC summaries:
                ave_label=tf.reduce_mean(node.label)
                std_label=tf.sqrt(tf.reduce_mean(tf.square(node.label-ave_label)))
                tf.summary.scalar('ave',ave_label)
                tf.summary.scalar('std',std_label)
                tf.summary.histogram('fake_label_hist',node.label)
                tf.summary.histogram('real_label_hist',rlabel)

                ##Disc summaries
                d_flabel=tf.cast(tf.round(d_fake_label),tf.int32)
                d_rlabel=tf.cast(tf.round(d_real_label),tf.int32)
                f_acc=tf.contrib.metrics.accuracy(tf.cast(tf.round(node.label),tf.int32),d_flabel)
                r_acc=tf.contrib.metrics.accuracy(tf.cast(tf.round(rlabel),tf.int32),d_rlabel)

                ave_d_fake_label=tf.reduce_mean(d_fake_label)
                std_d_fake_label=tf.sqrt(tf.reduce_mean(tf.square(d_fake_label-ave_d_fake_label)))

                ave_d_real_label=tf.reduce_mean(d_real_label)
                std_d_real_label=tf.sqrt(tf.reduce_mean(tf.square(d_real_label-ave_d_real_label)))


                tf.summary.scalar('ave_d_fake_abs_diff',tf.reduce_mean(tf.abs(node.label-d_fake_label)))
                tf.summary.scalar('ave_d_real_abs_diff',tf.reduce_mean(tf.abs(rlabel-d_real_label)))

                tf.summary.scalar('ave_d_fake_label',ave_d_fake_label)
                tf.summary.scalar('std_d_fake_label',std_d_fake_label)
                tf.summary.scalar('ave_d_real_label',ave_d_real_label)
                tf.summary.scalar('std_d_real_label',std_d_real_label)

                tf.summary.histogram('d_fake_label',d_fake_label)
                tf.summary.histogram('d_real_label',d_real_label)

                tf.summary.scalar('real_label_ave',tf.reduce_mean(rlabel))
                tf.summary.scalar('real_label_accuracy',r_acc)
                tf.summary.scalar('fake_label_accuracy',f_acc)



        ##Summaries picked from last gpu to run
        #tf.summary.scalar('d_loss_real',self.d_loss_real)
        #tf.summary.scalar('d_loss_fake',self.d_loss_fake)
        tf.summary.scalar('losslabel/d_loss_real_label',tf.reduce_mean(self.d_loss_real_label))
        tf.summary.scalar('losslabel/d_loss_fake_label',tf.reduce_mean(self.d_loss_fake_label))
        #tf.summary.scalar('g_loss_gan',self.g_loss_gan)
        tf.summary.scalar('losslabel/g_loss_label',self.g_loss_label)

        #tf.summary.scalar('losslabel/d_real_absdif',tf.reduce_mean(self.d_absdiff_real_label))
        #tf.summary.scalar('losslabel/d_fake_absdif',tf.reduce_mean(self.d_absdiff_fake_label))
        #tf.summary.scalar('losslabel/g_absdif',tf.reduce_mean(self.g_absdiff_label))


        tf.summary.image("G", self.G),
        tf.summary.image("AE_G", self.AE_G),
        tf.summary.image("AE_x", self.AE_x),

        tf.summary.scalar("loss/d_loss", self.d_loss),
        tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
        tf.summary.scalar("loss/g_loss", self.g_loss),

        if self.config.pretrain_type=='gan':
            summary_stats('dcc/real_dcc',self.dcc_real,hist=True)
            summary_stats('dcc/fake_dcc',self.dcc_fake,hist=True)
            tf.summary.scalar('loss/dcc_real_loss',self.dcc_loss_real)
            tf.summary.scalar('loss/dcc_fake_loss',self.dcc_loss_fake)

        elif self.config.pretrain_type=='wasserstein':
            summary_stats('dcc/real_dcc_logit',self.dcc_real_logit,hist=True)
            summary_stats('dcc/fake_dcc_logit',self.dcc_fake_logit,hist=True)
            summary_stats('dcc/slopes',self.dcc_slopes,hist=True)
            tf.summary.scalar('dcc/gan_loss',self.dcc_gan_loss)
            tf.summary.scalar('dcc/grad_loss',self.dcc_grad_loss)
            #summary_stats('dcc/gan_loss',self.dcc_gan_loss)
            #summary_stats('dcc/grad_loss',self.dcc_grad_loss)

        tf.summary.scalar('loss/c_loss',self.c_loss)

        tf.summary.scalar("misc/d_lr", self.d_lr),
        tf.summary.scalar("misc/g_lr", self.g_lr),
        tf.summary.scalar("misc/eqn1", self.eqn1),
        tf.summary.scalar("misc/eqn2", self.eqn2),

        #summaries of gpu-averaged values
        tf.summary.scalar("loss/d_loss_real",d_loss_real),
        tf.summary.scalar("loss/g_loss_image", g_loss_image),
        tf.summary.scalar("balance/l", self.balance_l),
        tf.summary.scalar("balance/k", self.balance_k),
        tf.summary.scalar("balance/z", self.balance_z),
        tf.summary.scalar("misc/measure", self.measure),
        tf.summary.scalar("misc/measure_complete", self.measure_complete),
        tf.summary.scalar("misc/k_t", self.k_t),
        tf.summary.scalar("misc/l_t", self.l_t),
        tf.summary.scalar("misc/z_t", self.z_t),
        self.summary_op=tf.summary.merge_all()


    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.G, feed_dict=inputs)
        #x = self.sess.run(self.G, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x

    def big_generate(self, save=True):
        #Make 16x16 image
        nrow=16

        all_images=None
        for i in range(nrow):
            idx,z_fixed = self.sess.run([self.g_step,self.z_fd])
            feed_fixed_z={self.z_fd[k]:val for k,val in z_fixed.items()}
            images = self.sess.run(self.G, feed_dict=feed_fixed_z)

            if all_images is None:
                all_images = images
            else:
                all_images = np.concatenate([all_images, images])

        if save:
            path = os.path.join(self.model_dir, '{}_BigG.png'.format(idx))
            save_image(all_images, path,nrow=nrow)
            print("[*] BigSamples saved: {}".format(path))

        return all_images

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        #inputs=data_fixed in self.train()
        items = {
            'real': inputs['x'],#x_fixed
            'fake': x_fake,#generated self.G
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_x, {self.x: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = batch_size/2

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)


    def get_data_from_loader(self,key=None):
        data = self.sess.run(self.data_loader)
        if self.data_format == 'NCHW':
            data['x'] = data['x'].transpose([0, 2, 3, 1])
        if key:
            return data[key]
        else:
            return data


    def intervention(self,inputs=None,root_path=None):

        #root_path = "./"#self.model_dir
        root_path=root_path or self.model_dir
        #ones=np.ones([self.batch_size,1])

        ##Batch_size is usually 16
        #haven't coded anything else yet
        assert self.batch_size==16

        if inputs==None:
            z_sample = self.sess.run(self.z_fd)#has z_gen in it
        else:
            z_sample=inputs

        #make 8x8 image 2 columns(?rows) at a time
        for node in self.cc.nodes:
            all_images=None

            stats=self.label_stats.loc[node.name]
            interp_label=np.linspace(stats['min_label'],stats['max_label'],8).reshape([8,1])
            interp_logit=np.linspace(stats['min_logit'],stats['max_logit'],8).reshape([8,1])
            setval=np.tile(interp_label,[2,1])
            setlogit=np.tile(interp_logit,[2,1])

            for a in range(4):
                #print(z_sample)
                z_fixed  = {key:np.tile(val[a::8],[1,1,8]).reshape(16,-1) for key,val in z_sample.items()}
                feed_fixed_z={self.z_fd[k]:val for k,val in z_fixed.items()}

                fd=feed_fixed_z
                #fd.update({node.label:setval})#LABELS
                fd.update({node.label_logit:setlogit})#LOGITS

                images,step=self.sess.run([self.G,self.step],fd)

                if all_images is None:
                    all_images = images
                else:
                    all_images = np.concatenate([all_images, images])

            intv_path='{}/{}_G_itv_{}.png'.format(root_path,step,node.name)
            print("[*] Samples saved: {}".format(intv_path))
            save_image(all_images,intv_path,nrow=8)
            #save_image(images,'{}/G_itv_{}{}.png'.format(root_path,node.name,step),nrow=4)



