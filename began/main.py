import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from causal_graph import get_causal_graph
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

from IPython.core import debugger
debug = debugger.Pdb().set_trace


'''
Some notes on each version of the model:


Previous model worked really well.
Everything except for bald, wearing eyeglasses, and mustache was captured
really pretty well.

murat wants new causal graph:

#Old
    #big_causal_graph=[
    #        ['Young',[]],
    #        ['Male',[]],
    #        ['Eyeglasses',['Young']],
    #        ['Bald',            ['Male','Young']],
    #        ['Mustache',        ['Male','Young']],
    #        ['Smiling',         ['Male','Young']],
    #        ['Wearing_Lipstick',['Male','Young']],
    #        ['Mouth_Slightly_Open',['Smiling']],
    #        ['Narrow_Eyes',        ['Smiling']],
    #    ]


#New
    new_big_causal_graph=[
            ['Young',[]],
            ['Male',[]],
            ['Eyeglasses',['Young']],
            ['Bald',            ['Male','Young']],
            ['Mustache',        ['Male','Young']],
            ['Smiling',         ['Male','Young']],
            ['Wearing_Lipstick',['Male','Young']],
            ['Mouth_Slightly_Open',['Young','Smiling']],
            ['Narrow_Eyes',        ['Male','Young','Smiling']],
        ]

Also let's go back to old noise model.
Disadvantage is have to control range of intervention for each label,
as it depends on the mean


BUG!!!!!!
how long was this here for. In old noise model:
        label_means=attributes.mean()
        p=label_means.values
    used labes in -1 to +1 to calculate mean
soln: just don't have any -1 +1 male

    attributes = 0.5*(attributes+1)
    #real_labels= (attributes+1)*0.5

    tf_labels = tf.convert_to_tensor(attributes.values, dtype=tf.uint8)
    #tf_labels = tf.convert_to_tensor(real_labels.values, dtype=tf.uint8)



Added this extra flip left/right noise because cycling through data so much
        image=tf.image.random_flip_left_right(image)


New noise method:
    label=tf.to_float(uint_label)
    if config.noisy_labels:
        #(fixed)Original Murat Noise model
        N=tf.random_uniform([len(p)],-.25,.25,)
        def label_mapper(p,label,N):
            #P \in {1-p, p}
            #L \in {-1, +1}
            #N \in [-1/4,1/4]
            P = 1-(label+p-2*label*p)#l=0 -> (1-p) or l=1 -> p
            L = 2*label-1# -1 or +1   :for label=0,1
            return 0.5+ .25*P*L + P*N

        min_label= label_mapper(label_means,0,-.25)
        max_label= label_mapper(label_means,+1,0.25)
        min_logit= logodds(min_label)
        max_logit= logodds(max_label)

        #needed for visualization range
        label_stats=pd.DataFrame({
            'mean':label_means,
            'min_label':min_label,
            'max_label':max_label,
            'max_logit':max_logit,
            'min_logit':min_logit,
             })
        label=label_mapper(label_means.values,label,N)





visualization:

        stats=self.config.label_stats.loc[node.name]
        interp_label=np.linspace(stats['min_label'],stats['max_label'],8).reshape([8,1])
        interp_logit=np.linspace(stats['min_logit'],stats['max_logit'],8).reshape([8,1])
        #interp_label=np.linspace(0.2,0.8,8).reshape([8,1])
        #interp_logit=np.linspace(-2*np.log(2),2*np.log(2),8).reshape([8,1])


Changed noisylabels to default True
    train_arg.add_argument('--noisy_labels', type=str2bool, default=True)
    #train_arg.add_argument('--noisy_labels', type=str2bool, default=False)

added
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

Changed a number of default config options to be more reasonable:
    new cmd to run:

%run main.py --dataset=celebA  --is_train=True --causal_model=new_big_causal_graph

'''

def get_trainer(config):
    print 'tf: resetting default graph!'
    tf.reset_default_graph()

    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        #setattr(config, 'batch_size', 64)
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    data_loader, label_stats= get_loader(config,
            data_path,config.batch_size,config.input_scale_size,
            config.data_format,config.split,
            do_shuffle,config.num_worker,config.is_crop)

    config.graph=get_causal_graph(config.causal_model)

    print 'Config:'
    print config

    trainer = Trainer(config, data_loader, label_stats)
    return trainer


def main(trainer,config):
    if config.dry_run:
        #debug()
        return


    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")

        trainer.intervention()
        #trainer.test()

def get_model(config=None):
    if not None:
        config, unparsed = get_config()
    return get_trainer(config)

if __name__ == "__main__":
    config, unparsed = get_config()
    trainer=get_trainer(config)
    main(trainer,config)
    ##debug mode: below is main() code

#    prepare_dirs_and_logger(config)
#
#    rng = np.random.RandomState(config.random_seed)
#    tf.set_random_seed(config.random_seed)
#
#    if config.is_train:
#        data_path = config.data_path
#        batch_size = config.batch_size
#        do_shuffle = True
#    else:
#        setattr(config, 'batch_size', 64)
#        if config.test_data_path is None:
#            data_path = config.data_path
#        else:
#            data_path = config.test_data_path
#        batch_size = config.sample_per_image
#        do_shuffle = False
#
#    data_loader = get_loader(config,
#            data_path,config.batch_size,config.input_scale_size,
#            config.data_format,config.split,
#            do_shuffle,config.num_worker,config.is_crop)
#
#    config.graph=get_causal_graph()
#
#    trainer = Trainer(config, data_loader)
#
#    if config.dry_run:
#        return
#
#    if config.is_train:
#        save_config(config)
#        trainer.train()
#    else:
#        if not config.load_path:
#            raise Exception("[!] You should specify `load_path` to load a pretrained model")
#        trainer.test()
#
