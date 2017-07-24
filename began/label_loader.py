import os
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import tensorflow as tf

from IPython.core import debugger
debug = debugger.Pdb().set_trace


def logodds(p):
    return np.log(p/(1.-p))

def get_label_loader(config, root, batch_size,do_shuffle=True,num_worker=4):
    '''This loads the image and the labels through a tensorflow queue.
    All of the labels are loaded regardless of what is specified in graph,
    because this model is gpu throttled anyway so there shouldn't be any
    overhead

    For multiple gpu, the strategy here is to have 1 queue with 2xbatch_size
    then use tf.split within trainer.train()
    '''

    dataset_name = os.path.basename(root)
    attr_file= glob("{}/*.{}".format(root, 'txt'))[0]
    setattr(config,'attr_file',attr_file)
    attributes = pd.read_csv(attr_file,delim_whitespace=True) #+-1
    attributes = 0.5*(attributes+1)
    #image_dir=os.path.join(root,'images')
    #filenames=[os.path.join(image_dir,j) for j in attributes.index]

    label_names=attributes.columns
    #num_examples_per_epoch=len(filenames)
    num_examples_per_epoch=len(attributes)
    min_fraction_of_examples_in_queue=0.001#have enough to do shuffling
    min_queue_examples=int(num_examples_per_epoch*min_fraction_of_examples_in_queue)

    #image_files = tf.convert_to_tensor(filenames, dtype=tf.string)
    tf_labels = tf.convert_to_tensor(attributes.values, dtype=tf.uint8)

    with tf.name_scope('label_queue'):
        #must be list
        str_queue=tf.train.slice_input_producer([tf_labels])

    #img_filename, uint_label= str_queue
    uint_label= str_queue[0]
    #img_contents=tf.read_file(img_filename)
    #image = tf.image.decode_jpeg(img_contents, channels=3)

    label_means=attributes.mean()# attributes is 0,1
    p=label_means.values

    label=tf.to_float(uint_label)

    if config.noisy_labels:
        #(fixed)Original Murat Noise model
        N=tf.random_uniform([len(p)],-.25,.25,)
        def label_mapper(p,label,N):
            '''
            P \in {1-p, p}
            L \in {-1, +1}
            N \in [-1/4,1/4]
            '''
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

    else:
        #No noise model!
        ones=np.ones_like(label_means)
        label_stats=pd.DataFrame({
            'mean':label_means,
            'min_label':0.*ones,
            'max_label':1.*ones,
            'min_logit':-5.*ones,
            'max_logit':5*ones,
        })

    dict_data={sl:tl for sl,tl in
               zip(label_names,tf.split(label,len(label_names)))}

    #label=tf.to_int32(str_label)#keep as float?

    print ('Filling label queue with %d labels before starting to train. '
        'I don\'t know how long this will take' % min_queue_examples)
    #I think there are 3 other threads used elsewhere
    num_preprocess_threads = max(num_worker-3,1)

    data_batch = tf.train.shuffle_batch(
            dict_data,
            batch_size=batch_size*num_devices,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size*num_devices,
            min_after_dequeue=min_queue_examples,
            #allow_smaller_final_batch=True)
            )

    return data_batch, label_stats



