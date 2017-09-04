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

class DataLoader(object):
    '''This loads the image and the labels through a tensorflow queue.
    All of the labels are loaded regardless of what is specified in graph,
    because this model is gpu throttled anyway so there shouldn't be any
    overhead

    For multiple gpu, the strategy here is to have 1 queue with 2xbatch_size
    then use tf.split within trainer.train()
    '''
    def __init__(self,label_names,config):
        self.label_names=label_names
        self.config=config
        self.scale_size=config.input_scale_size
        #self.data_format=config.data_format
        self.split=config.split
        self.do_shuffle=config.do_shuffle
        self.num_worker=config.num_worker
        self.is_crop=config.is_crop
        self.is_grayscale=config.grayscale

        attr_file= glob("{}/*.{}".format(config.data_path, 'txt'))[0]
        setattr(config,'attr_file',attr_file)

        attributes = pd.read_csv(config.attr_file,delim_whitespace=True) #+-1
        #Store all labels for reference
        self.all_attr= 0.5*(attributes+1)# attributes is {0,1}
        self.all_label_means=self.all_attr.mean()

        #but only return desired labels in queues
        self.attr=self.all_attr[label_names]
        self.label_means=self.attr.mean()# attributes is 0,1

        self.image_dir=os.path.join(config.data_path,'images')
        self.filenames=[os.path.join(self.image_dir,j) for j in self.attr.index]

        self.num_examples_per_epoch=len(self.filenames)
        self.min_fraction_of_examples_in_queue=0.001#go faster during debug
        #self.min_fraction_of_examples_in_queue=0.01
        self.min_queue_examples=int(self.num_examples_per_epoch*self.min_fraction_of_examples_in_queue)


    def get_label_queue(self,batch_size):
        tf_labels = tf.convert_to_tensor(self.attr.values, dtype=tf.uint8)#0,1

        with tf.name_scope('label_queue'):
            uint_label=tf.train.slice_input_producer([tf_labels])[0]
        label=tf.to_float(uint_label)

        #All labels, not just those in causal_model
        dict_data={sl:tl for sl,tl in
                   zip(self.label_names,tf.split(label,len(self.label_names)))}


        num_preprocess_threads = max(self.num_worker-3,1)

        data_batch = tf.train.shuffle_batch(
                dict_data,
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=self.min_queue_examples + 3 * batch_size,
                min_after_dequeue=self.min_queue_examples,
                )

        return data_batch

    def get_data_queue(self,batch_size):
        image_files = tf.convert_to_tensor(self.filenames, dtype=tf.string)
        tf_labels = tf.convert_to_tensor(self.attr.values, dtype=tf.uint8)

        with tf.name_scope('filename_queue'):
            #must be list
            str_queue=tf.train.slice_input_producer([image_files,tf_labels])
        img_filename, uint_label= str_queue

        img_contents=tf.read_file(img_filename)
        image = tf.image.decode_jpeg(img_contents, channels=3)

        image=tf.cast(image,dtype=tf.float32)
        if self.config.is_crop:#use dcgan cropping
            #dcgan center-crops input to 108x108, outputs 64x64 #centrally crops it #We emulate that here
            image=tf.image.resize_image_with_crop_or_pad(image,108,108)
            #image=tf.image.resize_bilinear(image,[scale_size,scale_size])#must be 4D

            resize_method=getattr(tf.image.ResizeMethod,self.config.resize_method)
            image=tf.image.resize_images(image,[self.scale_size,self.scale_size],
                    method=resize_method)
            #Some dataset enlargement. Might as well.
            image=tf.image.random_flip_left_right(image)

            ##carpedm-began crops to 128x128 starting at (50,25), then resizes to 64x64
            #image=tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
            #image=tf.image.resize_nearest_neighbor(image, [scale_size, scale_size])

            tf.summary.image('real_image',tf.expand_dims(image,0))



        label=tf.to_float(uint_label)
        #Creates a dictionary  {'Male',male_tensor, 'Young',young_tensor} etc..
        dict_data={sl:tl for sl,tl in
                   zip(self.label_names,tf.split(label,len(self.label_names)))}
        assert not 'x' in dict_data.keys()#don't have a label named "x"
        dict_data['x']=image

        print ('Filling queue with %d Celeb images before starting to train. '
            'I don\'t know how long this will take' % self.min_queue_examples)
        num_preprocess_threads = max(self.num_worker,1)

        data_batch = tf.train.shuffle_batch(
                dict_data,
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=self.min_queue_examples + 3 * batch_size,
                min_after_dequeue=self.min_queue_examples,
                )
        return data_batch

