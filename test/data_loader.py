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
    def __init__(config):
        self.config=config
        self.root=self.config.data_path
        self.scale_size=config.input_scale_size
        self.data_format=config.data_format
        self.split=config.split
        self.do_shuffle=config.do_shuffle
        self.num_worker=config.num_worker
        self.is_crop=config.is_crop
        self.is_grayscale=config.grayscale

        attributes = pd.read_csv(config.attr_file,delim_whitespace=True) #+-1
        self.attr= 0.5*(attributes+1)

        label_means=attributes.mean()# attributes is 0,1
        p=label_means.values

        self.image_dir=os.path.join(self.root,'images')
        self.filenames=[os.path.join(self.image_dir,j) for j in self.attr.index]

        self.label_names=self.attr.columns
        self.num_examples_per_epoch=len(self.filenames)
        self.min_fraction_of_examples_in_queue=0.001#go faster during debug
        #self.min_fraction_of_examples_in_queue=0.01#have enough to do shuffling
        self.min_queue_examples=int(self.num_examples_per_epoch*self.min_fraction_of_examples_in_queue)


    def get_label_queue(self,batch_size):
        tf_labels = tf.convert_to_tensor(self.attr.values, dtype=tf.uint8)#0,1

        with tf.name_scope('label_queue'):
            uint_label=tf.train.slice_input_producer([tf_labels])[0]

        #All labels, not just those in graph
        dict_data={sl:tl for sl,tl in
                   zip(self.label_names,tf.split(label,len(self.label_names)))}

        #label=tf.to_int32(str_label)#keep as float?

        print ('Filling label queue with %d labels before starting to train. '
            'I don\'t know how long this will take' % min_queue_examples)
        #I think there are 3 other threads used elsewhere?
        #Forget why I think that
        num_preprocess_threads = max(num_worker-3,1)

        data_batch = tf.train.shuffle_batch(
                dict_data,
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                #allow_smaller_final_batch=True)
                )

        return data_batch

    def get_data_loader(self):
        image_files = tf.convert_to_tensor(self.filenames, dtype=tf.string)
        tf_labels = tf.convert_to_tensor(self.attr.values, dtype=tf.uint8)

        with tf.name_scope('filename_queue'):
            #must be list
            str_queue=tf.train.slice_input_producer([image_files,tf_labels])
        img_filename, uint_label= str_queue

        img_contents=tf.read_file(img_filename)
        image = tf.image.decode_jpeg(img_contents, channels=3)


        image=tf.cast(image,dtype=tf.float32)
        if is_crop:#use dcgan cropping
            #dcgan center-crops input to 108x108, outputs 64x64 #centrally crops it
            #We emulate that here
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

        if data_format == 'NCHW':
            image = tf.transpose(image, [2, 0, 1])#3D
            #image = tf.transpose(image, [0, 3, 1, 2])#4D
        elif data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(data_format))


        #inputs to dictionary:
        #Creates a dictionary  {'Male',male_tensor, 'Young',young_tensor} etc..
        #dict_data={sl:tf.reshape(tl,[1,1]) for sl,tl in
        dict_data={sl:tl for sl,tl in
                   zip(label_names,tf.split(label,len(label_names)))}
        assert not 'x' in dict_data.keys()
        dict_data['x']=image

        #label=tf.to_int32(str_label)#keep as float?

        print ('Filling queue with %d Celeb images before starting to train. '
            'I don\'t know how long this will take' % min_queue_examples)
        #I think there are 3 other threads used elsewhere
        num_preprocess_threads = max(num_worker-3,1)
        #image_batch, real_label_batch = tf.train.shuffle_batch(
        data_batch = tf.train.shuffle_batch(
                dict_data,
                batch_size=batch_size*num_devices,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size*num_devices,
                min_after_dequeue=min_queue_examples,
                #allow_smaller_final_batch=True)
                )
        # Display the training images in the visualizer.
        #TODO: doesn't quite work because depends on whether is NCHW.Better elsewhere
        #tf.summary.image('real_images', data_batch['x'])
        return data_batch

