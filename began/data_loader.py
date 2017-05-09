import os
import pandas as pd
from PIL import Image
from glob import glob
import tensorflow as tf

def get_loader(config, root, batch_size, scale_size, data_format,
               split=None,do_shuffle=True,num_worker=4,is_crop=False,
               is_grayscale=False, seed=None):
    '''This loads the image and the labels through a tensorflow queue.
    All of the labels are loaded regardless of what is specified in graph,
    because this model is gpu throttled anyway so there shouldn't be any
    overhead

    For multiple gpu, the strategy here is to have 1 queue with 2xbatch_size
    then use tf.split within trainer.train()
    '''
    num_devices=max(1,config.num_gpu)#that are used in backprop

    dataset_name = os.path.basename(root)
    #if dataset_name in ['CelebA'] and split:

    #TODO datasplits
    #if split:
    #    root = os.path.join(root, 'splits', split)

    #Each dataset should have 1 .txt file that has labels and image filenames
    #that should be readable as a pandas dataframe
    #and the indices should be filenames
    #and the columns should be labels
    attr_file= glob("{}/*.{}".format(root, 'txt'))[0]
    attributes = pd.read_csv(attr_file,delim_whitespace=True)

    #shuffle here especilly if multigpu
    #if do_shuffle or config.num_gpu>1:
    #    attributes=attributes.sample(frac=1)#shuffle rows

    image_dir=os.path.join(root,'images')
    filenames=[os.path.join(image_dir,j) for j in attributes.index]
    real_labels= (attributes+1)*0.5
    label_names=attributes.columns

    num_examples_per_epoch=len(filenames)

    #-----------
    #DEBUG
    min_fraction_of_examples_in_queue=0.001#have enough to do shuffling
    #min_fraction_of_examples_in_queue=0.1#have enough to do shuffling
    #-----------


    min_queue_examples=int(num_examples_per_epoch*min_fraction_of_examples_in_queue)

    image_files = tf.convert_to_tensor(filenames, dtype=tf.string)
    tf_labels = tf.convert_to_tensor(real_labels.values, dtype=tf.uint8)

    with tf.name_scope('filename_queue'):
        #must be list
        str_queue=tf.train.slice_input_producer([image_files,tf_labels])

    img_filename, uint_label= str_queue
    img_contents=tf.read_file(img_filename)
    #reader=tf.WholeFileReader()
    image = tf.image.decode_jpeg(img_contents, channels=3)

    ##This might be needed with began cropping
    #with Image.open(filenames[0]) as img:
    #    w, h = img.size
    #    shape = [h, w, 3]
    #if is_grayscale:
    #    image = tf.image.rgb_to_grayscale(image)
    #image.set_shape(shape)

    image=tf.cast(image,dtype=tf.float32)
    if is_crop:#use dcgan cropping
        #dcgan center-crops input to 108x108, outputs 64x64 #centrally crops it
        image=tf.image.resize_image_with_crop_or_pad(image,108,108)
        #image=tf.image.resize_bilinear(image,[scale_size,scale_size])#must be 4D
        image=tf.image.resize_images(image,[scale_size,scale_size],
                method=tf.image.ResizeMethod.BILINEAR)

        ##carpedm-began crops to 128x128 starting at (50,25), then resizes to 64x64
        #image=tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
        #image=tf.image.resize_nearest_neighbor(image, [scale_size, scale_size])

    if data_format == 'NCHW':
        image = tf.transpose(image, [2, 0, 1])#3D
        #image = tf.transpose(image, [0, 3, 1, 2])#4D
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    label=tf.to_float(uint_label)
    if config.noisy_labels:
        label_means=attributes.mean()
        p=label_means.values

        ##Original Murat Noise model
        #noise=tf.random_uniform([len(p)],-.25,.25,)
        #P = label+p-2*label*p #p or (1-p):for label=0,1
        #L = 1-2*label# +1 or -1   :for label=0,1
        #label=0.5 + L*.25*P + P*noise

        #simple [0,1/2] or [1/2,1]
        #noise=tf.random_uniform([len(p)],0,0.5)
        #neg_noise=noise
        #pos_noise=noise+0.5
        #label= (1-label)*neg_noise + label*pos_noise

        #simple [0.2,1/2] or [1/2,0.8]
        noise=tf.random_uniform([len(p)],0.2,0.5)
        neg_noise=noise
        pos_noise=1.0-noise
        label= (1-label)*neg_noise + label*pos_noise

        #neg_noise=noise*p#U[0,p]
        #pos_noise=1-noise*(1-p)
        #label= label*


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
