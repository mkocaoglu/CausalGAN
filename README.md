# Causal(BE)GAN in Tensorflow

# (test comment)

<> (Tensorflow implementation of [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717).)

Authors' Tensorflow implementation of [CausalGAN: Learning Implicit Causal Models with Adversarial Training]

![alt text](./assets/314393_began_Bald_topdo1_botcond1.pdf)
### top: samples from do(Bald=1); bottom: samples from cond(Bald=1)
![alt text](./assets/314393_began_Mustache_topdo1_botcond1.pdf)
### top: samples from do(Mustache=1); bottom: samples from cond(Mustache=1)

## Requirements
- Python 2.7
- [Pillow](https://pillow.readthedocs.io/en/4.0.x/)
- [tqdm](https://github.com/tqdm/tqdm)
- [requests](https://github.com/kennethreitz/requests) (Only used for downloading CelebA dataset)
- [TensorFlow 1.1.0](https://github.com/tensorflow/tensorflow)

## Usage

First download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets with:

    $ apt-get install p7zip-full # ubuntu
    $ brew install p7zip # Mac
    $ python download.py


The Causal(BE)GAN code factorizes into two components, which can be trained or loaded independently: the causal_controller module specifies the model which learns a causal generative model over labels, and the causal_dcgan or causal_began modules learn a GAN over images given those labels. We denote training the causal controller over labels as "pretraining" (--is_pretrain=True), and training a GAN over images given labels as "training" (--is_train=True)

To train an implicit causal model over labels:

    $ python main.py --causal_model $model_key --is_pretrain True

where model_key specifies the causal_graph for the model by matching a key in the causal_graphs dictionary in causal_graph.py. To additionally train an image model from scratch:

    $ python main.py --causal_model $model_key --is_pretrain True --model_type began --is_train True

or alternatively one can load an existing causal_controller module when beginning to train (the more intensive) image training:

    $ echo CC-MODEL_PATH='./logs/celebA_0810_191625_0.145tvd_bcg/controller/checkpoints/CC-Model-20000'
    $ python main.py --causal_model $model_key --pt_load_path $CC-MODEL_PATH --model_type began --is_train True 

Instead of loading the model piecewise, once image training has been run once, the entire joint model can be loaded more simply by specifying the model directory:

    $ python main.py --causal_model $model_key --load_path ./logs/celebA_0815_170635 --model_type began --is_train True 

Tensorboard visualization of the most recently created model is simply (as long as port 6006 is free):

    $ python tboard.py


To interact with an already trained model I recommend the following procedure:

    ipython
    In [1]: %run main --causal_model 'my_model_key' --load_path './logs/celebA_0815_170635 --model_type 'began'

For example to sample N=22 interventional images from do(Smiling=1) (as long as your causal graph includes a "Smiling" node:

    In [2]: sess.run(model.G,{cc.Smiling.label:np.ones((22,1), trainer.batch_size:22})

Conditional sampling is most efficiently done through 2 session calls: the first to cc.sample_label to get, and the second feeds that sampled label to get an image. See trainer.causal_sampling for a more extensive example. Note that is also possible combine conditioning and intervention during sampling.

    In [3]: lab_samples=cc.sample_label(sess,do_dict={'Bald':1}, cond_dict={'Mustache':1},N=22)

will sample all labels from the joint distribution conditioned on Mustache=1 and do(Bald=1). These label samples can be turned into image samples as follows:

    In [4]: feed_dict={cc.label_dict[k]:v for k,v in lab_samples.iteritems()}
    In [5]: feed_dict[trainer.batch_size]=22
    In [6]: images=sess.run(trainer.G,feed_dict)


### Configuration
Since this really controls training of 3 different models (causal_controller, CausalGAN, and CausalBEGAN), many configuration options are available. To make things managable, there are 4 files corresponding to configurations specific to different parts of the model. Not all configuration combinations are tested. Default parameters are gauranteed to work.

configurations:
./config.py  :  generic data and scheduling
./causal_controller/config  :  specific to CausalController
./causal_dcgan/config  :  specific to CausalGAN
./causal_began/config  :  specific to CausalBEGAN

For convenience, the configurations used saved in 4 .json files in the model directory for future reference.


## Results

### Generator output (64x64) with `gamma=0.5` after 300k steps

<> ![all_G_z0_64x64](./assets/all_G_z0_64x64.png)


### Generator output (128x128) with `gamma=0.5` after 200k steps

<> ![all_G_z0_64x64](./assets/all_G_z0_128x128.png)


### Interpolation of Generator output (64x64) with `gamma=0.5` after 300k steps

<> ![interp_G0_64x64](./assets/interp_G0_64x64.png)


### Interpolation of Generator output (128x128) with `gamma=0.5` after 200k steps

<> ![interp_G0_128x128](./assets/interp_G0_128x128.png)

    
### Interpolation of Discriminator output of real images
<---   
![alt tag](./assets/AE_batch.png)   
![alt tag](./assets/interp_1.png)   
![alt tag](./assets/interp_2.png)   
![alt tag](./assets/interp_3.png)   
![alt tag](./assets/interp_4.png)   
![alt tag](./assets/interp_5.png)   
![alt tag](./assets/interp_6.png)   
![alt tag](./assets/interp_7.png)   
![alt tag](./assets/interp_8.png)   
![alt tag](./assets/interp_9.png)   
![alt tag](./assets/interp_10.png)
-->

## Related works
<---
- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)initial fork
- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
-->

## Authors

Christopher Snyder / [@22csnyder](http://22csnyder.github.io)
Murat Kocaoglu / [@mkocaoglu](http://mkocaoglu.github.io)
