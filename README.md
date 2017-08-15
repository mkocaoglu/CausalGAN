# Causal(BE)GAN in tensorflow

<>Tensorflow implementation of [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717).
Authors' Tensorflow implementation of [CausalGAN: Learning Implicit Causal Models with Adversarial Training]

<> ![alt tag](./assets/model.png)


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



    $ python main.py --dataset=YOUR_DATASET_NAME --use_gpu=True

To test a model (use your `load_path`):

    $ python main.py --dataset=CelebA --load_path=CelebA_0405_124806 --use_gpu=True --is_train=False --split valid


## Configuration
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
- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [DiscoGAN-pytorch](https://github.com/carpedm20/DiscoGAN-pytorch)
- [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow)
-->

## Authors

Christopher Snyder / [@22csnyder](http://22csnyder.github.io)
Murat Kocaoglu / [@mkocaoglu](http://mkocaoglu.github.io)
