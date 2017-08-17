# Causal(BE)GAN in Tensorflow

# (test comment)

Synthetic Data Figures
<> (Tensorflow implementation of [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717).)

Authors' Tensorflow implementation Synthetic portion of [CausalGAN: Learning Implicit Causal Models with Adversarial Training]

<>some results files

## Setup.

If not already set, make sure that run_datasets.sh is an executable by running
    $ chmod +x run_datasets.sh

## Usage

A single run of main.py trains as many GANs as are in models.py (presently 6) for a single --data_type. This author can fit 3 such runs on a single gpu and conveniently there are 3 datasets considered.

    $ CUDA_VISIBLE_DEVICES='0' python main.py --data_type=linear

Again the tboard.py utility is available to view the most recent model summaries.

    $ python tboard.py

Recovering statistics means averaging over many runs. Mass usage follows the script run_datasets.sh. This bash script will train all GAN models on each of 3 datasets 30 times per dataset. The following will train 2(calls) x 30(loop/call) x 3(datasets/loop) x 6(gan models/dataset)=1080(gan models)


    $ (open first terminal)
    $ CUDA_VISIBLE_DEVICES='0' ./run_datasets.sh
    $ (open second terminal)
    $ CUDA_VISIBLE_DEVICES='1' ./run_datasets.sh


## Collecting Statistics


## Results


## Authors

Christopher Snyder / [@22csnyder](http://22csnyder.github.io)
Murat Kocaoglu / [@mkocaoglu](http://mkocaoglu.github.io)
