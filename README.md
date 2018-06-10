# FloydHub Clone of carpedm20/DCGAN-tensorflow
This is a slightly modified version of carpedm20's code. It is meant for use with the <a href="https://floydhub.com" target="_blank">FloydHub</a> deep leaning platform. If you wish to run a DCGAN on your own hardware or cloud infrasturcture other than FloydHub, I would suggest looking at the <a href="https://github.com/carpedm20/DCGAN-tensorflow" target="_blank">original repository</a>.

## Useage

#### Getting your own training data

Although this repository contains code for downloading several standard datasets, I haven't tested this with FloydHub. I would recommend either using one of the datasets already avalaible on Floydhub or uploading your own.

If using your own dataset, ensure that 
* all images are the same dimensions and colour mode e.g. all 512x512px and either all greyscale or all RGB. A mixture of colour modes could lead the model to fail half-way through training.
* all images are placed in sub-directories in your dataset all starting with 'train' e.g. train or train01 train01 train03 etc.

#### First run
Once your Floydhub dataset and project are initilised, run a command like the following from the directory where you've initilised your project.

##### Explanation of command when using a FloydHub dataset (pre-existing or your own) 

    $ floyd run \
        --gpu (It is recommended to use GPU for deep learning over images.
               Ommiting this switch will result in slower training using CPU.) \
        --env tensorflow-1.8 (The original requirments are Tensorflow  0.12.1.
                              However, it works with newer versions.) \
        --data <PATH TO FLOYDHUB DATASET>:<DIRECTORY NAME FOR MOUNTING DATA> \
         "python main.py \
        --dataset=<DIRECTORY NAME FOR MOUNTING DATA> \
        --data_dir=/ (this needs to remain the root diretory as this is where 
                      FloydHub mounts datasets') \
        --batch_size= <THE NUMBER OF TRAINGING EXAMPLES TO PROCESS AT ONCE. 
                       THIS MUST BE A SQUARE NUMBER. e.g. 16, 25, 36, 49
                       GENERALLY THIS WILL SMALLER IF YOUT IMAGES ARE LARGER.
                       DEFAULT: 64> \
        --save_frequency=<HOW FREQUENTLY TO SAVE SAMPLE IMAGES DURING TRAINING.
                          DEFAULT: 100> \
        --generate_test_images=<HOW MANY TEST IMAGES TO GENERATE AT THE END OF 
                                TRAINING. DEFAULT: 50> \
        --input_height=<THE HEIGHT OF THE IMAGES IN YOUR TRAINING DATASET. 
                        DEFAULT 256> \
        --input_width=<THE WIDTH OF THE IMAGES IN YOUR TRAINING DATASET
                       DEFAULT: input_width> \
        --output_height=<THE HEIGHT OF THE IMAGES YOU WANT THE GAN TO OUTPUT
                         DEFAULT: 256> \
        --output_width=<THE WIDTH OF THE IMAGES YOU WANT THE GAN TO OUTPUT
                        DEFAULT: output_height> \
        --epoch=<NUMBER OF TIMES THE PROGRAM WILL CYCLE THOUGH YOUR TRAINING DATA 
                 BEFORE COMPLETING TRAINING. DEFAULT: 25> \
        --sample_dir=/output (this needs to remain '/output' as this is the only 
                      location FLoydhHub persists files once a command/job has 
                      been succesfully executed.) \
        --train"
        
##### Example of command when using a FloydHub dataset (pre-existing or your own) 

    $ floyd run \
        --gpu \
        --env tensorflow-1.8 \
        --data milkgan/datasets/selfie-100k-512/1:selfie-100k-512 \
        "python main.py \
        --dataset=selfie-100k-512 \
        --data_dir=/ \
        --batch_size=25 \
        --save_frequency=50 \
        --generate_test_images=25 \
        --input_height=512 \
        --input_width=512 \
        --output_height=512 \
        --output_width=512 \
        --epoch=5 \
        --sample_dir=/output \
        --train" 

#### Further runs
The code has been modified to save the model's checkpoint directory in FloydHub's output. This means that you can train your model incrementally. 

In order to do this, you will need to copy the **checkpoint** directory from the output of the previous run of your project and upload this to Floydhub as a separate dataset. In the example below the path of the checkpoint dataset is milkgan/datasets/self-dcgan-checkpoint-5-epochs/1.

##### Example of command when using FloydHub dataset and checkpoints from a previous job
    floyd run \
        --gpu \
        --env tensorflow-1.8 \
        --data milkgan/datasets/selfie-100k-512/1:selfie-100k-512 \
        --data milkgan/datasets/self-dcgan-checkpoint-5-epochs/1:checkpoint \
        "python main.py \
        --dataset=selfie-100k-512 \
        --data_dir=/ \
        --checkpoint_dir=/checkpoint \
        --batch_size=25 \
        --save_frequency=100 \
        --generate_test_images=25 \
        --input_height=512 \
        --input_width=512 \
        --output_height=512 \
        --output_width=512 \
        --epoch=5 \
        --sample_dir=/output \
        --train"

# Original Readme

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which are  stabilized Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

![alt tag](DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper.*


## Online Demo

[<img src="https://raw.githubusercontent.com/carpedm20/blog/master/content/images/face.png">](http://carpedm20.github.io/faces/)

[link](http://carpedm20.github.io/faces/)


## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) [moviepy](https://github.com/Zulko/moviepy) (for visualization)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset


## Usage (not guareneed to work)

First, download dataset with:

    $ python download.py mnist celebA

To train a model with downloaded dataset:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --train
    $ python main.py --dataset celebA --input_height=108 --train --crop

To test with an existing model:

    $ python main.py --dataset mnist --input_height=28 --output_height=28
    $ python main.py --dataset celebA --input_height=108 --crop

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train

If your dataset is located in a different root directory:

    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR --train
    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR
    $ # example
    $ python main.py --dataset=eyes --data_dir ../datasets/ --input_fname_pattern="*_cropped.png" --train
    

## Results

![result](assets/training.gif)

### celebA

After 6th epoch:

![result3](assets/result_16_01_04_.png)

After 10th epoch:

![result4](assets/test_2016-01-27%2015:08:54.png)

### Asian face dataset

![custom_result1](web/img/change5.png)

![custom_result1](web/img/change2.png)

![custom_result2](web/img/change4.png)

### MNIST

MNIST codes are written by [@PhoenixDai](https://github.com/PhoenixDai).

![mnist_result1](assets/mnist1.png)

![mnist_result2](assets/mnist2.png)

![mnist_result3](assets/mnist3.png)

More results can be found [here](./assets/) and [here](./web/img/).


## Training details

Details of the loss of Discriminator and Generator (with custom dataset not celebA).

![d_loss](assets/d_loss.png)

![g_loss](assets/g_loss.png)

Details of the histogram of true and fake result of discriminator (with custom dataset not celebA).

![d_hist](assets/d_hist.png)

![d__hist](assets/d__hist.png)


## Related works

- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [DiscoGAN-pytorch](https://github.com/carpedm20/DiscoGAN-pytorch)
- [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
