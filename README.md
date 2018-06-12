# FloydHub Clone of carpedm20/DCGAN-tensorflow
This is a slightly modified version of <a href="https://github.com/carpedm20" target="_blank">carpedm20</a>'s code. It is meant for use with the <a href="https://floydhub.com" target="_blank">FloydHub</a> deep learning platform. If you wish to run a DCGAN on your own hardware or cloud infrasturcture other than that of FloydHub, I would suggest looking at the <a href="https://github.com/carpedm20/DCGAN-tensorflow" target="_blank">original repository</a>.

## Useage

#### Getting training data

You can either use one of the datasets already on the FloydHub platform or upload your own.

If using your own dataset, ensure that 
* all images are the same dimensions and colour mode e.g. all 512x512px and either all greyscale or all RGB. A mixture of colour modes could lead the model to fail half-way through training.
* all images are placed in sub-directories within the root directory of your dataset. This program will only load images in subdirectoies beginning with the word '**train**' e.g. **train**ing_data (all images in one directory) or **train**01 **train**01 **train**03 etc. (images in several _batches_) 

#### First run
Once your Floydhub dataset and project are initilised, use the **floyd run** command from the directory where you've initialised your project.

##### Explanation of command flags

    $ floyd run \
        --gpu (It is recommended to use GPU for deep learning over images.
               Omiting this switch will result in slower training using CPU.) \
        --env tensorflow-1.8 (The original requirment was Tensorflow  0.12.1.
                              However, the code works with newer versions.) \
        --data <PATH TO FLOYDHUB DATASET>:<DIRECTORY NAME FOR MOUNTING DATA> \
         "python main.py \
        --dataset=<DIRECTORY NAME FOR MOUNTING DATA> \
        --checkpoint_dir=<THE DIRECTORY WHERE YOU WANT CHECPOINTS TO BE COPIED
                          FROM. THIS ALLOWS YOU TO RESUME TRAINING WHERE A PREVIOUS 
                          RUN OF THE PROGRAM LEFT OFF. DEFAULT: /checkpoint> \
        --batch_size= <THE NUMBER OF TRAINGING EXAMPLES TO PROCESS AT ONCE. 
                       THIS MUST BE A SQUARE NUMBER. e.g. 16, 25, 36, 49
                       GENERALLY THIS WILL BE SMALLER IF YOUR IMAGES ARE LARGER.
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
        --epoch=<NUMBER OF TIMES THE PROGRAM WILL CYCLE THROUGH YOUR TRAINING DATA 
                 BEFORE COMPLETING TRAINING. DEFAULT: 25> \
        --train"
        
##### Example of command using a FloydHub dataset

    $ floyd run \
        --gpu \
        --env tensorflow-1.8 \
        --data milkgan/datasets/selfie-100k-512/1:selfie-100k-512 \
        "python main.py \
        --dataset=selfie-100k-512 \
        --batch_size=25 \
        --save_frequency=50 \
        --generate_test_images=25 \
        --input_height=512 \
        --input_width=512 \
        --output_height=512 \
        --output_width=512 \
        --epoch=5 \
        --train" 

#### Further runs
The code has been modified to save the model's checkpoint directory in FloydHub's output. This means that you can train your model incrementally. 

In order to do this, you will need to copy the **checkpoint** directory from the output of the previous run of your project and upload this to Floydhub as a separate dataset. In the example below the path of the checkpoint dataset is milkgan/datasets/self-dcgan-checkpoint-5-epochs/1.

##### Example of command using FloydHub dataset and checkpoints from a previous job
    floyd run \
        --gpu \
        --env tensorflow-1.8 \
        --data milkgan/datasets/selfie-100k-512/1:selfie-100k-512 \
        --data milkgan/datasets/self-dcgan-checkpoint-5-epochs/1:checkpoint \
        "python main.py \
        --dataset=selfie-100k-512 \
        --checkpoint_dir=/checkpoint \
        --batch_size=25 \
        --save_frequency=100 \
        --generate_test_images=25 \
        --input_height=512 \
        --input_width=512 \
        --output_height=512 \
        --output_width=512 \
        --epoch=5 \
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
