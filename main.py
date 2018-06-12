import os
import shutil

import numpy as np

from model import DCGAN
from global_config import INTERNAL_CHECKPOINT_DIR, SAMPLE_DIR, DATA_DIR
from utils import pp, visualize, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images (must be a square number) [64]")
flags.DEFINE_integer("input_height", 256, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("save_frequency", 100, "Save sample images every n-th batch [100]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 50, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

    # If checkpoint_dir specified exists, copy its contents to the checkpoint dir used internally
    if os.path.exists(FLAGS.checkpoint_dir):
        shutil.copytree(FLAGS.checkpoint_dir, INTERNAL_CHECKPOINT_DIR)
    elif not os.path.exists(INTERNAL_CHECKPOINT_DIR):
        os.makedirs(INTERNAL_CHECKPOINT_DIR)


    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:

        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                z_dim=FLAGS.generate_test_images,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=INTERNAL_CHECKPOINT_DIR,
                sample_dir=SAMPLE_DIR,
                data_dir=DATA_DIR,
            )
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                z_dim=FLAGS.generate_test_images,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=INTERNAL_CHECKPOINT_DIR,
                sample_dir=SAMPLE_DIR,
                data_dir=DATA_DIR,
            )

        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(INTERNAL_CHECKPOINT_DIR)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        # Visualisation
        OPTION = 1
        visualize(sess, dcgan, FLAGS, OPTION)

        # Copy checkpoints to floydhub output for potential reuse
        shutil.copytree(INTERNAL_CHECKPOINT_DIR,
                        '{}/checkpoint'.format(SAMPLE_DIR))


if __name__ == '__main__':
    tf.app.run()
