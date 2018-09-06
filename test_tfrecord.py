from __future__ import division
import os
import tensorflow as tf
import numpy as np
from resnet_config import GleasonConfig
from tfrecord import read_and_decode, normalize, draw_grid
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import math

PI = tf.constant(math.pi)


def warp(img, model_dims=[224, 224, 3], size_of_batch=1):

    X = tf.random_uniform([model_dims[0], model_dims[1]])*2 - 1
    Y = tf.random_uniform([model_dims[0], model_dims[1]])*2 - 1
    X = tf.reshape(X, [1, model_dims[0],model_dims[1], 1])
    Y = tf.reshape(Y, [1, model_dims[0],model_dims[1], 1])

    mean = 0.0
    sigma = 1.0
    alpha = 10.0
    ksize = 128

    x = tf.linspace(-3.0, 3.0, ksize)
    z = ((1.0 / (sigma * tf.sqrt(2.0 * PI))) * tf.exp(tf.negative(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))))
    ksize = z.get_shape().as_list()[0]
    z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
    z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])

    X_convolved = tf.nn.conv2d(X, z_4d, strides=[1, 1, 1, 1], padding='SAME')
    Y_convolved = tf.nn.conv2d(Y, z_4d, strides=[1, 1, 1, 1], padding='SAME')

    X_convolved = (X_convolved / tf.reduce_max(X_convolved))*alpha
    Y_convolved = (Y_convolved / tf.reduce_max(Y_convolved))*alpha

    trans = tf.stack([X_convolved,Y_convolved], axis=-1)
    trans = tf.reshape(trans, [-1])

    batch_trans = tf.tile(trans, [size_of_batch])
    batch_trans = tf.reshape(batch_trans, [size_of_batch, model_dims[0], model_dims[1] ,2])

    img = tf.reshape(img, [size_of_batch, model_dims[0], model_dims[1], model_dims[2]])

    img = tf.contrib.image.dense_image_warp(img, batch_trans)

    return img

def test_tf_record(device):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = GleasonConfig() # loads configuration

    dic = {
            'rand_flip_left_right':False,
            'rand_flip_top_bottom':False,
            'rand_crop': False,
            'rand_rotate':False,
            'warp':False,
            'distort_brightness_constrast':False,
            'grayscale':False
        }

    # load training data
    filename_queue = tf.train.string_input_producer([config.exp_fn], num_epochs=1)
   
 
    images, labels, _  = read_and_decode(filename_queue = filename_queue,
                                                img_dims = [256, 256, 3],
                                                size_of_batch = 1,
                                                augmentations_dic = dic,
                                                num_of_threads = 1,
                                                shuffle = False)
    # norm_images = normalize(images)
    # warped_images = warp(images)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            while not coord.should_stop():

                np_img = sess.run(images)
                # print np.shape(image)
                # print np.shape(norm_img)
                # print norm_img

                print np_img
                plt.imshow(np.squeeze(np_img))
                plt.show()
                # f, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.imshow(np.squeeze(image), cmap='gray')
                # ax2.imshow(np.squeeze(warp_iamge), cmap='gray')
                # plt.show()

        except tf.errors.OutOfRangeError:
            print 'Done'
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    test_tf_record(5)