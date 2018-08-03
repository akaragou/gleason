from __future__ import division
import os
import tensorflow as tf
import numpy as np
from config import GleasonConfig
from tf_record import read_and_decode
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import math

PI = tf.constant(math.pi)

def test_tf_record(device):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = GleasonConfig() # loads configuration

    # load training data
    filename_queue = tf.train.string_input_producer(
    [config.exp_fn], num_epochs=4)
   
 
    train_images, train_masks = read_and_decode(filename_queue = filename_queue,
                                                img_dims = config.input_image_size,
                                                size_of_batch = 2,
                                                augmentations_dic = config.train_augmentations_dic,
                                                num_of_threads = 1,
                                                shuffle = False)

    
    iou = config.IOU(train_masks, train_masks)
    # mean = 0.0
    # sigma = 1.0
    # x = tf.linspace(-1.0, 1.0, 100)

    # z = ((1.0 / (sigma * tf.sqrt(2.0 * PI))) * tf.exp(tf.negative(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))))
    # ksize = z.get_shape().as_list()[0]
   
    # z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))

    # z_3d = tf.stack([z_2d,z_2d,z_2d],axis=-1)
    # print 'here'
    # print z_3d

    # z_4d = tf.reshape(z_3d, [ksize, ksize, 3, 1])
    # print(z_4d.get_shape().as_list())

    # convolved = tf.nn.depthwise_conv2d(train_images, z_4d, strides=[1, 1, 1, 1], padding='SAME')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))

    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            while not coord.should_stop():

                np_image, np_mask, np_iou = sess.run([train_images,train_masks,iou])
               
                f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                ax1.imshow(np.squeeze(np_image[0,:,:,:]))
                ax2.imshow(np.squeeze(np_mask[0,:,:,:]), cmap='gray')
                ax3.imshow(np.squeeze(np_image[1,:,:,:]))
                ax4.imshow(np.squeeze(np_mask[1,:,:,:]), cmap='gray')
                plt.show()
                
                # plt.pause(0.2)

        except tf.errors.OutOfRangeError:
            print 'Done'
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()



def test(device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    y_true = tf.constant([2, 1, 2, 2, 1, 4])
    y_pred = tf.constant([2, 1, 2, 2, 1, 4])

    con = tf.concat([y_true, y_pred], axis=-1) 
    uni = tf.unique(con)
    uni_size =tf.cast(tf.size(uni[0]), dtype=tf.float32)

    c =  tf.cast(tf.confusion_matrix(y_true, y_pred, num_classes = 5), dtype=tf.float32)
    intersection = tf.diag_part(c)
    ground_truth_set = tf.reduce_sum(c, axis=0)
    predicted_set = tf.reduce_sum(c, axis=1)

    union = ground_truth_set + predicted_set - intersection
    iou = intersection / (union + 1e-8)
    mean_iou = tf.reduce_sum(iou) / uni_size

    with tf.Session() as sess:

        print sess.run(mean_iou)
        # print sess.run(intermediate_tensor)
   
if __name__ == '__main__':
    test_tf_record(3)