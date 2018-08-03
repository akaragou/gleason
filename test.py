#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
import unet
import unet
import cv2
import h5py
from scipy import signal
from skimage import morphology
from tensorflow.contrib import slim
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from config import GleasonConfig
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tf_record import read_and_decode
import scipy.stats as st

def test_unet(device):
    """
    Loads test tf records and test and visaulizes models performance.
    Input: gpu device number 
    Output None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    config = GleasonConfig()

    # load test data
    test_filename_queue = tf.train.string_input_producer([config.test_fn], num_epochs = 1)

    test_images, test_masks  = read_and_decode(filename_queue = test_filename_queue,
                                                     img_dims = config.input_image_size,
                                                     size_of_batch =  1,
                                                     augmentations_dic = config.val_augmentations_dic,
                                                     num_of_threads = 1,
                                                     shuffle = False)

 
    with tf.variable_scope('unet') as unet_scope:
        with slim.arg_scope(unet.unet_arg_scope()):
            test_logits, _ = unet.Unet(test_images,
                                        is_training=False,
                                        num_classes = config.output_shape,
                                        scope=unet_scope)

    test_prob = tf.nn.softmax(test_logits)
    test_scores = tf.argmax(test_prob, axis=3)

    iou = config.IOU(test_scores, test_masks)

    restorer = tf.train.Saver()
    print "Variables stored in checpoint:"
    print_tensors_in_checkpoint_file(file_name=config.test_checkpoint, tensor_name='',all_tensors='')
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
 
        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
        restorer.restore(sess, config.test_checkpoint)        

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        count = 0
        test_preds = []
        total_iou = 0
        try:

            while not coord.should_stop():
                
                # gathering results for models performance and mask predictions
                np_iou, np_image, np_mask, np_predicted_mask = sess.run([iou, test_images, test_masks, test_scores])

                # f, (ax1, ax2, ax3) = plt.subplots(1,3)
                # ax1.imshow(np.squeeze(np_image))
                # ax1.set_title('input image')
                # ax1.axis('off')
                # ax2.imshow(np.squeeze(np_mask), cmap='gray')
                # ax2.set_title('ground truth mask')
                # ax2.axis('off')
                # ax3.imshow(np.squeeze(np_predicted_mask), cmap='gray')
                # ax3.set_title('predicted mask')
                # ax3.axis('off')
                # plt.show()
                # test_preds.append(np_predicted_mask)
                print "count: {0} || IOU: {1}".format(count, np_iou)
                count += 1 
                total_iou += np_iou

        except tf.errors.OutOfRangeError:
            print "Total Mean IOU: {0}".format(total_iou/count)
            print 'Done Testing model :)'
        finally:
            coord.request_stop()  
        coord.join(threads)



if __name__ == '__main__':

    # f = h5py.File('/media/data_cifs/andreas/connectomics/Berson/updated_Berson.h5', 'r')
    # volume = f['volume'][:].astype('uint8')
    # masks =  f['masks'][:].astype('uint8')
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()

    test_unet(args.device)





