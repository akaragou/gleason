#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
import unet
import itertools
from tensorflow.contrib import slim
import matplotlib.pyplot as plt
from config import GleasonConfig
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from old_tf_record import read_and_decode


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()

def test_Unet(device, trinary, grayscale, checkpoint):
    """
    Loads test tf records and test and visaulizes models performance.
    Input: gpu device number 
    Output None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    config = GleasonConfig()

    config.test_checkpoint = checkpoint

    # load test data
    test_filename_queue = tf.train.string_input_producer([config.test_fn], num_epochs = 1)

    if grayscale == "1":
        print "Converting to Grayscale..."
        config.val_augmentations_dic['grayscale'] = True
    else:
        config.val_augmentations_dic['grayscale'] = False

    if trinary == "1":
        print "Converting to Trinary..."
        config.output_shape = 3
    else:
        config.output_shape = 5

    test_images, test_masks, f = read_and_decode(filename_queue = test_filename_queue,
                                                     img_dims = config.input_image_size,
                                                     size_of_batch =  1,
                                                     augmentations_dic = config.val_augmentations_dic,
                                                     num_of_threads = 1,
                                                     shuffle = False)
    if trinary == "1":
        test_masks = tf.clip_by_value(test_masks, 0, 2)
    
    with tf.variable_scope('unet') as unet_scope:
        with slim.arg_scope(unet.unet_arg_scope()):
            test_logits, _ = unet.Unet(test_images,
                                        is_training=False,
                                        num_classes = config.output_shape,
                                        scope=unet_scope)

    test_scores = tf.argmax(test_logits, axis=3)

    iou = config.mean_IOU(test_scores, test_masks)
    accuracy = config.pixel_accuracy(test_scores, test_masks)

    c_f = tf.confusion_matrix(tf.reshape(test_masks, [-1]),tf.reshape(test_scores, [-1]),num_classes=config.output_shape)

    restorer = tf.train.Saver()
    print "Variables stored in checkpoint:"
    print_tensors_in_checkpoint_file(file_name=config.test_checkpoint, tensor_name='',all_tensors='')
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
 
        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
        restorer.restore(sess, config.test_checkpoint)        

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        count = 0
        test_preds = []
        total_iou = 0
        total_accuracy = 0
        confusion_matrix = [[0 for _ in range(config.output_shape)] for _ in range(config.output_shape)]

        try:

            while not coord.should_stop():
                
                # gathering results for models performance and mask predictions
                np_accuracy, np_iou, np_image, np_mask, np_predicted_mask, np_c_f = sess.run([accuracy, iou, test_images, test_masks, test_scores, c_f])

                print "count: {0} || Mean IOU: {1} || Pixel Accuracy: {2}".format(count, np_iou, np_accuracy)
                count += 1 
                total_iou += np_iou
                total_accuracy += np_accuracy

                for i in range(len(np_c_f)):
                    for j in range(len(np_c_f[0])):
                        confusion_matrix[i][j] += np_c_f[i][j]

                # f, (ax1, ax2, ax3) = plt.subplots(1,3)
                # ax1.imshow(np.squeeze(np_image), cmap='gray')
                # ax1.set_title('input image')
                # ax1.axis('off')
                # ax2.imshow(np.squeeze(np_mask),vmin=0,vmax=2,cmap='jet')
                # ax2.set_title('ground truth mask')
                # ax2.axis('off')
                # ax3.imshow(np.squeeze(np_predicted_mask),vmin=0,vmax=2,cmap='jet')
                # ax3.set_title('predicted mask')
                # ax3.axis('off')
                # plt.show()

        except tf.errors.OutOfRangeError:
            print "Total Mean IOU: {0}".format(total_iou/count)
            print "Total Pixel Accuracy: {0}".format(total_accuracy/count)
            print 'Done Testing model :)'
        finally:
            coord.request_stop()  
        coord.join(threads)

        np.save('results.npy',confusion_matrix)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("trinary")
    parser.add_argument("grayscale")
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    # test_Unet(args.device, args.trinary, args.grayscale, args.checkpoint)
    cf = np.load('results.npy')
    # plot_confusion_matrix(cf, ['Background', 'Benign', 'Gleason 3', 'Gleason 4', 'Gleason 5'], normalize=True)
    plot_confusion_matrix(cf, ['Background', 'Benign', 'Malignant'], normalize=True)




