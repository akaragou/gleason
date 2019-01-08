#!/usr/bin/env python
from __future__ import division
import argparse
import os
import tensorflow as tf
import datetime
import numpy as np
import time
from resnet_config import GleasonConfig
from tensorflow.contrib import slim
from operator import add
from sklearn import metrics
import resnet_v2
import unet_preprocess
from tfrecord import vgg_preprocessing, tfrecord2metafilename, read_and_decode, normalize
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import math

def test_resnet(device, num_classes, num_layers, dataset, normalization, checkpoint):
    """
    Computes accuracy for the test dataset
    Inputs: device - gpu device
            num_classes - number of output classes 
            num_layers - number of layers selected for ResNet 
            dataset - dataset selected, options are val and test
            normalization - normalization used options are standard z-score normalization or unet normalization
            checkpoint - file where graph model weights are stored
    Output: None
    """
    print dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = GleasonConfig() # loads pathology configuration 

    if dataset == 'val':  
        print "Using val..." 
        config.test_fn = os.path.join(config.main_dir, 'tfrecords/val.tfrecords')
    elif dataset == 'test':
        print "Using test..." 
        config.test_fn = os.path.join(config.main_dir, 'tfrecords/test.tfrecords')
    else:
        config.test_fn = None

    config.test_checkpoint = checkpoint
    print "Loading checkpoint: {0}".format(checkpoint)

    if int(num_classes) == 2:
        print "Converting to Output Shape to Binary..."
        config.output_shape = 2
    elif int(num_classes) == 4:
        config.output_shape = 4
    else:
        raise Exception('Invalid number of classes!')

    batch_size = 128
    # loading test data
    test_meta = np.load(tfrecord2metafilename(config.test_fn))
    print 'Using {0} tfrecords: {1} | {2} images'.format(dataset, config.test_fn, len(test_meta['labels']))
    test_filename_queue = tf.train.string_input_producer([config.test_fn] , num_epochs=1) # 1 epoch, passing through the
                                                                                            # the dataset once

    test_img, test_t_l, test_f_p  = read_and_decode(filename_queue = test_filename_queue,
                                           img_dims = config.input_image_size,
                                           model_dims = config.model_image_size,
                                           size_of_batch = batch_size,
                                           augmentations_dic = config.val_augmentations_dic,
                                           num_of_threads = 4,
                                           shuffle = False)
    if int(num_classes) == 2:
        print "Converting labels to Binary..."
        test_t_l = tf.clip_by_value(test_t_l, 0, 1)

    if num_layers == "50":
        print "Loading Resnet 50..."
        if normalization == "standard":
            print "Using standard normalization..."
            test_img = normalize(test_img)
        elif normalization == "unet":
            print "Using unet normalization..."
            test_img,_ = unet_preprocess.unet(test_img,
                                           is_training = False,
                                           is_batch_norm = False,
                                           num_channels = 1)
        else:
            raise Exception('Not known normalization! Options are: standard and unet.')
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg)):
            test_target_logits, _ = resnet_v2.resnet_v2_50(inputs = test_img, 
                                                       num_classes = config.output_shape,
                                                       is_training = False) 
    elif num_layers == "101":
        print "Loading Resnet 101..."
        if normalization == "standard":
            print "Using standard normalization..."
            test_img = normalize(test_img)
        elif normalization == "unet":
            print "Using unet normalization..."
            test_img,_ = unet_preprocess.unet(test_img,
                                           is_training = False,
                                           is_batch_norm = False,
                                           num_channels = 1)
        else:
            raise Exception('Not known normalization! Options are: standard and unet.')
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg)):
            test_target_logits, _ = resnet_v2.resnet_v2_101(inputs = test_img, 
                                                           num_classes = config.output_shape,
                                                           is_training = False)
    else:
        raise Expection("Wrong number of layers! allowed numbers are 50 and 101.")
        
    target_prob = tf.nn.softmax(test_target_logits)
    prob_and_label_files = [target_prob,  test_t_l, test_f_p]
    restorer = tf.train.Saver()
    print "Variables stored in checkpoint:"
    print_tensors_in_checkpoint_file(file_name=config.test_checkpoint, tensor_name='', all_tensors='')
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      
        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

        restorer.restore(sess, config.test_checkpoint)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        all_predictions_target = []
        all_predictions_t_n = []
        all_labels = []
        all_files = []

        batch_num = 1
        try:
            print "Total number of batch iterations needed: {0}".format(int(math.ceil(len(test_meta['labels'])/batch_size)))
            while not coord.should_stop():
               
                np_prob_and_label_files = sess.run(prob_and_label_files)

                target_probs = np_prob_and_label_files[0]
                labels = np_prob_and_label_files[1] 
                files = np_prob_and_label_files[2]
                
                all_labels += list(labels) 
                all_files += list(files)
                all_predictions_target += list(np.argmax(target_probs, axis=1)) 
                print "evaluating current batch number: {0}".format(batch_num)
                batch_num +=1
         
        except tf.errors.OutOfRangeError:
            print "{0} accuracy: {1:.2f}".format(dataset, (metrics.accuracy_score(all_labels, all_predictions_target)*100))
            if int(num_classes) == 2:
                print "{0} precision: {1:.2f}".format(dataset, (metrics.precision_score(all_labels, all_predictions_target)*100))
                print "{0} recall: {1:.2f}".format(dataset, (metrics.recall_score(all_labels, all_predictions_target)*100))
            print 
        finally:
            coord.request_stop()  
        coord.join(threads) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("num_classes")
    parser.add_argument("num_layers")
    parser.add_argument("dataset")
    parser.add_argument("normalization")
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    test_resnet(args.device, args.num_classes, args.num_layers, args.dataset, args.normalization, args.checkpoint)
    