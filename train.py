#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
import unet
from tensorflow.contrib import slim
from config import GleasonConfig
from tf_record import read_and_decode
import resource


def train(device, loss_name, trinary, grayscale):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = GleasonConfig() # loads configuration

    # load training data
    train_filename_queue = tf.train.string_input_producer(
    [config.pretrain_train_fn], num_epochs=config.num_train_epochs)
    # load validation data
    val_filename_queue = tf.train.string_input_producer(
    [config.pretrain_val_fn], num_epochs=config.num_train_epochs)

    model_train_name = 'pretraining_gleason_unet'
    name = loss_name
    
    if trinary == 1:
        name = name + '_' + 'trinary'
    else:
        name = name + '_' + 'multi'
    if grayscale == 1:
        name = name + '_' + 'grayscale'
    else:
        name = name + '_' + 'color'

    dt_stamp = time.strftime(name + "_%Y_%m_%d_%H_%M_%S")
    out_dir = config.get_results_path(model_train_name, dt_stamp)
    summary_dir = config.get_summaries_path(model_train_name, dt_stamp)
    print '-'*60
    print 'Training model: {0}'.format(dt_stamp)
    print '-'*60

    if grayscale == 1:
        print "Converting to Grayscale..."
        config.train_augmentations_dic['grayscale'] = True
        config.val_augmentations_dic['grayscale'] = True

    else:
        config.train_augmentations_dic['grayscale'] = False
        config.val_augmentations_dic['grayscale'] = False

    if trinary == 1:
        print "Converting to Output Shape to Trinary..."
        config.output_shape = 3
    else:
        config.output_shape = 5

    train_images, train_masks, _ = read_and_decode(filename_queue = train_filename_queue,
                                                img_dims = config.input_image_size,
                                                size_of_batch = config.train_batch_size,
                                                augmentations_dic = config.train_augmentations_dic,
                                                num_of_threads = 2,
                                                shuffle = True)
    
    val_images, val_masks, _  = read_and_decode(filename_queue = val_filename_queue,
                                             img_dims = config.input_image_size,
                                             size_of_batch =  config.val_batch_size,
                                             augmentations_dic = config.val_augmentations_dic,
                                             num_of_threads = 2,
                                             shuffle = True)

    if trinary == 1:
        print "Converting Masks to Trinary..."
        train_masks = tf.clip_by_value(train_masks, 0, 2)
        val_masks = tf.clip_by_value(val_masks, 0, 2)

    step = tf.train.get_or_create_global_step()
    step_op = tf.assign(step, step+1)

    # summaries to use with tensorboard check https://www.tensorflow.org/get_started/summaries_and_tensorboard

    with tf.variable_scope('unet') as unet_scope:
        with tf.name_scope('train') as train_scope:

            with slim.arg_scope(unet.unet_arg_scope()):
                train_logits, end_points = unet.Unet(train_images,
                                            is_training=True,
                                            is_batch_norm = True,
                                            num_classes = config.output_shape,
                                            scope=unet_scope)

                train_prob = tf.nn.softmax(train_logits)
                train_scores = tf.argmax(train_prob, axis=3)
                

            train_pred_mask = tf.cast(tf.expand_dims(train_scores,-1), dtype=tf.float32)
           
            
            flatten_train_masks = tf.reshape(train_masks, [-1])
            flatten_train_logits = tf.reshape(train_logits, [-1, config.output_shape])

            one_hot_lables = tf.one_hot(flatten_train_masks, config.output_shape, axis=-1)
            if loss_name == 'cross_entropy':
                batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_hot_lables, logits = flatten_train_logits)
            elif loss_name == 'weighted_cross_entropy':
                batch_loss = config.weighted_cross_entropy(one_hot_lables,flatten_train_logits,trinary,'updated_train_class_weights.npy')
            elif loss_name == 'focal_loss':
                batch_loss = config.focal_loss(one_hot_lables, flatten_train_logits,trinary,'updated_train_class_weights.npy')
            else:
                raise Exception("Not known loss! options are cross entropy and focal loss")    
            
            loss = tf.reduce_mean(batch_loss)
            tf.summary.scalar("loss", loss)

            if config.decay_learning_rate:
                lr = tf.train.exponential_decay(
                learning_rate = config.initial_learning_rate,
                global_step = step_op,
                decay_steps = config.decay_steps,
                decay_rate = config.learning_rate_decay_factor,
                staircase = True) # if staircase is True decay the learning rate at discrete intervals
            else:
                lr = tf.constant(config.initial_learning_rate)

            if config.optimizer == "adam":
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # used to update batch norm params. see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
                with tf.control_dependencies(update_ops):
                    train_op =  tf.train.AdamOptimizer(lr).minimize(loss)
            elif config.optimizer == "sgd":
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op =  tf.train.GradientDescentOptimizer(lr).minimize(loss)
            elif config.optimizer == "nestrov":
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op =  tf.train.MomentumOptimizer(lr, config.momentum, use_nesterov=True).minimize(loss)
            else:
                raise Exception("Not known optimizer! options are adam, sgd or nestrov")
               
        unet_scope.reuse_variables() # training variables are reused in validation graph 

        with tf.name_scope('val') as val_scope:

            with slim.arg_scope(unet.unet_arg_scope()):
                val_logits, _ = unet.Unet(val_images,
                                            is_training=False,
                                            is_batch_norm = True,
                                            num_classes = config.output_shape,
                                            scope=unet_scope)

                val_prob = tf.nn.softmax(val_logits)
                val_scores = tf.argmax(val_prob, axis=3)

            val_pred_mask = tf.cast(tf.expand_dims(val_scores,-1), dtype=tf.float32)

        with tf.name_scope('iou'):
            train_iou = config.mean_IOU(train_masks, train_scores)
            val_iou = config.mean_IOU(val_masks, val_scores)

        with tf.name_scope('accuracy'):
            train_accuracy = config.pixel_accuracy(train_masks, train_scores)
            val_accuracy = config.pixel_accuracy(val_masks, val_scores)

        tf.summary.scalar("train IOU", train_iou)
        tf.summary.scalar("val IOU", val_iou)
        tf.summary.scalar("train pixel acc", train_accuracy)
        tf.summary.scalar("val pixel acc", val_accuracy)

        train_masks = tf.cast(train_masks, dtype=tf.float32)
        val_masks = tf.cast(val_masks, dtype=tf.float32)

    with tf.name_scope('train_data'):
        train_images = train_images / 2.0
        train_images = train_images + 0.5
        tf.summary.image('train images', train_images, max_outputs=1)
        tf.summary.image('train masks', train_masks, max_outputs=1)

        if trinary == 1:
            train_pred_mask = config.rescale(train_pred_mask, 0, 2)
        else:
            train_pred_mask = config.rescale(train_pred_mask, 0, 4)

        tf.summary.image('train pred mask', train_pred_mask, max_outputs=1)

    with tf.name_scope('val_data'):
        val_images = val_images / 2.0
        val_images = val_images + 0.5
        tf.summary.image('validation images', val_images, max_outputs=1)
        tf.summary.image('validation masks', val_masks, max_outputs=1)

        if trinary == 1:
            val_pred_mask = config.rescale(val_pred_mask, 0, 2)
        else:
            val_pred_mask = config.rescale(val_pred_mask, 0, 4)

        tf.summary.image('val pred mask', val_pred_mask, max_outputs=1)

    saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=100)
   
    summary_op = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        np.save(os.path.join(out_dir, 'training_config_file'), config)
        max_val_acc, losses = -float('inf'), []

        try:

            while not coord.should_stop():
         
                start_time = time.time()
                step_count, loss_value, lr_value, train_iou_value, train_accuracy_value, _ = sess.run([step_op, loss, lr, train_iou, train_accuracy, train_op])
                losses.append(loss_value)
                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                step_count = step_count - 1 

                # print('Iteration ', step_count, ' maxrss: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) 
                if step_count % config.validate_every_num_steps == 0:

                    it_iou = np.asarray([])
                    it_acc = np.asarray([])
                    
                    for num_vals in range(config.num_batches_to_validate_over):
                     
                        it_iou = np.append(
                            it_iou, sess.run(val_iou))

                        it_acc = np.append(
                            it_acc, sess.run(val_accuracy))

                    val_iou_total = it_iou.mean()
                    val_acc_total = it_acc.mean()

                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step_count)

                    msg = '{0}: step {1}, loss = {2:.4f} ({3:.2f} examples/sec; '\
                        + '{4:.2f} sec/batch) | Train IOU = {5:.3f} | Train Accuracy = {6:.3f} '\
                        +  '| Val IOU = {7:.3f} | Val Accuracy = {8:.3f} | logdir = {9}'
                    print msg.format(
                          datetime.datetime.now(), step_count, loss_value,
                          (config.train_batch_size / duration), float(duration),
                          train_iou_value, train_accuracy_value, val_iou_total,
                          val_acc_total, summary_dir)
                    print "learning rate: ", lr_value

                    # Save the model checkpoint if it's the best yet
                    if val_acc_total >= max_val_acc:
                        file_name = 'unet_{0}_{1}'.format(dt_stamp, step_count)
                        saver.save(
                            sess,
                            config.get_checkpoint_filename(model_train_name, file_name))
                    
                        max_val_acc = val_acc_total
            
                else:
                    # Training status
                    msg = '{0}: step {1}, loss = {2:.4f} ({3:.2f} examples/sec; '\
                        + '{4:.2f} sec/batch) | Train IOU =  {5:.3f} | Train Accuracy =  {6:.3f}'
                    print msg.format(datetime.datetime.now(), step_count, loss_value,
                          (config.train_batch_size / duration),
                          float(duration),train_iou_value, train_accuracy_value)
                    # End iteration

        except tf.errors.OutOfRangeError:
            print 'Done training for {0} epochs, {1} steps.'.format(config.num_train_epochs, step_count)
        finally:
            coord.request_stop()
            np.save(os.path.join(out_dir, 'training_loss'), losses)
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("loss")
    parser.add_argument("trinary")
    parser.add_argument("grayscale")
    args = parser.parse_args()
    train(args.device, args.loss, int(args.trinary), int(args.grayscale)) # select gpu to train model on


