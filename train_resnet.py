#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
import resnet_v2
import unet_preprocess
from tensorflow.contrib import slim
from resnet_config import GleasonConfig
from tfrecord import vgg_preprocessing, tfrecord2metafilename, read_and_decode, normalize

def print_model_variables():
    print "Model Variables:"
    for var in slim.get_model_variables():
        print var 

def train_resnet(device, num_classes, num_layers, normalization):
    """
    Loads training and validations tf records and trains resnet model and validates every number of fixed steps.
    Input: gpu device number 
    Output None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = GleasonConfig() # loads pathology configuration defined in resnet_config
    # load training data

    train_meta = np.load(tfrecord2metafilename(config.train_fn))
    print 'Using train tfrecords: {0} | {1} images'.format(config.train_fn, len(train_meta['labels']))
    train_filename_queue = tf.train.string_input_producer(
    [config.train_fn], num_epochs=config.num_train_epochs)
    # load validation data
    val_meta = np.load(tfrecord2metafilename(config.val_fn))
    print 'Using test tfrecords: {0} | {1} images'.format(config.val_fn, len(val_meta['labels']))
    val_filename_queue = tf.train.string_input_producer(
    [config.val_fn], num_epochs=config.num_train_epochs)

    # defining model names and setting output and summary directories
    model_train_name = 'lung_resnet' + '_' + num_layers 
    if int(num_classes) == 2:
        model_train_name = model_train_name + '_' + 'binary'
    elif int(num_classes) == 4:
        model_train_name = model_train_name + '_' + 'multi'
    else:
        raise Exception('Invalid number of classes!')


    if normalization == "standard":
        model_train_name = model_train_name + '_' + 'standard'
    elif normalization == "unet":
        model_train_name = model_train_name + '_' + 'unet'
    else:
        raise Exception('Not known normalization! Options are: standard and unet.')


    dt_stamp = time.strftime(model_train_name  + "_%Y_%m_%d_%H_%M_%S")
    out_dir = config.get_results_path(model_train_name, dt_stamp)
    summary_dir = config.get_summaries_path(model_train_name, dt_stamp)
    print '-'*60
    print 'Training model: {0}'.format(dt_stamp)
    print '-'*60

    if int(num_classes) == 2:
        print "Converting to Output Shape to Binary..."
        config.output_shape = 2
    elif int(num_classes) == 4:
        config.output_shape = 4
    else:
        raise Exception('Invalid number of classes!')

    train_img, train_t_l, _ = read_and_decode(filename_queue = train_filename_queue,
                                             img_dims = config.input_image_size,
                                             model_dims = config.model_image_size,
                                             size_of_batch = config.train_batch_size,
                                             augmentations_dic = config.train_augmentations_dic,
                                             num_of_threads = 4,
                                             shuffle = True)

    val_img, val_t_l, _  = read_and_decode(filename_queue = val_filename_queue,
                                         img_dims = config.input_image_size,
                                         model_dims = config.model_image_size,
                                         size_of_batch = config.val_batch_size,
                                         augmentations_dic = config.val_augmentations_dic,
                                         num_of_threads = 4,
                                         shuffle = False)

    if int(num_classes) == 2:
        print "Converting labels to Binary..."
        train_t_l = tf.clip_by_value(train_t_l, 0, 1)
        val_t_l = tf.clip_by_value(val_t_l, 0, 1)

    # summaries to use with tensorboard check https://www.tensorflow.org/get_started/summaries_and_tensorboard
    tf.summary.image('train images', train_img, max_outputs=10)
    tf.summary.image('validation images', val_img, max_outputs=10)

    # creating step op that counts the number of training steps
    step = tf.train.get_or_create_global_step()
    step_op = tf.assign(step, step+1)

    if num_layers == "50":
        print "Loading Resnet 50..."
        with tf.variable_scope('resnet_v2_50') as resnet_scope:
            with tf.name_scope('train') as train_scope:

                if normalization == "standard":
                    train_img = normalize(train_img)
                elif normalization == "unet":
                    print train_img
                    train_img, _ = unet_preprocess.unet(train_img,
                                                     is_training = True,
                                                     is_batch_norm = True,
                                                     scope=resnet_scope,
                                                     num_channels = 1)
                    print train_img
                else:
                    raise Exception('Not known normalization! Options are: standard and unet.')
                with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg)):
                    train_target_logits, _ = resnet_v2.resnet_v2_50(inputs = train_img,                                                               
                                                                    num_classes = config.output_shape,
                                                                    scope = resnet_scope,
                                                                    is_training = True)
            # print_model_variables()
            resnet_scope.reuse_variables() # training variables are reused in validation graph 
            with tf.name_scope('val') as val_scope:
                
                if normalization == "standard":
                    val_img = normalize(val_img)
                elif normalization == "unet":
                    val_img,_ = unet_preprocess.unet(val_img,
                                                   is_training = False,
                                                   is_batch_norm = True,
                                                   scope=resnet_scope,
                                                   num_channels = 1)
                else:
                    raise Exception('Not known normalization! Options are: standard and unet.')
                with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg)):
                    val_target_logits, _ = resnet_v2.resnet_v2_50(inputs = val_img,
                                                                  num_classes = config.output_shape, 
                                                                  scope = resnet_scope,
                                                                  is_training=False)
    elif num_layers == "101":
        print "Loading Resnet 101..."
        with tf.variable_scope('resnet_v2_101') as resnet_scope:
            with tf.name_scope('train') as train_scope:

                if normalization == "standard":
                    train_img = normalize(train_img)
                elif normalization == "unet":
                    train_img, _ = unet_preprocess.unet(train_img,
                                                     is_training = True,
                                                     is_batch_norm = True,
                                                     scope = resnet_scope,
                                                     num_channels = 1)
                else:
                    raise Exception('Not known normalization! Options are: standard and unet.')
                with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg)):
                    train_target_logits, _ = resnet_v2.resnet_v2_101(inputs = train_img,
                                                                     num_classes = config.output_shape, 
                                                                     scope = resnet_scope,
                                                                     is_training = True)
            # print_model_variables()
            resnet_scope.reuse_variables() # training variables are reused in validation graph 
            with tf.name_scope('val') as val_scope:
                
                if normalization == "standard":
                    val_img = normalize(val_img)
                elif normalization == "unet":
                    val_img,_ = unet_preprocess.unet(val_img,
                                                   is_training = False,
                                                   is_batch_norm = True,
                                                   scope = resnet_scope,
                                                   num_channels = 1)
                else:
                    raise Exception('Not known normalization! Options are: standard and unet.')
                with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg)):
                    val_target_logits, _ = resnet_v2.resnet_v2_101(inputs = val_img, 
                                                                   num_classes = config.output_shape,
                                                                   scope = resnet_scope,
                                                                   is_training = False)

    
    else:
        raise Expection("Wrong number of layers! allowed numbers are and 50 and 101.")


    train_t_l_one_hot = tf.one_hot(train_t_l, config.output_shape)

    class_weights = np.load('train_class_weights.npy').astype(np.float32)

    if int(num_classes) == 2:
        class_weights = np.array([class_weights[0], class_weights[1]*class_weights[2]*class_weights[3]])
    tf_class_weights = tf.constant(class_weights)
    weight_map = tf.multiply(train_t_l_one_hot, tf_class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = train_t_l_one_hot, logits = train_target_logits)

    weighted_batch_loss = tf.multiply(batch_loss, weight_map)

    loss = tf.reduce_mean(weighted_batch_loss)
    tf.summary.scalar("loss", loss)

    lr = tf.train.exponential_decay(
            learning_rate = config.initial_learning_rate,
            global_step = step_op,
            decay_steps = config.decay_steps,
            decay_rate = config.learning_rate_decay_factor,
            staircase = True) # if staircase is True decay the learning rate at discrete intervals

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

    train_prob = tf.nn.softmax(train_target_logits)
    train_accuracy = config.class_accuracy(logits=train_prob, labels=train_t_l)
    tf.summary.scalar("training accuracy", train_accuracy)

    val_prob = tf.nn.softmax(val_target_logits)
    val_accuracy = config.class_accuracy(logits=val_prob, labels=val_t_l)
    tf.summary.scalar("validation accuracy", val_accuracy)

    if config.restore:
        # adjusting variables to keep in the model
        # variables that are exluded will allow for transfer learning (normally fully connected layers are excluded)
        exclusions = [scope.strip() for scope in config.checkpoint_exclude_scopes]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        print "Restroing variables:"
        for var in variables_to_restore:
            print var
        restorer = tf.train.Saver(variables_to_restore)

    saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=100)

    summary_op = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))

        if config.restore:
            restorer.restore(sess, config.model_path)

        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        np.save(os.path.join(out_dir, 'training_config_file'), config)

        val_acc_max, losses = 0, []

        try:

            while not coord.should_stop():
                
                start_time = time.time()
                step_count, loss_value, train_acc, lr_value, _ = sess.run([step_op, loss, train_accuracy, lr, train_op])
                losses.append(loss_value)
                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                step_count = step_count - 1 

                if step_count % config.validate_every_num_steps == 0:
                    it_val_acc = np.asarray([])
                    for num_vals in range(config.num_batches_to_validate_over):
                        # Validation accuracy as the average of n batches
                        it_val_acc = np.append(
                            it_val_acc, sess.run(val_accuracy))
                    
                    val_acc_total = it_val_acc.mean()
                    # Summaries
                    summary_str= sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step_count)

                    # Training status and validation accuracy
                    msg = '{0}: step {1}, loss = {2:.4f} ({3:.2f} examples/sec; '\
                        + '{4:.2f} sec/batch) | Training accuracy = {5:.4f} '\
                        + '| Validation accuracy = {6:.4f} | logdir = {7}'
                    print msg.format(
                          datetime.datetime.now(), step_count, loss_value,
                          (config.train_batch_size / duration), float(duration),
                          train_acc, val_acc_total, summary_dir)
                    # Save the model checkpoint if it's the best yet
                    if val_acc_total >= val_acc_max:
                        file_name = 'resnet_{0}_{1}'.format(dt_stamp, step_count)
                        saver.save(
                            sess,
                            config.get_checkpoint_filename(model_train_name, file_name))
                        # Store the new max validation accuracy
                        val_acc_max = val_acc_total

                else:
                    # Training status
                    msg = '{0}: step {1}, loss = {2:.4f} ({3:.2f} examples/sec; '\
                        + '{4:.2f} sec/batch) | Training accuracy = {5:.4f}'
                    print msg.format(datetime.datetime.now(), step_count, loss_value,
                          (config.train_batch_size / duration),
                          float(duration), train_acc)
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
    parser.add_argument("num_classes")
    parser.add_argument("num_layers")
    parser.add_argument("normalization")
    args = parser.parse_args()
    train_resnet(args.device,  args.num_classes, args.num_layers, args.normalization) 
