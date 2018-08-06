#!/usr/bin/python3
"""runs a model on some data."""

#from config import Config
#from dataset import Dataset
#import os
import tensorflow as tf
#from tensorflow.contrib.tpu.python.tpu import bfloat16
#import time
#import flags
#import records
import models

def metric_fn(labels, logits):
    """calculate the accuracy"""
    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}

def model_fn(features, labels, mode, params):
    """model function to give back a TPUEstimatorSpec."""
    # check params
    use_tpu = params['use_tpu']
    num_classes = params['num_classes']
    loss_name = params['loss_name']
    optimizer_name = params['optimizer_name']
    
    # build model
    train_logits = models.model[params['model_name']](features)

    # use standard name
    train_masks = labels

    flatten_train_masks = tf.reshape(train_masks, [-1])
    flatten_train_logits = tf.reshape(train_logits, [-1, num_classes])
    onehot_labels = tf.one_hot(flatten_train_masks, num_classes, axis=-1)
    if loss_name == 'cross_entropy':
        train_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=flatten_train_logits)
    elif loss_name == 'sigmoid_cross_entropy':
        train_loss = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=onehot_labels,
                logits=flatten_train_logits)
    elif loss_name == 'focal_loss':
        predictions = tf.nn.sigmoid(train_logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1. - predictions)
        epsilon = params['epsilon'] # 1e-8
        gamma = params['gamma'] # 2.0
        alpha = params['alpha'] # 0.25
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        train_loss = tf.reduce_sum(
                -alpha_t * tf.pow(1. - predictions_pt, gamma) * 
                tf.log(predictions_pt+epsilon), axis=1)

    else:
        raise Exception("unknown loss")
                
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        if optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params['lr']) #XXX:  change
        elif optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(params['lr'])
        elif optimizer_name == 'nestrov':
            optimizer = tf.train.MomentumOptimizer(params['lr'],
                    params['lr_momentum'], use_nesterov=True)
        else:
            raise Exception("unknown optimizer")

        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(
            loss=train_loss, global_step=tf.train.get_global_step())

        if params['use_tpu']:
            spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=train_loss,
                    train_op=train_op,
                    eval_metrics=(metric_fn, [labels, train_logits]))
        else:
            spec = tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=train_loss,
                    train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        if params['use_tpu']:
            spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=tf.estimator.ModeKeys.EVAL,
                    loss=train_loss,
                    eval_metrics=(metric_fn, [labels, train_logits]))
        else:
            spec = tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.EVAL,
                    loss=train_loss)
    if mode == tf.estimator.ModeKeys.PREDICT:
        pass # predict the thing

    return spec

