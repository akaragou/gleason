#!/usr/bin/python3
"""runs a model on some data."""

from config import Config
import os
import tensorflow as tf
#from tensorflow.contrib.tpu.python.tpu import bfloat16
import time
import flags
import records
import models
import model

def main():
    """run the model."""
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # get the configuration for the current run.
    FLAGS = flags.get()
    config = Config(FLAGS.config)

    datareader = records.Reader(FLAGS.data_dir)
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project)

    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations,
                num_shards=FLAGS.num_shards))

    params = {
            'data_dir': FLAGS.data_dir,
            'train': True,
            'use_tpu': FLAGS.use_tpu,
            'model_name': config['model']['name'],
            'num_classes': config['model']['num_classes'],
            'alpha': config['train']['alpha'],
            'gamma': config['train']['gamma'],
            'epsilon': 1e-8,
    }
    estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model.model_fn,
            use_tpu=FLAGS.use_tpu,
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size,
            params=params,
            config=run_config)

    start_time = time.time()
    estimator.train(input_fn=datareader.input_fn,
            max_steps=FLAGS.train_steps)
    elapsed_time = int(time.time() - start_time)
    tf.logging.info('Finished training up to step %d in %d seconds.' % (FLAGS.train_steps, elapsed_time))

    
if __name__ == '__main__':
    main()
