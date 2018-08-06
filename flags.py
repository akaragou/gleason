#!/usr/bin/python3
"""runs a model on some data."""

import tensorflow as tf

def get():
    """get the tensorflow flags."""
    tf.flags.DEFINE_string(
            "config",
            help="",
            default="configs/default.conf")
    tf.flags.DEFINE_string(
            "tpu",
            help="",
            default=None)
    tf.flags.DEFINE_string(
            "tpu_zone",
            help="",
            default="us-central1-f")
    tf.flags.DEFINE_string(
            "gcp_project",
            help="",
            default="beyond-dl-1503610372419")
    tf.flags.DEFINE_string(
            "data_dir",
            help="",
            default="")
    tf.flags.DEFINE_string(
            "model_dir",
            help="",
            default=None)
    tf.flags.DEFINE_bool(
            "use_tpu",
            help="",
            default=True)
    tf.flags.DEFINE_integer(
            "iterations",
            help="",
            default=50)
    tf.flags.DEFINE_integer(
            "num_shards",
            help="",
            default=8)
    tf.flags.DEFINE_integer(
            "batch_size",
            help="",
            default=1024)
    tf.flags.DEFINE_integer(
            "train_steps",
            help="",
            default=32)
    tf.flags.DEFINE_integer(
            "eval_steps",
            help="",
            default=5000)
    tf.flags.DEFINE_string(
            "mode",
            help="",
            default='train')
    
    return tf.flags.FLAGS

