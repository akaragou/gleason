#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
from layers.feedforward import conv


def build_model(data_tensor, reuse, training, output_channels):
    """Create the hgru from Learning long-range..."""
    conv_kernel = [
        [3, 3, 1],
        [3, 3, 3],
        [3, 3, 3],
    ]
    up_kernel = [2, 2, 1]
    # filters = [28, 36, 48, 64, 80]
    filters = [24, 32, 48, 64, 80]
    print(data_tensor.dtype)
    with tf.variable_scope('cnn', reuse=reuse):
        # Unclear if we should include l0 in the down/upsample cascade
        with tf.variable_scope('in_embedding', reuse=reuse):
            in_emb = conv.conv3d_layer(
                bottom=data_tensor,
                name='l0',
                stride=[1, 1, 1],
                padding='SAME',
                num_filters=filters[0],
                kernel_size=[5, 5, 1],
                trainable=training,
                use_bias=True)
            in_emb = tf.nn.elu(in_emb)
    
        print(in_emb.dtype)

        # Downsample
        l1 = conv.down_block(
            layer_name='l1',
            bottom=in_emb,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse)
        print(l1.dtype)
        l2 = conv.down_block(
            layer_name='l2',
            bottom=l1,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            reuse=reuse)
        print(l2.dtype)
        l3 = conv.down_block(
            layer_name='l3',
            bottom=l2,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            reuse=reuse)
        print(l3.dtype)
        l4 = conv.down_block(
            layer_name='l4',
            bottom=l3,
            kernel_size=conv_kernel,
            num_filters=filters[4],
            training=training,
            reuse=reuse)
        print(l4.dtype)

        # Upsample

        ul3 = conv.up_block(
            layer_name='ul3',
            bottom=l4,
            skip_activity=l3,
            kernel_size=up_kernel,
            num_filters=filters[3],
            training=training,
            reuse=reuse)
        ul3 = conv.down_block(
            layer_name='ul3_d',
            bottom=ul3,
            kernel_size=conv_kernel,
            num_filters=filters[3],
            training=training,
            reuse=reuse,
            include_pool=False)
        ul2 = conv.up_block(
            layer_name='ul2',
            bottom=ul3,
            skip_activity=l2,
            kernel_size=up_kernel,
            num_filters=filters[2],
            training=training,
            reuse=reuse)
        ul2 = conv.down_block(
            layer_name='ul2_d',
            bottom=ul2,
            kernel_size=conv_kernel,
            num_filters=filters[2],
            training=training,
            reuse=reuse,
            include_pool=False)
        ul1 = conv.up_block(
            layer_name='ul1',
            bottom=ul2,
            skip_activity=l1,
            kernel_size=up_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse)
        ul1 = conv.down_block(
            layer_name='ul1_d',
            bottom=ul1,
            kernel_size=conv_kernel,
            num_filters=filters[1],
            training=training,
            reuse=reuse,
            include_pool=False)
        ul0 = conv.up_block(
            layer_name='ul0',
            bottom=ul1,
            skip_activity=in_emb,
            kernel_size=up_kernel,
            num_filters=filters[0],
            training=training,
            reuse=reuse)

        with tf.variable_scope('out_embedding', reuse=reuse):
            out_emb = conv.conv3d_layer(
                bottom=ul0,
                name='out_emb',
                stride=[1, 1, 1],
                padding='SAME',
                num_filters=output_channels,
                kernel_size=[5, 5, 1],
                trainable=training,
                use_bias=True)
    return out_emb


