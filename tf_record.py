#!/usr/bin/env python
from __future__ import division
import os
import glob
import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
from scipy import misc
import time
import random
import math
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage.filters import gaussian_filter
import cv2

PI = tf.constant(math.pi)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def draw_grid(im, grid_size):
  """ Draws grid lines on input images
  Inputs: im - input image
          grid_size - number of vertical/horizontal grid lines
  Output: im - image with grid lines
  """
  shape = im.shape
  for i in range(0, shape[1], grid_size):
      cv2.line(im, (i, 0), (i, shape[0]), color=(0,0,0),thickness=2)
  for j in range(0, shape[0], grid_size):
      cv2.line(im, (0, j), (shape[1], j), color=(0,0,0),thickness=2)
     
  return im

def encode(img_filepath, mask_filepath, mean, std):

  image = np.load(img_filepath)
  mask = np.load(mask_filepath)

  # image = draw_grid(image, 50)

  img_raw = image.tostring()
  m_raw = mask.tostring()

  example = tf.train.Example(features=tf.train.Features(feature={

    'image_raw': _bytes_feature(img_raw),
    'mask_raw':_bytes_feature(m_raw),
    # 'mean':_float_feature(mean),
    # 'std':_float_feature(std)

  }))

  return example

def distort_brightness_constrast(image, ordering=0):
  if ordering == 0:
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
  else:
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
  return tf.clip_by_value(image, 0.0, 1.0)

def create_tf_record(tfrecords_filename, images_masks_stats):

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    # for i in tqdm(range(len(images_masks_stats))):

    #     image = np.load(images_masks_stats[i][0])
    #     mask = np.load(images_masks_stats[i][1])

    #     image = draw_grid(image, 50)

    #     if is_img_resize:              
    #         image = misc.imresize(image, (256, 256))
    #         mask = misc.imresize(mask, (256, 256))

    #     img_raw = image.tostring()
    #     m_raw = mask.tostring()

    #     example = tf.train.Example(features=tf.train.Features(feature={
    
    #             'image_raw': _bytes_feature(img_raw),
    #             'mask_raw':_bytes_feature(m_raw),

    #            }))

    #     writer.write(example.SerializeToString())

    # writer.close()

    with ProcessPoolExecutor(12) as executor:
      futures = [executor.submit(encode, i, m, mean, std) for i, m, mean, std in images_masks_stats]

      kwargs = {
          'total': len(futures),
          'unit': 'it',
          'unit_scale': True,
          'leave': True
      }

      for f in tqdm(as_completed(futures), **kwargs):
          pass
      print "Done loading futures!"
      print "Writing examples..."
      for i in tqdm(range(len(futures))):
        try:
            example = futures[i].result()
            writer.write(example.SerializeToString())
        except Exception as e:
            print "Failed to write example!"

    print '-' * 90
    print 'Generated tfrecord at %s' % tfrecords_filename
    print '-' * 90


def read_and_decode(filename_queue=None, img_dims=[512,512,3], size_of_batch=16,\
                     augmentations_dic=None, num_of_threads=2, shuffle=True):
   
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
    
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string),
        # 'mean': tf.FixedLenFeature([], tf.float64),
        # 'std': tf.FixedLenFeature([], tf.float64)
        
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    mask = tf.decode_raw(features['mask_raw'], tf.uint8)

    # mean = tf.decode_raw(features['mean'], tf.float64)
    # std = tf.decode_raw(features['std'],  tf.float64)

    image = tf.reshape(image, img_dims)
    mask = tf.reshape(mask, img_dims[:2])

    image = tf.to_float(image)
    mask = tf.to_float(mask)

    mask = tf.expand_dims(mask,-1)
    image = image/255
    image_mask = tf.concat([image, mask], axis=-1)

    if augmentations_dic['rand_flip_left_right']:
        image_mask = tf.image.random_flip_left_right(image_mask)
        image = image_mask[...,:3]
        mask = image_mask[...,3]
        mask = tf.expand_dims(mask,-1)

    image_mask = tf.concat([image, mask], axis=-1)
    if augmentations_dic['rand_flip_top_bottom']:     
        image_mask = tf.image.random_flip_up_down(image_mask)
        image = image_mask[...,:3]
        mask = image_mask[...,3]
        mask = tf.expand_dims(mask,-1) 

    if augmentations_dic['rand_rotate']:
        elems = tf.convert_to_tensor([0, PI/2, PI, (3*PI)/2])
        sample = tf.squeeze(tf.multinomial(tf.log([[0.25, 0.25, 0.25, 0.25]]), 1)) 
        random_angle = elems[tf.cast(sample, tf.int32)]
        image = tf.contrib.image.rotate(image, random_angle)
        mask = tf.contrib.image.rotate(mask, random_angle)
    
    if shuffle:
        image, mask = tf.train.shuffle_batch([image, mask],
                                           batch_size=size_of_batch,
                                           capacity=100 + 3 * size_of_batch,
                                           min_after_dequeue=100,
                                           num_threads=num_of_threads)
    else:
        image, mask = tf.train.batch([image, mask],
                                   batch_size=size_of_batch,
                                   capacity=100,
                                   allow_smaller_final_batch=True,
                                   num_threads=num_of_threads)
    
    if augmentations_dic['warp']:
        X = tf.random_uniform([img_dims[0], img_dims[1]])*2 - 1
        Y = tf.random_uniform([img_dims[0], img_dims[1]])*2 - 1
        X = tf.reshape(X, [1, img_dims[0],img_dims[1], 1])
        Y = tf.reshape(Y, [1, img_dims[0],img_dims[1], 1])

        mean = 0.0
        sigma = 1.0
        alpha = 20.0
        ksize = 256

        x = tf.linspace(-3.0, 3.0, ksize)
        z = ((1.0 / (sigma * tf.sqrt(2.0 * PI))) * tf.exp(tf.negative(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))))
        ksize = z.get_shape().as_list()[0]
        z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
        z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])

        X_convolved = tf.nn.conv2d(X, z_4d, strides=[1, 1, 1, 1], padding='SAME')
        Y_convolved = tf.nn.conv2d(Y, z_4d, strides=[1, 1, 1, 1], padding='SAME')

        X_convolved = (X_convolved / tf.reduce_max(X_convolved))*alpha
        Y_convolved = (Y_convolved / tf.reduce_max(Y_convolved))*alpha

        trans = tf.stack([X_convolved,Y_convolved], axis=-1)
        trans = tf.reshape(trans, [-1])

        batch_trans = tf.tile(trans, [size_of_batch])
        batch_trans = tf.reshape(batch_trans, [size_of_batch, img_dims[0], img_dims[1] ,2])

        image = tf.reshape(image, [size_of_batch, img_dims[0], img_dims[1], img_dims[2]])
        mask = tf.reshape(mask, [size_of_batch, img_dims[0], img_dims[1], 1])

        image = tf.contrib.image.dense_image_warp(image, batch_trans)
        mask = tf.contrib.image.dense_image_warp(mask, batch_trans)

    if augmentations_dic['grayscale']:
        image = tf.image.rgb_to_grayscale(image)

    if augmentations_dic['distort_brightness_constrast']:
        elems = tf.convert_to_tensor([0, 1])
        sample = tf.squeeze(tf.multinomial(tf.log([[0.25, 0.25]]), 1)) 
        rand_int = elems[tf.cast(sample, tf.int32)]
        image = distort_brightness_constrast(image, ordering=rand_int)

    mask = tf.to_int64(mask)

    return image, mask
