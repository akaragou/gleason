#!/usr/bin/python3

import functools
import os
import tensorflow as tf

class Reader():
    """Reader reads from a tfrecord file to produce an image."""

    def __init__(self, data_dir, img_dims=[512,512,3], augmentations={}, shuffle=True):
        """initialize the reader with a tfrecord dir and dims."""
        self.data_dir = data_dir
        self.img_dims = img_dims
        self.augmentations = augmentations
        self.shuffle = True

    def dataset_parser(self, value):
        """parse the tfrecords."""

        keys_to_features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
        }

        PI = tf.constant(3.1415926535, dtype=tf.float32)
        
        parsed = tf.parse_single_example(value, keys_to_features)

        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        mask = tf.decode_raw(parsed['mask_raw'], tf.uint8)

        image = tf.reshape(image, self.img_dims)
        mask = tf.reshape(mask, self.img_dims[:2])

        image = tf.image.rgb_to_grayscale(image)

        image = tf.to_float(image)
        mask = tf.to_float(mask)

        mask = tf.expand_dims(mask,-1) 
        image_mask = tf.concat([image, mask], axis=-1)
        image_mask = tf.random_crop(image_mask, [384, 384, 2])

        image = tf.expand_dims(image_mask[:,:,0], -1)
        mask = tf.expand_dims(image_mask[:,:,1], -1)

        if 'rand_flip_left_right' in self.augmentations:
            image_mask = tf.image.random_flip_left_right(image_mask)
            image = image_mask[...,:3]
            mask = image_mask[...,3]
            mask = tf.expand_dims(mask,-1)

        image_mask = tf.concat([image, mask], axis=-1)
        if 'rand_flip_top_bottom' in self.augmentations:
            image_mask = tf.image.random_flip_up_down(image_mask)
            image = image_mask[...,:3]
            mask = image_mask[...,3]
            mask = tf.expand_dims(mask,-1) 

        if 'rand_rotate' in self.augmentations:
            elems = tf.convert_to_tensor([0, PI/2, PI, (3*PI)/2])
            sample = tf.squeeze(tf.multinomial(tf.log([[0.25, 0.25, 0.25, 0.25]]), 1)) 
            random_angle = elems[tf.cast(sample, tf.int32)]
            image = tf.contrib.image.rotate(image, random_angle)
            mask = tf.contrib.image.rotate(mask, random_angle)
        

        if 'warp' in self.augmentations:
            X = tf.random_uniform([self.img_dims[0], self.img_dims[1]])*2 - 1
            Y = tf.random_uniform([self.img_dims[0], self.img_dims[1]])*2 - 1
            X = tf.reshape(X, [1, self.img_dims[0],self.img_dims[1], 1])
            Y = tf.reshape(Y, [1, self.img_dims[0],self.img_dims[1], 1])

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

            #batch_trans = tf.tile(trans, [size_of_batch])
            #batch_trans = tf.reshape(batch_trans, [size_of_batch, img_dims[0], img_dims[1] ,2])

            #image = tf.reshape(image, [size_of_batch, img_dims[0], img_dims[1], img_dims[2]])
            #mask = tf.reshape(mask, [size_of_batch, img_dims[0], img_dims[1], 1])

            # image = tf.contrib.image.dense_image_warp(image, batch_trans)
            # mask = tf.contrib.image.dense_image_warp(mask, batch_trans)

        image = image/255
        mask = tf.cast(mask, dtype=tf.int64)

        return image, mask

    def input_fn(self, params):
        """input function provides a single batch for train or eval."""
        batch_size = params['batch_size']
        is_training = params['train']

        file_pattern = os.path.join(
                self.data_dir, 'train-*' if is_training else 'validation-*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
        
        if is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 1024*1024*8
            return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)

        dataset = dataset.apply(
                tf.contrib.data.parallel_interleave(
                    fetch_dataset, cycle_length=64, sloppy=True))

        if self.shuffle:
            dataset = dataset.shuffle(1024)
        
        dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    self.dataset_parser, batch_size=batch_size,
                    num_parallel_batches=8, drop_remainder=True))
        
        dataset = dataset.prefetch(32)
        
        return dataset

#
# def read_and_decode(filename_queue=None, img_dims=[512,512,3], size_of_batch=16,\
#                     augmentations_dic=None, num_of_threads=2, shuffle=True):
   
