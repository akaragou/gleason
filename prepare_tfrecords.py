#!/usr/bin/env python
from __future__ import division
import argparse
import tensorflow as tf
import glob
import numpy as np
from scipy import misc
import os
from tfrecord import create_tf_record
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat
from tqdm import tqdm 
import random

def calculate_class_ratios(main_data_dir, dataset):
    """
    Calculate class and save weights for different Gleason gradings
    Inputs: main_data_dir - main directory for project
            dataset - dataset selected, options are train, val and test
    """
    files = glob.glob(os.path.join(main_data_dir, dataset) + '/*.png')

    targe_label_dic = {0:0 ,1:0, 2:0, 3:0}

    for img in files:

        file_name = img.split('/')[-1].split('_')
        class_name = file_name[0].lower() + '_' + file_name[1] 

        if class_name == "gleason_5":
            targe_label_dic[3] += 1
        elif class_name == "gleason_4":
            targe_label_dic[2] += 1
        elif class_name == "gleason_3":
            targe_label_dic[1] += 1
        else:
            targe_label_dic[0] += 1

    values = [targe_label_dic[k] for k in range(4)]
    f_i = [sum(values)/v for v in values]
    class_weights = [f/sum(f_i) for f in f_i]
    print "class ratios:", [v/sum(values) for v in values]
    print "class weights:", class_weights
    np.save(dataset + '_class_weights.npy', class_weights)


def build_tfrecords(main_data_dir, main_tfrecords_dir, dataset):
    """
    creating tfrecords for image patches
    Inputs: main_data_dir - main directory for project
            main_tfrecords_dir - main directory to store tfrecords in
            dataset - dataset selected, options are train, val and test
    """
    files = glob.glob(os.path.join(main_data_dir, dataset) + '/*.png')
    
    target_labels = [] 

    for img in files:

        file_name = img.split('/')[-1].split('_')
        class_name = file_name[0].lower() + '_' + file_name[1] 

        if class_name == "gleason_5":
            target_labels.append(3)
        elif class_name == "gleason_4":
            target_labels.append(2)
        elif class_name == "gleason_3":
            target_labels.append(1)
        else:
            target_labels.append(0)

    create_tf_record(os.path.join(main_tfrecords_dir, dataset +'.tfrecords'), files, target_labels)

if __name__ == '__main__':
    main_data_dir = '/media/data_cifs/andreas/pathology/gleason_training_patches/'
    main_tfrecords_dir = '/media/data_cifs/andreas/pathology/gleason_training_patches/tfrecords'

    build_tfrecords(main_data_dir, main_tfrecords_dir, 'train')
    build_tfrecords(main_data_dir, main_tfrecords_dir, 'val')
    build_tfrecords(main_data_dir, main_tfrecords_dir, 'test')
