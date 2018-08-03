#!/usr/bin/env python
from __future__ import division
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from tf_record import create_tf_record
from tqdm import tqdm

def match(filepaths):

    slides = []
    masks = []

    for i in range(len(filepaths)):

        file_name = filepaths[i].split('/')[-1]
        file_type = file_name.split('_')[1]

        if file_type == 'slide':
            slides.append(filepaths[i])
        else:
            masks.append(filepaths[i])

    matched = []
    for i in tqdm(range(len(masks))):
        for j in range(len(slides)):

            s_file_name = slides[j].split('/')[-1]
            s_key = s_file_name.split('_')[0] + '_' + s_file_name.split('_')[3] + '_' + s_file_name.split('_')[4] 

            m_file_name = masks[i].split('/')[-1]
            m_key = m_file_name.split('_')[0] + '_' + m_file_name.split('_')[4] + '_' + m_file_name.split('_')[5] 

            if s_key == m_key:
                matched.append((slides[j],masks[i]))

    return matched

def build_tfrecords(main_data_dir):

    tf_record_file_path = os.path.join(main_data_dir, 'tfrecords')

    train_file_paths = glob.glob(os.path.join(main_data_dir, 'train2') + '/*.npy')
    val_file_paths =  glob.glob(os.path.join(main_data_dir, 'val2') + '/*.npy')
    test_file_paths =  glob.glob(os.path.join(main_data_dir, 'test2') + '/*.npy')

    train_slides_masks = match(train_file_paths)
    create_tf_record(os.path.join(tf_record_file_path, 'train2.tfrecords'), train_slides_masks, model_img_dims=[512,512], is_img_resize = False)

    val_slides_masks = match(val_file_paths)
    create_tf_record(os.path.join(tf_record_file_path, 'val2.tfrecords'), val_slides_masks, model_img_dims=[512,512], is_img_resize = False)

    test_slides_masks = match(test_file_paths)
    create_tf_record(os.path.join(tf_record_file_path, 'test2.tfrecords'), test_slides_masks, model_img_dims=[512,512], is_img_resize = False)
    
if __name__ == '__main__':

    main_data_dir = '/media/data_cifs/andreas/pathology/gleason_training'
    build_tfrecords(main_data_dir)
    