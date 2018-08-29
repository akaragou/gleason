#!/usr/bin/env python
from __future__ import division
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from tf_record import create_tf_record
from tqdm import tqdm
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

def match(filepaths, return_masks_only=False):

    slides = []
    masks = []

    for i in range(len(filepaths)):

        file_name = filepaths[i].split('/')[-1]
        file_type = file_name.split('_')[3]

        if file_name.split('_')[3] == 'mask' or file_name.split('_')[3]+ '_' + file_name.split('_')[4] == 'normal_mask':
            masks.append(filepaths[i])
        else:
            slides.append(filepaths[i])
    
    print len(masks)
    print len(slides)

    if return_masks_only:
        return masks

    matched = []
    for i in tqdm(range(len(masks))):
        for j in range(len(slides)):

            s_file_name = slides[j].split('/')[-1]
            s_key = s_file_name.split('_')[0] + '_' + s_file_name.split('_')[-2] + '_' + s_file_name.split('_')[-1]

            m_file_name = masks[i].split('/')[-1]
            m_key = m_file_name.split('_')[0] + '_' + m_file_name.split('_')[-2]  + '_' + m_file_name.split('_')[-1]

            if s_key == m_key:
                matched.append((slides[j],masks[i]))
    return matched

def populate_dic(class_dic, mask_filepath):

    mask = np.load(mask_filepath)
    flattened_masks = np.ravel(mask)
    unique, counts = np.unique(flattened_masks, return_counts=True)
    for u, c in zip(unique, counts):
        class_dic[u] += c
    return class_dic

def calculate_class_weights(mask_filepaths, data_set):

    print "Calculating class weights..."
    class_dic = {0:0, 1:0, 2:0, 3:0, 4:0}
    class_dic_list = [class_dic for _ in range(len(mask_filepaths))]

    results_dic = {0:0, 1:0, 2:0, 3:0, 4:0}
    with ProcessPoolExecutor(16) as executor:
        futures = [executor.submit(populate_dic, class_dic, m) for class_dic, m in zip(class_dic_list, mask_filepaths)]
        kwargs = {
          'total': len(futures),
          'unit': 'it',
          'unit_scale': True,
          'leave': True
        }
        for f in tqdm(as_completed(futures), **kwargs):
            pass
        print "Done loading futures!"
        for i in tqdm(range(len(futures))):
            try:
                result = (futures[i].result())
                for i in range(5):
                    results_dic[i] += result[i]
            except Exception as e:
                print "Failed to count labels"

    values = [results_dic[k] for k in range(5)]
    f_i = [sum(values)/v for v in values]
    class_weights = [f/sum(f_i) for f in f_i]
    print [v/sum(values) for v in values]
    print class_weights
    np.save(data_set + '_class_weights.npy', class_weights)


def build_tfrecords(main_data_dir):

    tf_record_file_path = os.path.join(main_data_dir, 'tfrecords')

    train_file_paths = glob.glob(os.path.join(main_data_dir, 'updated_train') + '/*.npy')
    val_file_paths =  glob.glob(os.path.join(main_data_dir, 'updated_val') + '/*.npy')
    test_file_paths =  glob.glob(os.path.join(main_data_dir, 'updated_test') + '/*.npy')

    print "Creating Train tfrecords..."
    train_slides_masks = match(train_file_paths)
    calculate_class_weights(match(train_file_paths, True), 'updated_train')
    create_tf_record(os.path.join(tf_record_file_path, 'train.tfrecords'), train_slides_masks)

    print
    print "Creating Val tfrecords..."
    val_slides_masks = match(val_file_paths)
    create_tf_record(os.path.join(tf_record_file_path, 'val.tfrecords'), val_slides_masks)

    print   
    print "Creating Test tfrecords..."
    test_slides_masks = match(test_file_paths)
    create_tf_record(os.path.join(tf_record_file_path, 'test.tfrecords'), test_slides_masks)
    
if __name__ == '__main__':

    main_data_dir = '/media/data_cifs/andreas/pathology/gleason_training'
    build_tfrecords(main_data_dir)
    