#!/usr/bin/env python
from __future__ import division
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from tf_record import create_tf_record
from tqdm import tqdm
import csv

def match(filepaths):

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
    # print masks
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

def calculate_class_weights(mask_filepaths):

    print "Calculating class weights..."
    class_dic = {}

    for i in tqdm(range(len(mask_filepaths))):
        mask = np.load(mask_filepaths[i][1])
        flattened_masks = np.ravel(mask)
        unique, counts = np.unique(flattened_masks, return_counts=True)

        for u, c in zip(unique, counts):

            if u not in class_dic:
                class_dic[u] = c
            else:
                class_dic[u] += c

    sorted_class_keys = sorted(class_dic)
    print sorted_class_keys
    values = [class_dic[k] for k in sorted_class_keys]
    print values
    f_i = map(lambda x: sum(values)/x, values)
    print f_i
    class_weights = np.array(map(lambda x: x/sum(f_i), f_i), dtype=np.float32)
    print class_weights
    # f_i = map(lambda x: len(flattened_masks)/x, 
    #                         class_dic.values())
    # class_weights =np.array(map(lambda x: x/sum(f_i), f_i), dtype=np.float32)
    # np.save('class_weights_Berson.npy', class_weights)
    print "Done calculating class weights!"

def build_tfrecords(main_data_dir):

    tf_record_file_path = os.path.join(main_data_dir, 'tfrecords')

    train_file_paths = glob.glob(os.path.join(main_data_dir, 'train') + '/*.npy')
    val_file_paths =  glob.glob(os.path.join(main_data_dir, 'val') + '/*.npy')
    test_file_paths =  glob.glob(os.path.join(main_data_dir, 'test') + '/*.npy')

    # print "Creating Train tfrecords..."
    # train_slides_masks = match(train_file_paths)
    # create_tf_record(os.path.join(tf_record_file_path, 'train.tfrecords'), train_slides_masks)

    print
    print "Creating Val tfrecords..."
    val_slides_masks = match(val_file_paths)
    # calculate_class_weights(val_slides_masks)
    create_tf_record(os.path.join(tf_record_file_path, 'val.tfrecords'), val_slides_masks)

    # print   
    print "Creating Test tfrecords..."
    test_slides_masks = match(test_file_paths)
    create_tf_record(os.path.join(tf_record_file_path, 'test.tfrecords'), test_slides_masks)
    
if __name__ == '__main__':

    main_data_dir = '/media/data_cifs/andreas/pathology/gleason_training'
    build_tfrecords(main_data_dir)
    