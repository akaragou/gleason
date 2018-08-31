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

def match_npy(filepaths, return_masks_only=False):

    slides = {}
    masks = {}

    for i in range(len(filepaths)):

        file_name = filepaths[i].split('/')[-1]

        if file_name.split('_')[3] == 'mask' or (file_name.split('_')[3] + '_' +file_name.split('_')[4] == 'normal_mask'):
            if file_name.split('_')[-4] == 'anno':
                m_key = file_name.split('_')[0] + '_' + file_name.split('_')[-3] + '_' + file_name.split('_')[-2] + '_' + file_name.split('_')[-1].split('.')[0]
            else:
                m_key = file_name.split('_')[0] + '_' + file_name.split('_')[-2]  + '_' + file_name.split('_')[-1].split('.')[0]
            masks[m_key] = filepaths[i]
        else:
            if file_name.split('_')[-4] == 'anno':
                s_key = file_name.split('_')[0] + '_' + file_name.split('_')[-3] + '_' + file_name.split('_')[-2] + '_' + file_name.split('_')[-1].split('.')[0]
            else:
                s_key = file_name.split('_')[0] + '_' + file_name.split('_')[-2] + '_' + file_name.split('_')[-1].split('.')[0]
            slides[s_key] = filepaths[i]
    
    print len(masks.keys())
    print len(slides.keys())

    if return_masks_only:
        return masks.values()

    matched = []
    for key in slides.keys():
        matched.append((slides[key],masks[key]))

    return matched


def match_img(imgs, img_filepaths, mask_filepaths):

    matched = []

    for full_path in imgs:
        file_name = full_path.split('/')[-1].split('.')[0]
        matched.append((img_filepaths + file_name + '.jpg', mask_filepaths + file_name + '.png'))

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

def build_tfrecords():

    main_data_dir = '/media/data_cifs/andreas/pathology/gleason_training/'
    tf_record_file_path = '/media/data_cifs/andreas/pathology/gleason_training/tfrecords/'

    train_file_paths = glob.glob(os.path.join(main_data_dir, 'train') + '/*.npy')
    val_file_paths =  glob.glob(os.path.join(main_data_dir, 'val') + '/*.npy')
    test_file_paths =  glob.glob(os.path.join(main_data_dir, 'test') + '/*.npy')


    train_slides_masks = match_npy(train_file_paths)
    print len(train_slides_masks)
    print "Creating Train tfrecords..."
    calculate_class_weights(match_npy(train_file_paths, True), 'train')
    create_tf_record(os.path.join(tf_record_file_path, 'train.tfrecords'), train_slides_masks)

    print
    print "Creating Val tfrecords..."
    val_slides_masks = match_npy(val_file_paths)
    print len(val_slides_masks)
    create_tf_record(os.path.join(tf_record_file_path, 'val.tfrecords'), val_slides_masks)

    print   
    print "Creating Test tfrecords..."
    test_slides_masks = match_npy(test_file_paths)
    print len(test_slides_masks)
    create_tf_record(os.path.join(tf_record_file_path, 'test.tfrecords'), test_slides_masks)

def build_tfrecords_gleason_pretraining():

    main_data_dir = '/media/data_cifs/andreas/pathology/gleason_training/'      
    tf_record_file_path = '/media/data_cifs/andreas/pathology/gleason_training/tfrecords/'

    train_file_paths = glob.glob(os.path.join(main_data_dir, 'pretraining_gleason_train') + '/*.npy')
    val_file_paths =  glob.glob(os.path.join(main_data_dir, 'pretraining_gleason_val') + '/*.npy')

    print "Creating Train tfrecords..."
    train_slides_masks = match_npy(train_file_paths)
    print len(train_slides_masks)
    create_tf_record(os.path.join(tf_record_file_path, 'pretraining_gleason_train.tfrecords'), train_slides_masks)

    print
    print "Creating Val tfrecords..."
    val_slides_masks = match_npy(val_file_paths)
    print len(val_slides_masks)
    create_tf_record(os.path.join(tf_record_file_path, 'pretraining_gleason_val.tfrecords'), val_slides_masks)

def build_tfrecords_coco_pretraining():

    main_data_dir = '/media/data_cifs/andreas'      
    tf_record_file_path = '/media/data_cifs/andreas/pathology/gleason_training/tfrecords/'

    train_imgs = glob.glob(os.path.join(main_data_dir, 'coco_train2017') + '/*.jpg')
    val_imgs = glob.glob(os.path.join(main_data_dir, 'coco_val2017') + '/*.jpg')

    print "Creating Train tfrecords..."
    train_slides_masks = match_img(train_imgs, os.path.join(main_data_dir, 'coco_train2017/'), os.path.join(main_data_dir, 'coco_stuffthingmaps_trainval2017/train2017/'))
    create_tf_record(os.path.join(tf_record_file_path, 'pretraining_coco_train.tfrecords'), train_slides_masks, file_type='img')

    print
    print "Creating Val tfrecords..."
    val_slides_masks = match_img(val_imgs, os.path.join(main_data_dir, 'coco_val2017/'), os.path.join(main_data_dir, 'coco_stuffthingmaps_trainval2017/val2017/'))
    create_tf_record(os.path.join(tf_record_file_path, 'pretraining_coco_val.tfrecords'), val_slides_masks, file_type='img')

if __name__ == '__main__':

    build_tfrecords_gleason_pretraining()
    # build_tfrecords_coco_pretraining()
    