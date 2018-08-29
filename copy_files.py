from __future__ import division
import os
import glob
import shutil
import random
from tqdm import tqdm
import numpy as np
import concurrent.futures
from sklearn.utils import shuffle


def copy_train(file_name):

    if file_name.split('_')[3] == 'mask' or file_name.split('_')[3]+ '_' + file_name.split('_')[4] == 'normal_mask':
        all_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'
    else:
        all_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/imgs/'
    
    shutil.copyfile(all_data_path + file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/train/'+file_name)
    
    return file_name

def copy_val(file_name):

    if file_name.split('_')[3] == 'mask' or file_name.split('_')[3]+ '_' + file_name.split('_')[4] == 'normal_mask':
        all_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'
    else:
        all_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/imgs/'
    
    shutil.copyfile(all_data_path + file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/val/'+file_name)
    
    return file_name

def copy_test(file_name):

    if file_name.split('_')[3] == 'mask' or file_name.split('_')[3]+ '_' + file_name.split('_')[4] == 'normal_mask':
        all_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'
    else:
        all_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/imgs/'
    
    shutil.copyfile(all_data_path + file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/test/'+file_name)
    
    return file_name

def copy_all_imgs():

    image_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/imgs/'
    mask_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'

    images_full_path = glob.glob(image_data_path + '*.npy')
    mask_full_path = glob.glob(mask_data_path + '*.npy')

    all_data_path = images_full_path + mask_full_path

    all_data = []
    for i in range(len(all_data_path)):
        file_name = all_data_path[i].split('/')[-1]
        all_data.append(file_name) 

    train = []
    val = []
    test = []

    moffit_ids = ['2290506', '2290660', 'moffitt10', 'moffitt11', 'moffitt13', 'moffitt14', 'moffitt15', 'moffitt16', 'moffitt17', 
                    'moffitt18', 'moffitt4', 'moffitt5', 'moffitt6', 'moffitt7', 'moffitt8', 'moffitt9']

    val_ids = ['TMH0014A', 'TMH0014F', 'TMH0018F', 'TMH0019H', 'TMH0020C', 'TMH0034G', 'TMH0041D', 'TMH0069L', 'TMH0070L']

    test_ids = ['TMH0014I', 'TMH0018E', 'TMH0019G', 'TMH0024E', 'TMH0025E', 'TMH0034D', 'TMH0041E', 'TMH0069J', 'TMH0070K']

    not_train = moffit_ids + val_ids + test_ids

    for i in range(len(all_data)):

        file_id = all_data[i].split('_')[0]

        if file_id in test_ids:
            test.append(all_data[i])
        elif file_id in val_ids:
            val.append(all_data[i])
        elif file_id not in not_train:
            train.append(all_data[i])

    print "moving train images..."
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        executor.map(copy_train, train)
    print "done moving train images!"

    print "moving val images..."
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        executor.map(copy_val, val)
    print "done moving val images!"

    print "moving test images..."
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        executor.map(copy_test, test)

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

def get_unique_ids():

    data_path = '/media/data_cifs/andreas/pathology/gleason_training/test/'

    images_full_path = glob.glob(data_path + '*.npy')
    img_ids = []

    for i in range(len(images_full_path)):
        id_name = images_full_path[i].split('/')[-1].split('_')[0]
        img_ids.append(id_name) 

    img_ids = np.array(img_ids)
    img_ids = np.unique(img_ids)

    for i_id in img_ids:
        print i_id


if __name__ == '__main__':
    get_unique_ids()