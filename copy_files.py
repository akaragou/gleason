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
    
    shutil.copyfile('/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'+file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/train/'+file_name)
    
    return file_name

def copy_val(file_name):
    
    shutil.copyfile('/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'+file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/val/'+file_name)
    
    return file_name

def copy_test(file_name):
    
    shutil.copyfile('/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'+file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/test/'+file_name)
    
    return file_name

def copy_all_imgs():

    image_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/imgs/'
    mask_data_path = '/media/data_cifs/andreas/pathology/miriam/training_crops/masks/'

    images_full_path = glob.glob(mask_data_path + '*.npy')
    all_images = []

    for i in range(len(images_full_path)):

        img_name = images_full_path[i].split('/')[-1]
        all_images.append(img_name) 

    train_images = []
    val_images = []
    test_images = []

    val_ids = ['TMH0014A', 'TMH0014F', 'TMH0018B', 'TMH0019B', 'TMH0019J', 
                'TMH0020C', 'TMH0069L', 'TMH0070M-2', 'moffitt15', 'moffitt6']

    test_ids = ['TMH0018E', 'TMH0019G', 'TMH0024E', 'TMH0025E', 'TMH0034D', 
                'TMH0034G', 'TMH0069C', 'moffitt14', 'moffitt16', 'moffitt4']

    for i in range(len(all_images)):

        img_id = all_images[i].split('_')[0]

        if img_id in test_ids:
            test_images.append(all_images[i])
        elif img_id in val_ids:
            val_images.append(all_images[i])
        else:
            train_images.append(all_images[i])


    print "moving train images..."
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        executor.map(copy_train, train_images)
    print "done moving train images!"

    print "moving val images..."
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        executor.map(copy_val, val_images)
    print "done moving val images!"

    print "moving test images..."
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        executor.map(copy_test, test_images)

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