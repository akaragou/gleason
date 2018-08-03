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
    
    shutil.copyfile('/media/data_cifs/andreas/pathology/miriam/gleason_img_crops/'+file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/train2/'+file_name)
    
    return file_name

def copy_val(file_name):
    
    shutil.copyfile('/media/data_cifs/andreas/pathology/miriam/gleason_img_crops/'+file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/val2/'+file_name)
    
    return file_name

def copy_test(file_name):
    
    shutil.copyfile('/media/data_cifs/andreas/pathology/miriam/gleason_img_crops/'+file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/test2/'+file_name)
    
    return file_name



# def copy_all_imgs():

#     image_data_path = '/media/data_cifs/andreas/pathology/miriam/gleason_img_crops/'
#     mask_data_path = '/media/data_cifs/andreas/pathology/miriam/gleason_mask_crops/'

#     images_full_path = glob.glob(image_data_path + '*.npy')
#     all_images = []

#     for i in range(len(images_full_path)):

#         img_name = images_full_path[i].split('/')[-1]
#         all_images.append(img_name) 


#     train_images = []
#     val_images = []
#     test_images = []

#     val_ids = ['TMH0034D','TMH0041C','TMH0045H','TMH0020D','TMH0019G','TMH0025E']

#     test_ids = ['TMH0023C','TMH0024D','TMH0024E','TMH0034J','TMH0043B',
#                 'TMH0045F','TMH0019G','TMH0018F','TMH0012C','TMH0023D']



#     for i in range(len(all_images)):

#         img_id = all_images[i].split('_')[0]


#         if img_id in test_ids:
#             test_images.append(all_images[i])
#         elif img_id in val_ids:
#             val_images.append(all_images[i])
#         else:
#             train_images.append(all_images[i])


#     print "moving train images..."
#     with concurrent.futures.ProcessPoolExecutor(16) as executor:
#         executor.map(copy_train, train_images)
#     print "done moving train images!"

#     print "moving val images..."
#     with concurrent.futures.ProcessPoolExecutor(16) as executor:
#         executor.map(copy_val, val_images)
#     print "done moving val images!"

#     print "moving test images..."
#     with concurrent.futures.ProcessPoolExecutor(16) as executor:
#         executor.map(copy_test, test_images)

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

def copy(file_name):
    
    shutil.copyfile('/media/data_cifs/andreas/pathology/gleason_training/test/'+file_name, 
                    '/media/data_cifs/andreas/pathology/gleason_training/test2/'+file_name)
    
    return file_name

def copy_imgs():

    old_data_path = '/media/data_cifs/andreas/pathology/gleason_training/test/'
    images_full_path = glob.glob(old_data_path + '*.npy')
    old_images = []
    for i in range(len(images_full_path)):
        img_name = images_full_path[i].split('/')[-1]
        old_images.append(img_name) 


    new_data_path = '/media/data_cifs/andreas/pathology/gleason_training/test2/'
    images_full_path = glob.glob(new_data_path + '*.npy')
    new_images = []
    for i in range(len(images_full_path)):
        img_name = images_full_path[i].split('/')[-1]
        new_images.append(img_name) 


    potential_move = []
    for img in old_images:

        if img not in new_images:
            potential_move.append(img)

    potential_move_matched = match(potential_move)

    count = 0

    to_move_matched = []
    for img in potential_move_matched:

        to_move_matched.append(img)
        count += 1
        if count == 500:
            break

  
    to_move = []
    for m in to_move_matched:
        to_move.append(m[0])
        to_move.append(m[1])
    print to_move
    print len(to_move)


    print "moving ..."
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        executor.map(copy, to_move)
    print "done moving!"
        

if __name__ == '__main__':
    copy_imgs()