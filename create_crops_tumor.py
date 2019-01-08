from __future__ import division
import xml.etree.ElementTree as ET
import openslide as ops
import math
import cv2
import scipy
import numpy as np
import argparse
import os
import random
from tqdm import tqdm

def get_coordinates(xml, zoom):
    """
    Parsing annotated region cordinates from xml data
    Inputs: xml - data containing annotated benign and Gleason regions
            zoom - level of zoom on the slide
    Outputs: anno_coordinates - array of annotated region cordinates
             anno_names - names for annotated regions
    """
    xml_tree = ET.parse(xml)
    xml_root = xml_tree.getroot()
    anno_coordinates = []
    anno_names = []
    count = 0
    for annotation in xml_root.iter('Annotation'):
        for region in annotation.iter('Region'):
            coordinates = []
            for node in region.getiterator():
                if node.tag == 'Vertex':
                    x, y = float(float(node.get("X")))/float(math.pow(4,zoom)), float(float(node.get("Y")))/float(math.pow(4,zoom)) # normalizing by zoom level
                    coordinates.append([x,y])
            anno_coordinates.append(coordinates)
            name = annotation.get('Name').split(' ')
            if name[0] == '':
                print "here"
                color = annotation.get('LineColor')
                if int(color) == 0:
                    anno_name = 'Gleason_3'
                elif int(color) == 16711935:
                    anno_name = 'Gleason_4'
                elif int(color) == 65535:
                    anno_name = 'Gleason_5'
            else:
                if len(name) > 1:
                    anno_name = name[0] + '_' + name[1]
                else:
                    anno_name = annotation.get('Name')
            anno_names.append(anno_name)
    return anno_coordinates, anno_names

def propagate(x, y, xml, slide, patch_dim, tumor_threshold, whiteness_threshold, roi_corners, zoom, anno_name, data_dir):
    """
    Analyze potential malignant image patch  
    Inputs: x - start x-coordinate
            y - start y-corrdiate 
            xml - xml filename
            patch_dim - dimension of patches
            hit_threshold - percentage of patch that needs to be benign to be given the label of benign
            whiteness_threshold - percentage of whiteness that we allow for a patch to have
            roi_corners - the corners of the annoted region
            zoom - level of zoom on the slide
            anno_name - name of the annotation
            data_dir - directory to save image crops in
    Outptus: None
    """
    start_point = (x, y)
    img = slide.read_region(start_point,zoom,(patch_dim,patch_dim))
    np_img = np.asarray(img)    
    tumor_count = 0 
    whiteness_count = 0 

    bad_crop = False
    
    for row in range(np_img.shape[0]):
        for col in range(np_img.shape[1]):
            if cv2.pointPolygonTest(roi_corners,(int(col+x/math.pow(4,zoom)),int(row+y/math.pow(4,zoom))),False) >= 0: # if pixel contains tumor also normalizing by zoom level
                tumor_count += 1
            if int(np_img[row][col][0]) + int(np_img[row][col][1]) + int(np_img[row][col][2]) >= 600: # if pixel is white
                whiteness_count += 1

            if whiteness_count >= whiteness_threshold:
                bad_crop = True
                break

        if bad_crop:
            break

    if tumor_count >= tumor_threshold and not bad_crop:

        file_name = anno_name + '_' + xml.split('/')[-1].split('.')[0] + '_' + str(zoom) + '_' + str(x) + '_' + str(y) + '.png'
        full_file_path = os.path.join(data_dir, file_name)
        img.save(full_file_path)


def create_crops(svs,xml,anno_coordinates,anno_names,patch_dim,zoom,data_dir):
    """
    Goes through histopathology svs slide and xml annotations creating malignant crops
    Inputs: svs - file containing slide at mutiple zoom levels
            xml - data containing annotated benign and Gleason regions
            anno_coordinates - array of annotated region cordinates
            anno_names - names for annotated regions
            zoom - level of zoom on the slide
            anno_name - name of the annotation
            data_dir - directory to save image crops in
    Outptus: None
    """
    print "Evaluating img: ", svs.split('/')[-1].split('.')[0]
    tumor_threshold = (patch_dim*patch_dim*0.5)
    whiteness_threshold = (patch_dim*patch_dim*0.6)

    slide = ops.OpenSlide(svs)

    for anno in range(len(anno_coordinates)):
        print "evaluating annotation number: ", anno
        roi_corners = np.array(anno_coordinates[anno], dtype=np.int32)
        anno_name = str(anno_names[anno])
        if anno_name == 'Cores' or anno_name == 'Core' or anno_name == 'core' or anno_name == 'cores' or anno_name == '':
            continue
        print anno_name
        max_x = - float("inf")
        max_y = - float("inf")
        min_x = float("inf")
        min_y = float("inf")

        # creating bouding box for annotated region
        for cords in roi_corners:
            if cords[0] > max_x:
                max_x = cords[0]

            if cords[0] < min_x:
                min_x = cords[0]

            if cords[1] > max_y:
                max_y = cords[1]

            if cords[1] < min_y:
                min_y = cords[1]

        start_width = min_x
        start_height = min_y

        total_width = max_x - min_x
        total_height = max_y - min_y

        num_width_iters = int(np.ceil(total_width/patch_dim))
        num_height_iters = int(np.ceil(total_height/patch_dim))

        print "number of iterations across width: ", num_width_iters
        print "number of iterations across height: ", num_height_iters

        for x in tqdm(range(num_width_iters)):
            for y in range(num_height_iters):

                width = x * int(patch_dim)
                height = y * int(patch_dim)

                current_width = start_width+width
                current_height = start_height+height
                
                propagate(current_width,current_height,xml,slide,patch_dim,tumor_threshold,whiteness_threshold,roi_corners,zoom,anno_name,data_dir)
           

    print "patches creation finished!"
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('svs')
    parser.add_argument('xml')
    parser.add_argument('zoom')
    parser.add_argument('data_dir')

    args = parser.parse_args()
    zoom = int(args.zoom)
    anno_coordinates, anno_names = get_coordinates(args.xml,zoom)
    patch_dims = 256
    create_crops(args.svs,args.xml,anno_coordinates,anno_names,patch_dims,zoom,args.data_dir)


