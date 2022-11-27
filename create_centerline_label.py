import sys
sys.path.append('..')
import os
import cv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from skimage.morphology import skeletonize, binary_erosion,binary_dilation,rectangle,binary_opening,remove_small_objects

from configs import get_config_no_args



def data_preprocess(labels_path, centerline_label_path, is_overwrite=True) :
       
    if is_overwrite and isdir(centerline_label_path):
        shutil.rmtree(centerline_label_path)
    os.makedirs(centerline_label_path, exist_ok=True)
    series_ids = subfiles(labels_path, join=False, suffix='png')
    
    for i in series_ids:
        
        gt = cv2.imread(os.path.join(labels_path, i), 0)
        scrawl_gt = np.ones(gt.shape)*255
        vessel = skeletonize(np.array(gt)/255).astype(np.uint8)
        # backgroud = binary_erosion(np.array(255-gt)/255,rectangle(100,100)).astype(np.uint8)
        # vessel = binary_opening(np.array(gt)/255,rectangle(3,3)).astype(np.bool8)
        # vessel = binary_erosion(np.array(gt)/255,rectangle(3,3)).astype(np.bool8)
        # vessel = remove_small_objects(vessel,min_size=300,connectivity=1).astype(np.uint8)
        backgroud = skeletonize(np.array(255-gt)/255).astype(np.uint8)
        scrawl_gt[vessel==1] = 1
        scrawl_gt[backgroud==1] = 0

        # np.save(scrawl_gt)
        
        

        
        # cv2.imwrite(os.path.join(centerline_label_path, f"vessel_{i}"), vessel)
        cv2.imwrite(os.path.join(centerline_label_path, i), scrawl_gt)
        print(f"{i} DONE")
        


    
if __name__ == '__main__':
    labels_path="/home/lwt/data/CVSS/training/labels"
    scribbles_label_path = "/home/lwt/data/CVSS/training/centerline_vessels"
    data_preprocess(labels_path, scribbles_label_path, 
                                 is_overwrite=True
                               )
    

 
