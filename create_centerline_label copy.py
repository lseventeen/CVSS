import sys
sys.path.append('..')
import os
import cv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from skimage.morphology import skeletonize, binary_erosion,binary_dilation,rectangle

from configs import get_config_no_args



def data_preprocess(labels_path, centerline_label_path, is_overwrite=True) :
       
    if is_overwrite and isdir(centerline_label_path):
        shutil.rmtree(centerline_label_path)
    os.makedirs(centerline_label_path, exist_ok=True)
    series_ids = subfiles(labels_path, join=False, suffix='png')
    
    for i in series_ids:
        
        gt = cv2.imread(os.path.join(labels_path, i), 0)
        scrawl_gt = np.ones(gt.shape)*255
        vessel = skeletonize(np.array(gt)/255).astype(np.uint8)*255
        # backgroud = binary_erosion(1-np.array(gt)/255,rectangle(100,100)).astype(np.uint8)*255
        backgroud = skeletonize(np.array(255-gt)/255).astype(np.uint8)*255
        scrawl_gt[vessel==255] = 1
        scrawl_gt[backgroud==255] = 0

        # np.save(scrawl_gt)
        
        

        
        # cv2.imwrite(os.path.join(centerline_label_path, f"vessel_{i}"), vessel)
        cv2.imwrite(os.path.join(centerline_label_path, i), scrawl_gt)
        print(f"{i} DONE")
        


    
if __name__ == '__main__':
    config = get_config_no_args()
    data_preprocess(labels_path=config.DATASET.TRAIN_LABEL_PATH,
                                 centerline_label_path=config.DATASET.SCRAWL_LABEL_PATH,
                                 is_overwrite=True
                               )
    

 
