import glob
import os
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.segmentation import random_walker
import shutil
from batchgenerators.utilities.file_and_folder_operations import *




def pseudo_label_generator(data, seed):
    # in the seed array: 0 means background, 1 to 3 mean class 1 to 3, 4 means: unknown region
    markers = np.ones_like(seed)
    markers[seed == 4] = 0
    markers[seed == 0] = 1
    markers[seed == 1] = 2
    markers[seed == 2] = 3
    markers[seed == 3] = 4
    sigma = 0.35
    data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                             out_range=(-1, 1))
    pseudo_label = random_walker(data, markers, beta=100, mode='bf')
    return pseudo_label-1
 

for i in sorted(glob.glob("../data/ACDC_training/*_scribble.nii.gz"))[2:]:
    print(i.replace("_scribble.nii.gz", ".nii.gz"))
    img_itk = sitk.ReadImage(i.replace("_scribble.nii.gz", ".nii.gz"))
    image = sitk.GetArrayFromImage(img_itk)
    scribble = sitk.GetArrayFromImage(sitk.ReadImage(i))
    pseudo_volumes = np.zeros_like(image)
    for ind, slice_ind in enumerate(range(image.shape[0])):
        if 1 not in np.unique(scribble[ind, ...]) or 2 not in np.unique(scribble[ind, ...]) or 3 not in np.unique(scribble[ind, ...]):
            pass
        else:
            pseudo_volumes[ind, ...] = pseudo_label_generator(
                image[ind, ...], scribble[ind, ...])
    pseudo_volumes_itk = sitk.GetImageFromArray(pseudo_volumes)
    pseudo_volumes_itk.CopyInformation(img_itk)
    sitk.WriteImage(pseudo_volumes_itk, i.replace(
        "_scribble.nii.gz", "_random_walker.nii.gz"))



def data_preprocess(labels_path, centerline_label_path, is_overwrite=True) :
       
    if is_overwrite and isdir(centerline_label_path):
        shutil.rmtree(centerline_label_path)
    os.makedirs(centerline_label_path, exist_ok=True)
    series_ids = subfiles(labels_path, join=False, suffix='png')
    
    for i in series_ids:
        
        gt = cv2.imread(os.path.join(labels_path, i), 0)
        new_gt = morphology.skeletonize(np.array(gt)/255).astype(np.uint8)*255
        cv2.imwrite(os.path.join(centerline_label_path, i), new_gt)
        print(f"{i} DONE")
        


    
if __name__ == '__main__':
    config = get_config_no_args()
    data_preprocess(labels_path=config.DATASET.TRAIN_LABEL_PATH,
                                 centerline_label_path=config.DATASET.CENTERLINE_LABEL_PATH,
                                 is_overwrite=True
                               )
    