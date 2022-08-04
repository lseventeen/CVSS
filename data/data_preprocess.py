import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from configs.config import get_config_no_args

class data_preprocess(object):
    def __init__(self, config, is_overwrite=True) -> None:
        self.config = config 
        self.train_images_path = self.config.DATASET.TRAIN_IMAGE_PATH
        self.train_labels_path = self.config.DATASET.TRAIN_LABEL_PATH
        self.preprocess_path = self.config.DATASET.PROPRECESS_PATH
        self.patch_size = self.config.DATASET.PATCH_SIZE 
        self.stride = self.config.DATASET.STRIDE
        if is_overwrite and isdir(self.preprocess_path):
            shutil.rmtree(self.preprocess_path)
        os.makedirs(self.preprocess_path, exist_ok=True)

    def get_patch(self, image):
        image_list = []
        _, h, w = image.shape

        pad_h = self.stride - (h - self.patch_size) % self.stride
        pad_w = self.stride - (w - self.patch_size) % self.stride
       
        image = F.pad(torch.tensor(image),
                          (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, self.patch_size, self.stride).unfold(
                2, self.patch_size, self.stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
                image.shape[0] * image.shape[1], image.shape[2], self.patch_size, self.patch_size)
        for sub in image:
            image_list.append(np.array(sub))
        return image_list

    def preprocess(self,):
        num = 0
        for i in range(40):
            image_each_slice = []
            for j in range(8):
                img = cv2.imread(os.path.join(
                    self.train_images_path, f"image_s{i}_i{j}.png"), 0)
                image_each_slice.append(img)
            seq = np.array(image_each_slice)
            mn = seq.mean()
            std = seq.std()
            seq = (seq - mn) / (std + 1e-8)

            gt = cv2.imread(os.path.join(
                self.train_labels_path, f"label_s{i}.png"), 0)
            gt = np.array(gt/255)[np.newaxis]

            seq = self.get_patch(seq)
            gt = self.get_patch(gt)

            for i in range(len(seq)):
                if gt[i].max() > 0:
                    np.savez_compressed(os.path.join(self.preprocess_path, "%s.npz" % num), img=seq[i],lab=gt[i])
                    print(f"{num} done" )
                    num += 1

if __name__ == '__main__':
    config = get_config_no_args()
    dp = data_preprocess(config)
    dp.preprocess()
