import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation
import cv2

class CVSS_train_dataset(Dataset):
    def __init__(self, config, series_ids, is_val=False):
        self.data_path = config.DATASET.PROPRECESS_PATH
        self.series_ids = series_ids
        self.is_val = is_val

        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])

    def __getitem__(self, idx):
        data_id = self.series_ids[idx]
        data_load = np.load(join(self.data_path, data_id))
        img = torch.from_numpy(data_load["img"]).float()
        gt = torch.from_numpy(data_load["lab"]).float()

        if not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        return img, gt

    def __len__(self):
        return len(self.series_ids)


class CVSS_test_dataset(Dataset):
    def __init__(self, config):
        self.image_path = config.DATASET.TEST_IMAGE_PATH
        self.label_path = config.DATASET.TEST_LABEL_PATH
        self.series_ids = subfiles(config.DATASET.TEST_IMAGE_PATH, join=False, suffix='png')
    def __getitem__(self, idx):
        img_id = self.series_ids[idx]
        
        img = cv2.imread(os.path.join(self.image_path, img_id), 0)
        seq = np.array(image_each_slice)
        mn = seq.mean()
        std = seq.std()
        seq = (seq - mn) / (std + 1e-8)
        gt = cv2.imread(os.path.join(self.image_path, "label_" + img_id.split("_")[1]) + ".png", 0)

        img = torch.from_numpy(data_load["img"]).float()
        gt = torch.from_numpy(data_load["lab"]).float()

       

        return img, gt

    def __len__(self):
        return len(self.series_ids)
