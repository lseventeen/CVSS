import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import torch
from torch.utils.data import Dataset
from data.data_augmentation import Compose, Standardize, ToTensor, CropToFixed, HorizontalFlip, BlobsToMask, VerticalFlip, RandomRotate90, GaussianBlur3D, RandomContrast, AdditiveGaussianNoise,ElasticDeformation
from PIL import Image
import cv2
import torch.nn.functional as F
from skimage.transform import resize  

class CVSS_train_dataset(Dataset):
    def __init__(self, config, series_ids):
        self.images_path = config.DATASET.TRAIN_IMAGE_PATH
        
        
        # self.pretrain_dataset_path = config.DATASET.PRETRAIN_DATASET_PATH
        # self.pretrain_dataset_name = config.DATASET.PRETRAIN_DATASET_NAME
        self.train_mode = config.TRAIN.MODE
        print(self.train_mode)
        assert self.train_mode in ["normal", "scrawl","largeVessel", "centerline"]
        if self.train_mode == "scrawl":
            self.labels_path = config.DATASET.SCRAWL_LABEL_PATH
        elif self.train_mode == "largeVessel":
            
            self.labels_path = config.DATASET.LARGE_VESSEL_LABEL_PATH
        elif self.train_mode == "centerline":
            self.labels_path = config.DATASET.CENTERLINE_LABEL_PATH
        else:
            self.labels_path = config.DATASET.TRAIN_LABEL_PATH
        self.series_ids = series_ids
        self.size = config.DATASET.PATCH_SIZE
        self.num_each_epoch = config.DATASET.NUM_EACH_EPOCH
        

        

        self.images, self.gts = self.CVSS_process(self.labels_path)
        # elif self.train_mode == "scrawl":
        #     self.images, self.gts = self.CVSS_process(self.scrawl_labels_path)
        # elif self.train_mode == "pretrain":
        #     self.load_pretrain_data(self.pretrain_dataset_path,self.pretrain_dataset_name)
       
        
        seed = np.random.randint(123)
       
        if self.train_mode == "scrawl" or self.train_mode =="largeVessel" or self.train_mode =="centerline":
            
            self.seq_DA = Compose([
                CropToFixed(np.random.RandomState(seed),size=self.size),
                HorizontalFlip(np.random.RandomState(seed)),
                VerticalFlip(np.random.RandomState(seed)),
                RandomRotate90(np.random.RandomState(seed)),
                # RandomContrast(np.random.RandomState(seed),execution_probability=0.5),
                # GaussianBlur3D(execution_probability=0.5),
                # AdditiveGaussianNoise(np.random.RandomState(seed), scale=(0., 0.1), execution_probability=0.1),
                ToTensor(False)
            ])
            self.gt_DA = Compose([
                CropToFixed(np.random.RandomState(seed),size=self.size),
                HorizontalFlip(np.random.RandomState(seed)),
                VerticalFlip(np.random.RandomState(seed)),
                RandomRotate90(np.random.RandomState(seed)),
                ToTensor(False)
            
            ])
           
        else: 
            self.seq_DA = Compose([
            CropToFixed(np.random.RandomState(seed),size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            # RandomContrast(np.random.RandomState(seed),execution_probability=0.5),
            # ElasticDeformation(np.random.RandomState(seed),spline_order=3),
            # GaussianBlur3D(execution_probability=0.5),
            # AdditiveGaussianNoise(np.random.RandomState(seed), scale=(0., 0.1), execution_probability=0.1),
            ToTensor(False)
        
        ])
            self.gt_DA = Compose([
            CropToFixed(np.random.RandomState(seed),size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            # ElasticDeformation(np.random.RandomState(seed),spline_order=1),
            BlobsToMask(),
            ToTensor(False)
        ])
    def load_pretrain_data(self, data_path, data_name):
        full_imgs, full_gts = [],[]
        
        for i in data_name:
            imgs,gts = self.pretrain_data_process(data_path, i)
            full_imgs.extend(imgs)
            full_gts.extend(gts)
        return full_imgs,full_gts
            


    # def pretrain_data_process(self,data_path, data_name):

    #     img_path = os.path.join(data_path,data_name, "images")
    #     gt_path = os.path.join(data_path,data_name, "labels")
    #     file_list = list(sorted(os.listdir(img_path)))
        
    #     img_list = []
    #     gt_list = []

    #     for i, file in enumerate(file_list):
    #         if data_name == "DRIVE":
    #             img = Image.open(os.path.join(img_path, file))
    #             gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
    #             img_list.append(np.array(img).transpose(2,0,1))
    #             gt_list.append(np.array(gt)[np.newaxis])

    #         elif data_name == "CHASEDB1":
                    
    #             img = Image.open(os.path.join(img_path, file))
    #             gt = Image.open(os.path.join(gt_path, file[0:9] + '_1stHO.png'))
                    
    #             img_list.append(np.array(img).transpose(2,0,1))
    #             gt_list.append(np.array(gt)[np.newaxis])
    #         elif data_name == "STARE":
                
    #             img = Image.open(os.path.join(img_path, file))
    #             gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
            
    #             img_list.append(np.array(img).transpose(2,0,1))
    #             gt_list.append(np.array(gt)[np.newaxis])

    #         elif data_name == "DCA1":
                
    #             img = cv2.imread(os.path.join(data_path, file), 0)
    #             gt = cv2.imread(os.path.join(data_path, file[:-4] + '_gt.pgm'), 0)
    #             gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
    #             img_list.append(np.array(img)[np.newaxis])
        
    #             gt_list.append(np.array(gt)[np.newaxis])
                
    #         elif data_name == "CHUAC":
                
    #             img = cv2.imread(os.path.join(img_path, file), 0)
    #             if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
    #                 tail = "PNG"
    #             else:
    #                 tail = "png"
    #             gt = cv2.imread(os.path.join(
    #                 gt_path, "angio"+file[:-4] + "ok."+tail), 0)
    #             gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
    #             img = cv2.resize(
    #                 img, (512, 512), interpolation=cv2.INTER_LINEAR)
    #             img_list.append(np.array(img)[np.newaxis])
    #             gt_list.append(np.array(gt)[np.newaxis])
    #     img_list = self.extend_slice(img_list)
    #     return img_list,gt_list
        
 
    def extend_slice(self,images,new_slice=8):
        h,w = images[0].shape[1:]
        new_shape = [new_slice,h,w]
        new_images = []
        for s in images:
            sequence = resize(s,new_shape,order=3,mode = "edge",anti_aliasing=False)
            new_images.append(sequence)
        return new_images
       
    def CVSS_process(self,label_path):
        images = []
        gts = []
        for i in self.series_ids:
            image_each_slice = []
            for j in range(8):
                img = cv2.imread(os.path.join(
                    self.images_path, f"image_s{i}_i{j}.png"), 0)
                image_each_slice.append(img)
            seq = np.array(image_each_slice)
            mn = seq.mean()
            std = seq.std()
            seq = (seq - mn) / (std + 1e-8)
            seq = torch.from_numpy(seq).float()
            gt = cv2.imread(os.path.join( 
                label_path, f"label_s{i}.png"), 0)
            gt = np.array(gt)[np.newaxis]
    
            images.append(seq)
            gts.append(gt)

        return images, gts
 
    def __getitem__(self, idx):
        # data_id = self.series_ids[idx]
        torch.manual_seed(idx)
        id =  np.random.randint(len(self.images))
        img = self.images[id]
        gt = self.gts[id]
        
        
        img = self.seq_DA(img)
        gt = self.gt_DA(gt)
        return img, gt[0].long()

    def __len__(self):
        return self.num_each_epoch


class CVSS_test_dataset(CVSS_train_dataset):
    def __init__(self, config, series_ids, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        self.series_ids = series_ids
        self.img_list, self.gt_list = self.preprocess()
        self.img_patch = self.get_patch(self.img_list,self.patch_size,self.stride)
        self.gt_patch = self.get_patch(self.gt_list,self.patch_size,self.stride)
        
        
   
    def get_patch(self,image_list, patch_size, stride):
        patch_list = []
        _, h, w = image_list[0].shape

        pad_h = stride - (h - patch_size[0]) % stride
        pad_w = stride - (w - patch_size[1]) % stride
        for image in image_list:
            image = F.pad(image,(0, pad_w, 0, pad_h), "constant", 0)
            image = image.unfold(1, patch_size[0], stride).unfold(2, patch_size[1], stride).permute(1, 2, 0, 3, 4)
            image = image.contiguous().view(image.shape[0] * image.shape[1], image.shape[2], patch_size[0], patch_size[1])
            for sub in image:
                patch_list.append(sub)
        return patch_list
    def preprocess(self,):
        images = []
        gts = []
        for i in self.series_ids:
            image_each_slice = []
            for j in range(8):
                img = cv2.imread(os.path.join(
                    self.images_path, f"image_s{i}_i{j}.png"), 0)
                image_each_slice.append(img)
            seq = np.array(image_each_slice)
            mn = seq.mean()
            std = seq.std()
            seq = (seq - mn) / (std + 1e-8)
            seq = torch.from_numpy(seq).float()

            gt = cv2.imread(os.path.join(
                self.labels_path, f"label_s{i}.png"), 0)
            gt = np.array(gt/255)[np.newaxis]
            gt = torch.from_numpy(gt).float()

            images.append(seq)
            gts.append(gt)

        return images, gts

    def __getitem__(self, idx):
        
        
        img = self.img_patch[idx]
        gt = self.gt_patch[idx]
        return img,gt[0].long()

    def __len__(self):
        return len(self.img_patch)






    

        
                
       
        
           
        



