import argparse
import os
import pickle
from random import shuffle
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import safe_load
from scipy import ndimage, stats
from skimage import color, measure
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import removeConnectedComponents
from utils.helpers import dir_exists,remove_files
from skimage.transform import resize
from sklearn.model_selection import train_test_split
def data_process(data_path, save_path, num_sequence, patch_size, stride):
    save_training_images_path = os.path.join(save_path, "training","images")
    save_training_images_patch_path = os.path.join(save_path, "training","image_patch")
    save_training_labels_path = os.path.join(save_path, "training","labels")
    save_training_labels_patch_path = os.path.join(save_path, "training","label_patch")
    save_test_images_path = os.path.join(save_path, "test","images")
    save_test_images_patch_path = os.path.join(save_path, "test","image_patch")
    save_test_labels_path = os.path.join(save_path, "test","labels")
    save_test_labels_patch_path = os.path.join(save_path, "test","label_patch")
  
    dir_exists(save_training_images_path)
    dir_exists(save_training_images_patch_path)
    dir_exists(save_training_labels_path)
    dir_exists(save_training_labels_patch_path)
    dir_exists(save_test_images_path)
    dir_exists(save_test_images_patch_path)
    dir_exists(save_test_labels_path)
    dir_exists(save_test_labels_patch_path)


    img_list,gt_list = DSA_preprocess(data_path, num_sequence)
    train_seq, test_seq, train_lab, test_lab = train_test_split(img_list, gt_list, test_size = 1/3, random_state = 0)

    train_seq_patch = get_patch(train_seq,patch_size, stride)
    train_lab_patch = get_patch(train_lab,patch_size, stride)
    
    # train_seq_patch,train_lab_patch = select_patch_with_lab(train_seq_patch,train_lab_patch)
    test_seq_patch = get_patch(test_seq,patch_size, stride)
    test_lab_patch = get_patch(test_lab,patch_size, stride)
    # test_lab_patch = get_patch_2D(test_lab,patch_size, stride)
    save(train_seq,train_seq_patch, save_training_images_path,save_training_images_patch_path, "train_img")
    save(train_lab,train_lab_patch, save_training_labels_path,save_training_labels_patch_path, "train_lab",False)
    save(test_seq,test_seq_patch, save_test_images_path, save_test_images_patch_path,"test_img")
    save(test_lab,test_lab_patch, save_test_labels_path, save_test_labels_patch_path,"test_lab",False)
    save_full_lab(test_lab, save_test_labels_path)
    print(f"seq_patch size:{len(train_seq_patch)}\nnew_seq_patch size:{len(train_seq_patch)}")
def select_patch_with_lab(seq_patch, lab_patch):
    new_seq_patch = []
    new_lab_patch = []
    for i in range(len(seq_patch)):
        if lab_patch[i].max() > 0:
            new_seq_patch.append(seq_patch[i])
            new_lab_patch.append(lab_patch[i])
    
    return new_seq_patch,new_lab_patch

def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape

    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(torch.tensor(sub1), (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list


def save(imgs_list,imgs_patch_list, path,patch_path, type, is_seq =True):
    save_pkl(imgs_patch_list, patch_path, type)
    if is_seq:
        save_seq_png(imgs_list, path, type)
    else:
        save_lab_png(imgs_list, path, type)

def save_pkl(imgs_list, path, type):
    for i, sub in enumerate(imgs_list):
        file = f'{i}.pkl'
        with open(file=os.path.join(path, file), mode='wb') as f:
            pickle.dump(np.array(sub), f)
            print(f'save_{type} : {file}')
    

def save_seq_png(seqs_list, path, type):
    for id_s, seq in enumerate(seqs_list):
        for id_i,img in enumerate(seq):
            file = f"{id_s}_{id_i}.png"
            cv2.imwrite(f"{path}/{file}", ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8))
            print(f'save_seqs_{type} : {file}')

def save_lab_png(labs_list, path, type):
    for id_s, lab in enumerate(labs_list):
        file = f"{id_s}.png"
        cv2.imwrite(f"{path}/{file}", lab[0])
        print(f'save_labs_{type} : {file}')

def save_full_lab(labs_list, path):
    with open(file=os.path.join(path, "label.pkl"), mode='wb') as f:
        pickle.dump(np.array(labs_list), f)

def DSA_preprocess(path,num_sequence):
    label_path = os.path.join(path, "labels")
    image_path = os.path.join(path, "images")
    
    image_files = list(sorted(os.listdir(image_path)))
    label_files = list(sorted(os.listdir(label_path)))
    slice_count = []
    sequences_list = []
    for i in range(1,num_sequence+1):
        slice_count_each_sequence = 0
        image_each_slice = []
        for j in image_files:
            if int(j[:2]) == i:
                slice_count_each_sequence += 1
                img = cv2.imread(os.path.join(image_path, j), 0)
                image_each_slice.append(img)
        slice_count.append(slice_count_each_sequence)
        sequences_list.append(np.array(image_each_slice))
    # slice_num = np.median(np.array(slice_count))
    s,h,w = sequences_list[0].shape
    new_shape = [8,h,w]
    image_full = []
    for s in sequences_list:
        sequence = resize(s,new_shape,mode = "edge",anti_aliasing=False)
        mn = sequence.mean()
        std = sequence.std()
        print(sequence.shape, sequence.dtype, mn, std)
        sequence = (sequence - mn) / (std + 1e-8)
        # image_full.append(ToTensor()(sequence))
        image_full.append(sequence)
        
    label_full = []
    for i in range(1,num_sequence+1):
        label_list = []
        for j in label_files:
            if int(j[:2]) == i:
                label = cv2.imread(os.path.join(label_path, j), 0)
                label_list.append(np.where(label >= 100, 1., 0.).astype(np.float32))
                print(j)
        label = np.array(label_list).max(axis=0)
        label_full.append(label.reshape(1,label.shape[0],label.shape[1]))
    return image_full,label_full

                
    

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()


    with open('config.yaml', encoding='utf-8') as file:
        config = safe_load(file)  # 为列表类型

    data_path = "/home/lwt/data/DSA"
    save_path="/home/lwt/data_pro/DSA"
    data_process(config["data_path"], config["save_path"], 60, **config["data_process"])