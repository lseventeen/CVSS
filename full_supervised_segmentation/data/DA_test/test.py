from transfors_SE import LabelToAffinities, Compose, BlobsToMask, HorizontalFlip, VerticalFlip, RandomRotate, RandomRotate90, GaussianBlur3D, RandomContrast, Normalize, ElasticDeformation, StandardLabelToBoundary, AdditiveGaussianNoise, AdditivePoissonNoise, LabelToBoundaryAndAffinities,  LabelToBoundaryAndAffinities, Standardize
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os 
import torch
image_path = '/home/lwt/data/CVSS/training/images/'
label_path = '/home/lwt/data/CVSS/training/labels/'
import random
image_each_slice = []
i = 0
for j in range(8):
    img = cv2.imread(os.path.join(image_path, f"image_s{i}_i{j}.png"), 0)
    image_each_slice.append(255-img)
seq = np.array(image_each_slice)/255
# seq = torch.from_numpy(seq).float()
gt = cv2.imread(os.path.join(label_path, f"label_s{i}.png"), 0)
gt = np.array(gt/255)[np.newaxis]
# gt = torch.from_numpy(gt).float()
seq = seq[:,400:464,400:464]
gt = gt[0,400:464,400:464]

# t = ElasticDeformation(np.random.RandomState(),spline_order = 1, execution_probability=1)

# t(gt)

# t = HorizontalFlip(np.random.RandomState())
# GLOBAL_RANDOM_STATE = np.random.RandomState(47)
# seed = GLOBAL_RANDOM_STATE.randint(10000000)
# np.random.seed(1)
# a1 = np.random.RandomState(1)
# a2 = np.random.RandomState(1)
# # random.seed(1)
# t = VerticalFlip(random_state=a1)
# t1 = VerticalFlip(random_state=a2)

seed = 0
# a1 = np.random.RandomState(seed)
# a2 = np.random.RandomState(seed)
# t1 = ElasticDeformation(a1,spline_order = 3, execution_probability=1)
# t2 = ElasticDeformation(a2,spline_order = 1, execution_probability=1)

trans = Compose(
    [ElasticDeformation(np.random.RandomState(seed),spline_order = 1, execution_probability=1),
    BlobsToMask()
    ]
)



o2 = trans(gt)
print(1)
