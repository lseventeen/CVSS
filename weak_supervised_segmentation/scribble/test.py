import cv2
import os
import numpy as np




def get_color(img, gt_bak, gt_scr):
    H, W,_ = img.shape
  
    for i in range(H):
        for j in range(W):
            # if gt_bak[i, j] == 255:
            #     img[i,j,:]=[0,0,128]+img[i,j,:]*0.5
            if gt_scr[i, j] == 255:
                img[i,j,:]=[128,0,0]+img[i,j,:]*0.5
           
    return img

img = cv2.imread("scribble/img.png")
gt_bak = cv2.imread("scribble/bak.PNG",0)
gt_bak = np.where(gt_bak > 0, 255, 0).astype(np.uint8)
gt_scr = cv2.imread("scribble/scr.PNG",0)
gt_scr = np.where(gt_scr > 0, 255, 0).astype(np.uint8)
cv2.imwrite("scribble/color.png", get_color(img, gt_bak, gt_scr))
print(gt_bak.shape)




                