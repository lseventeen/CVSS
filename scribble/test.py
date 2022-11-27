import cv2
import os
import numpy as np
gt = cv2.imread("scribble/b16.png")
gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
cv2.imwrite("b162.png", gt)
print(gt.shape)