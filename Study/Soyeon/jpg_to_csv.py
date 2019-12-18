import numpy as np
import cv2 #pip install opencv-python
import os

img_array = cv2.imread('C:/Users/CSE_125-2/Desktop/images.jpg', cv2.IMREAD_GRAYSCALE)
print(img_array)
img_array = (img_array.flatten())
img_array  = img_array.reshape(-1, 1).T
print(img_array)
with open('output.csv', 'ab') as f:
    np.savetxt(f, img_array, delimiter=",")


