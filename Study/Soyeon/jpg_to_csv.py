import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import csv

img = cv2.imread('C:/Users/CSE_125-2/Desktop/images.jpg',cv2.COLOR_BGR2GRAY)     
# pixels = np.array(img)
# plt.imshow(pixels)
# plt.show()
value = np.asarray(img, dtype=np.int)
value = value.flatten()
#print(value.size) # 240*240*3
with open("output.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(value)
