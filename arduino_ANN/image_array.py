import csv
import pandas as pd
import numpy as np
import os
import cv2
np.set_printoptions(threshold=np.inf)

IMG_DIR = 'C:/Users/user/Desktop/training_final/mid/'

for img in os.listdir(IMG_DIR):

        img_array = cv2.imread(os.path.join(IMG_DIR,img), 0)

        value = np.asarray(img_array, dtype=np.int)
        value = value.flatten()
        value[value<150]=0
        value[value>=150]=1
        print("{", end='')
        for cell in value:
            print(cell, end=',')
        print("1},")
        