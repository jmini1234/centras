import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import os

filename='C:/Users/CSE_125-2/Desktop/output.csv'
savepath='C:/Users/CSE_125-2/Desktop/test/'

with open(filename) as f:
    reader = csv.reader(f, delimiter=';')
    
    isLabel=True
    fish_len=""
    fish_dis=""
    savename=""
    for row in reader:
        if isLabel:
            fish_len, fish_dis = row
            savename=str(str(fish_len)+"_"+str(fish_dis)+".jpg")
            isLabel=False
        else:
            value = np.asarray(row, dtype=np.int)
            value = value.flatten()
            pixels = np.array(value).reshape(240,240,3)
            cv2.imwrite(os.path.join(savepath,savename), pixels)
            isLabel=True
        
# pixels = np.array(df).reshape(240,240,3)
# plt.imshow(pixels)
# plt.show()
