import numpy as np
import cv2 ##!pip install opencv-python
import os
import re # 파일 이름에서 크기와 거리 추출
import matplotlib.pyplot as plt
import csv

IMG_DIR = 'C:/Users/CSE124/Desktop/training/'

for img in os.listdir(IMG_DIR):
        #print(img)
        p = re.compile('(\d+(\.\d+)*)')  #img name 중 실수 문자열 검색 
        result = p.findall(img)
        #print(result)
        supervised=list()
        supervised.append(result[0][0])
        #print(supervised)
        img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.COLOR_BGR2GRAY)
        #plt.imshow(pixels)
        #plt.show()
        value = np.asarray(img_array, dtype=np.int)
        value = value.flatten()
        print(value)
        for i in range(len(value)):
            if value[i]<150:
                value[i]=0
            else:
                value[i]=255
    
        supervised.extend(value)
        print(value)
        with open('C:/Users/CSE124/Desktop/training.csv', 'a', encoding="UTF-8", newline='') as f:
            writer = csv.writer(f, delimiter=",", lineterminator='\n')
            writer.writerow(supervised)
