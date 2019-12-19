from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os, glob
import numpy as np

data_aug_gen=ImageDataGenerator(
    rescale=1./240
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    cannel_shift_range=0.5,
    brightness_range=[0.2,1.0]
    fill_mode='nearest'
    )
print(data_aug_gen)

for categoty in categories:
    image_dir=base_dir+category
    files=glob.glob(image_dir+"/*.jpg")
    print(categoty, " 파일 길이 : ", len(files))
    
    for image in files:
        img=load_img(image)
        print(img)
        x=img_to_array(img)
        x=x.reshape((1,)+x.shape)
        i=0
        name=imge.split('\\')[1].split('.')[0]
        for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir="C:/Users/CSE_125-2/Desktop/test"):
            i+=1
            if i>3:
                break
        
