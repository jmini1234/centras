from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        rotation_range=5,    #회전 5도 
        width_shift_range=0.01,  #수평으로 2.4 픽셀정도로만 움직임 
        height_shift_range=0.01, #수직으로 2.4 픽셀정도로만 움직임 
        shear_range=0, #밀림 강도 0 -> 회전 
        zoom_range=0,  #확대/축소 하지 않음 
        horizontal_flip = True,  #한 방향으로 들어오니까 수평 회전은 o ?  
        vertical_flip=False, #거꾸로 들어 올 수 있으니까 수직 회전 x ?
        fill_mode='nearest')

for img in os.listdir('./test/') :
    prefix = os.path.splitext(img)[0]    
    img = load_img('./test/'+img)  # PIL 이미지
    x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
    x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열
    
    i=1
    # 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
    # 지정된 `preview/` 폴더에 저장합니다i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='preview', save_prefix=prefix, save_format='jpg'):
        i += 1
        if i > 20:
            break  # 이미지 20장을 생성하고 마칩니다

