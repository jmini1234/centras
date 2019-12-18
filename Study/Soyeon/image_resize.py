from PIL import Image
import os

IMG_DIR = 'C:/Users/soyeon/Desktop/fishdata/'

for img in os.listdir(IMG_DIR):
    image = Image.open(os.path.join(IMG_DIR,img))
    # resize 할 이미지 사이즈 
    resize_image = image.resize((240, 240))
    # 저장할 파일 Type : JPEG, PNG 등 
    # 저장할 때 Quality 수준 : 보통 95 사용 
    resize_image.save(os.path.join(IMG_DIR,img), "JPEG", quality=95 )
