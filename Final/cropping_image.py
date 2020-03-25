from PIL import Image
import os

IMG_DIR = 'C:/Users/user/Desktop/training/large/'

for img in os.listdir(IMG_DIR):
    image = Image.open(os.path.join(IMG_DIR,img))
    # cropping 할 이미지 사이즈 
    area=(20,40,150,240) #(가로 시작점, 세로 시작점, 가로 범위, 세로 범위)
    cropped_img=image.crop(area)
    # 저장할 파일 Type : JPEG, PNG 등 
    # 저장할 때 Quality 수준 : 보통 95 사용 
    cropped_img.save(os.path.join(IMG_DIR,img), "JPEG", quality=95 )

