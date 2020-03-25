om PIL import Image
import os

IMG_DIR = 'C:/Users/user/Desktop/test/large_test/'

for img in os.listdir(IMG_DIR):
    
    image = Image.open(os.path.join(IMG_DIR,img))
    # cropping 할 이미지 사이즈 
    area=(60,20,180,220) #(가로 시작점, 세로 시작점, 가로 범위, 세로 범위)
    cropped_img=image.crop(area)
    #cropped_img.show()
    # 저장할 파일 Type : JPEG, PNG 등 
    # 저장할 때 Quality 수준 : 보통 95 사용 
    cropped_img.save(os.path.join(IMG_DIR,img), "