from tensorflow.compat.v2.keras.models import model_from_json
import os
import re
import cv2
import numpy as np

# model.json 파일 열기
json_file = open("./my_nn_arch.json", "r") 
loaded_model_json = json_file.read() 
json_file.close() 

# json파일로부터 model 로드하기
loaded_model = model_from_json(loaded_model_json)

# 로드한 model에 weight 로드하기
loaded_model.load_weights("./my_nn_weights.h5")
print("Loaded model from disk")


#5~13, 14~20, 20~
#테스트 완료 후 prediction code
IMG_DIR = 'C:/Users/user/Desktop/0215test/0215test'

for img in os.listdir(IMG_DIR):
    #print(img)
    p = re.compile('(\d+(\.\d+)*)')  #img name 중 실수 문자열 검색 
    result = p.findall(img)
    print("답:", result[0][0])
    
    img_array = cv2.imread(os.path.join(IMG_DIR,img))
    value = np.asarray(img_array, dtype=np.int)
    value = value.flatten()
   
    for i in range(len(value)):
        if value[i]<150:
            value[i]=0
        else:
            value[i]=255
    value = value.reshape(1, 240,240,3)
    predictions = loaded_model.predict(value)
    print("prediction 결과", np.argmax(predictions[0]))
    print("****************************************************************")
