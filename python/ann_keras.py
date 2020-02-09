from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("./pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]

class_names = ['small', 'medium', 'large']

'''
print(train_images.shape) #(1619, 172800)
print(test_images.shape) #(460, 172800)
print(train_labels.shape) #(1619, 3)
print(test_labels.shape) #(460, 3)
'''


#모델 구성
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(240, 240, 1619)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
''' 
#모델 훈련
model.fit(train_images, train_labels, epochs=5)

#정확도 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)
'''


'''
#테스트 완료 후 prediction code
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
'''
