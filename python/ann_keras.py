##TODO: 정확도 왜 낮게 나왔는지 분석 - classification 경계값이 
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

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

train_images=train_images.reshape(1619, 240, 240, 3)
test_images=test_images.reshape(460, 240, 240, 3)

#trainSet과 validationSet 나누기
x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2)

#모델 구성
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(240, 240, 3)),
    keras.layers.Dense(1619, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
print(model.summary())

#모델 훈련
model.fit(
    x_train,
    y_train,
    batch_size=100,
    epochs=20, 
    callbacks=[ModelCheckpoint('my_model_weights.h5', monitor='val_loss', save_best_only=True)],
    validation_data=(x_test, y_test)
)

#모델 저장
model.save('my_model_weights.h5')

#정확도 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)

'''
#테스트 완료 후 prediction code
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
'''
