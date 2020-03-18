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

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

with open("./pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]

batch_size = 128
nb_classes = 10
nb_epoch = 1

train_images=train_images.reshape(1619, 240, 240, 3)
test_images=test_images.reshape(460, 240, 240, 3)

#trainSet과 validationSet 나누기
x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2)

#모델 구성
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(240, 240, 3)),
    keras.layers.Dense(1619, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
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
    epochs=5, 
    callbacks=[ModelCheckpoint('my_model_weights.h5', monitor='val_loss', save_best_only=True)],
    validation_data=(x_test, y_test)
)


# store model
with open('./my_nn_arch.json', 'w') as fout:
    fout.write(model.to_json())
model.save_weights('./my_nn_weights.h5', overwrite=True)

# store one sample in text file
with open("./fish.dat", "w") as fin:
    fin.write("1 240 240\n")
    a = x_train[0,0]
    for b in a:
        fin.write(str(b)+'\n')

# get prediction on saved sample
# c++ output should be the same ;)
print('Prediction on saved sample:')
print(str(model.predict(x_train[:1])))

print("=======================================================================================")
print("model weights:")
print(model.layers[2].get_weights())


# on my pc I got:
#[[ 0.03729606  0.00783805  0.06588034  0.21728528  0.01093729  0.34730983
#   0.01350389  0.02174525  0.26624694  0.01195715]]