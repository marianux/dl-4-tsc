#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:35:31 2018

@author: mariano
"""
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

from tensorflow.python.client import device_lib

import os
#os.environ["CUDA_DEVICE_ORDER"]="01:00.0"   # see issue #152
os.environ["CUDA_DEVICE_ORDER"]="04:00.0"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

batch_size = 128
num_classes = 10
epochs = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

with tf.device('/device:XLA_GPU:0'):
#with tf.device('GPU'):
#with tf.device('CPU'):
#with tf.device('/device:CPU:0'):

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
print('Test loss:', score[0])
print('Test accuracy:', score[1])

