from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import cv2

batch_size = 32 # Jumlah data per iterasi
num_classes = 6 # Jumlah kelas
epochs = 25
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# Membagi data ke training dan testing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Mengambil data yang hewan saja
train_idx = np.where(np.logical_and(y_train >=2, y_train <= 7))
test_idx = np.where(np.logical_and(y_test >=2, y_test <= 7))
x_train = x_train[train_idx[0]]
y_train = y_train[train_idx[0]]
x_test = x_test[test_idx[0]]
y_test = y_test[test_idx[0]]

labels = [  'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck']

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# normalisasi
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train -= 2
y_test -= 2

# One-hot encoding data kategorik
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Membangun model
# referensi : https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
# https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
# VGG net

model = Sequential()

# Convolutional layer

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(32,32,3), padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu')) # Layer convolution
model.add(MaxPooling2D(pool_size=(2,2))) # Layer pooling, untuk downsampling : menurunkan dimensi dan supress noise
model.add(Dropout(0.25)) # Dropout mencegah overfit

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu')) # Layer convolution
model.add(MaxPooling2D(pool_size=(2,2))) # Layer pooling, untuk downsampling : menurunkan dimensi dan supress noise
model.add(Dropout(0.25)) 

model.add(Flatten()) # Menghubungkan layer conv dengan layer dense, konversi matriks ke vektor kolom



img_data = x_train[0]
cv2.imshow('Image',img_data)
cv2.waitKey(0)
img_data = np.expand_dims(img_data, axis=0)
vgg_feature = model.predict(img_data)

# dimensi feature
print(vgg_feature.shape)

print('Feature vector : ')
print(vgg_feature)
