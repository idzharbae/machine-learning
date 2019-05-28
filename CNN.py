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
epochs = 50
# data_augmentation = True
# num_predictions = 20
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

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu')) # Layer convolution
model.add(MaxPooling2D(pool_size=(2,2))) # Layer pooling, untuk downsampling : menurunkan dimensi dan supress noise
model.add(Dropout(0.25)) 

model.add(Flatten()) # Menghubungkan layer conv dengan layer dense, konversi matriks ke vektor kolom
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# opt = 'adam'

# Train model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])