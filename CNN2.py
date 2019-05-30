from __future__ import print_function
import keras
from keras import regularizers, optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import cv2

batch_size = 32 # Jumlah data per iterasi
num_classes = 6 # Jumlah kelas
epochs = 25
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'
lr_drop = 20
lr = 0.1
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

model = Sequential()

# Convolutional layer
# Menggunakan arsitektur VGG16

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(32,32,3), padding='same',
    kernel_regularizer=regularizers.l2(0.0005))) # L2 regularizer untuk mengurangi weight yang terlalu besar
# https://greydanus.github.io/2016/09/05/regularization/
model.add(BatchNormalization()) # normalisasi output dari layer sebelumnya, mengurangi overfitting, memungkinkan learning rate lebih tinggi
model.add(Dropout(0.25)) # Dropout mencegah overfit
# https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2))) # Layer pooling, untuk downsampling : menurunkan dimensi dan supress noise

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # Menghubungkan layer conv dengan layer dense, konversi matriks ke vektor kolom

# Classifier layer

model.add(Dense(4096,kernel_regularizer=regularizers.l2(0.0005))) # Hidden layer
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes)) # Output layer
model.add(Activation('softmax'))

# initiate SGD optimizer
sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
# opt = 'adam'

# Train model
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Augmentasi data untuk memperbanyak sample dengan melakukan modifikasi kepada gambar yang sudah ada
datagen = ImageDataGenerator(
        # Secara acak menggeser gambar secara horizontal (fraction of total width)
        width_shift_range=0.1,
        # Secara acak menggeser gambar secara vertikal (fraction of total height)
        height_shift_range=0.1,
        horizontal_flip=True,  # Secara acak membalik gambar
        rotation_range=15 # Secara acak memutar gambar
        )

# Pengurangan learning rate setiap epoch
def lr_scheduler(epoch):
            return lr * (0.5 ** (epoch // lr_drop))
reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

# Aplikasikan augmentasi ke training data
datagen.fit(x_train)
# Mulai pembelajaran
model.fit_generator(datagen.flow(x_train, y_train,
                    batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    callbacks=[reduce_lr],
                    workers=4)

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
