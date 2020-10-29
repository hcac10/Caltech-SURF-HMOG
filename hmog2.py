#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_file(filepath):
    print("file loading")
    dataframe = read_csv(filepath, header = None, sep = ',', engine = 'python')
    return dataframe.values
 
def load_dataset_group(group, prefix=''):
    print('dataset group loading')
    filepath1 = prefix + group + '/swipe_positions/'
    filepath2 = prefix + group + '/output_positions/'
    # load all 2 files as a single array
    # load input data
    X = load_file(filepath1 + 'swipes_'+group+'.csv')
    # load class output
    y = load_file(filepath2 + '/sub_'+group+'.csv')

    return X, y
def load_dataset(prefix=''):
    print('dataset loading')
    # load all test
    testX, testy = load_dataset_group('test', prefix + '/Users/hcac/Desktop/5x5Test/')
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + '/Users/hcac/Desktop/5x5Test/')
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

batch_size = 128
num_classes = 6
epochs = 1

# input image dimensions
img_x, img_y = 540, 960

x_train, y_train, x_test, y_test = load_dataset()
print('original y train shape:', y_train.shape)
print('original x_train shape:', x_train.shape)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print('x_train[0]:', x_train[0])

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train.shape after to_categorically', y_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_sc_train = scaler.transform(x_train)
x_sc_test = scaler.transform(x_test)

pca = PCA(n_components=NCOMPONENTS)
x_pca_train = pca.fit_transform(x_sc_train)
x_pca_test = pca.transform(x_sc_test)
pca_std = np.std(x_pca_train)

print("x_sc_train.shape: ", x_sc_train.shape)
print('x_pca_train.shape:', x_pca_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_pca_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x__pca_test, y_test),
          callbacks=[history])
score = model.evaluate(x_pca_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:




