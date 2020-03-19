#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:19:38 2020

@author: menglu
"""
import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt

from pandas import read_csv
import numpy as np

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import backend as K

road4 = np.load('/Users/menglu/Documents/Github/deep_learning/predictors/road4.npy')
road3 = np.load('/Users/menglu/Documents/Github/deep_learning/predictors/road3.npy')
road2 = np.load('/Users/menglu/Documents/Github/deep_learning/predictors/road2.npy')
road5 = np.load('/Users/menglu/Documents/Github/deep_learning/predictors/road5.npy')
road1 = np.load('/Users/menglu/Documents/Github/deep_learning/predictors/road1.npy')

road234=np.array(( road2,road3, road4, road5))
road234.shape
 
plt.imshow(road2[:,:,1])
plt.show() 


mal = [599,2225,2478,2504] # id of removed rasters
ap = read_csv('/Users/menglu/Documents/Github/deep_learning/airbase_oaq.csv')
ap.shape[0]-2634
ap = ap[:-3042]
ap = ap.drop(mal)
ap.shape
ap.dropna # no na same as ap
Y = ap['value_mean']

data_augmentation = True


Xtrainv =road234[:,:,:,1:2300] # 2300 for training and validation, consisting of trainging and validation
Xtest =road234[:,:,:,2300:]  #330 for testing, not going to touch
Ytrainv = Y[1:2300]
Ytest = Y[2300:]




# define
# kernal initializer is important!
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape, kernel_initializer = 'normal'))
    model.add(layers.BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

     
    
    model.add(Conv2D(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
   
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(1))
 
    model.add(Activation('relu'))
    return model
# I obtained higer accuracy with dropout and bachnorm. The mae is almost almost be around 10. 
# averagepooling is not as steady as maxpooling
# data augumentation assumes the inputs are images, with 1, 3, or 4 channels: grayscale, rgb, 
# the first dimension is always the number of samples. 

k  = 4  # 4fold
num_validation_samples = Xtrainv.shape[3]//k
validation_scores = []
for fold in range (k):
    validation_X= Xtrainv[:,:,:, num_validation_samples*fold : num_validation_samples*(fold+1)]
    training_X = Xtrainv[:,:,:,:num_validation_samples*fold] + Xtrainv[:,:,:,num_validation_samples*(fold+1):]
    validation_Y= Ytrainv[num_validation_samples*fold : num_validation_samples*(fold+1)]
    training_Y = Ytrainv[:num_validation_samples*fold] + Xtrainv[num_validation_samples*(fold+1):] 

    if K.image_data_format() == 'channels_first':
        input_shape = (training_X.shape[0], training_X.shape[1], training_X.shape[2])
        validation_X = validation_X.reshape(validation_X.shape[3], validation_X.shape[0], validation_X.shape[1], validation_X.shape[2])
        training_X =training_X.reshape(training_X.shape[3], training_X.shape[0], training_X.shape[1], training_X.shape[2])
        Xtest = Xtest.reshape(Xtest.shape[3], Xtest.shape[0], Xtest.shape[1], Xtest.shape[2])
    
    
    else:
        input_shape = (training_X.shape[1], training_X.shape[2], training_X.shape[0]) 
        validation_X =validation_X.reshape(validation_X.shape[3], validation_X.shape[1], validation_X.shape[2], validation_X.shape[0])
        training_X =training_X.reshape(training_X.shape[3], training_X.shape[1], training_X.shape[2], training_X.shape[0])
    
        Xtest = Xtest.reshape(Xtest.shape[3], Xtest.shape[1], Xtest.shape[2], Xtest.shape[0])

    
    

 

    m = cnn_model() 
    print(cnn_model().summary())
    m.compile(loss='mae',
              optimizer='adam',
              metrics=['mse', 'mae'])

    
 
    history = m.fit(training_X, training_Y,
          batch_size= 100,
          epochs= 5,
          verbose=1,
          validation_data=(validation_X, validation_Y))
    validation_score =history['mae']
    validation_scores.append(validation_score)
    validation_score =np.average(validation_scores)
    testscore = m.evaluate(Xtest, Ytest, verbose=0)
    print('val-:',validation_score, 'test:', testscore)
  
    print(history.history.keys())
# "Loss"
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
# save before show(), otherwise wont work
#    plt.savefig("/Users/menglu/Documents/deep_learning/apcnn50poc_mae.png")
    plt.show()
 