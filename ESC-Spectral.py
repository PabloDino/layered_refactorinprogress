from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Conv1D,Conv2D, GlobalAveragePooling2D, InputLayer, \
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape,Conv2DTranspose, Concatenate,concatenate, Input, AveragePooling2D
from keras.layers.normalization import BatchNormalization   

from keras.optimizers import Adam              
import tensorflow as tf
import sys, os
from time import time
import numpy as np
import os
from config import *
import datetime
import random
import keras.optimizers
import librosa
import librosa.display
import pandas as pd
import warnings
from keras import backend as K


import utils

from utils import importData, preprocess, recall_m, precision_m, f1_m 
from timeit import default_timer as timer



totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 100
latent_dim=8
dataSize=128

timesteps = 128 # Length of your sequences
input_dim = 128 

def buildModel():

    model_a = Sequential()

    l_input_shape_a=(128, 128,1,1)
    input_shape_a=(128, 128,1)
    model_a_in = Input(shape=input_shape_a)
    
    re0a = Reshape(target_shape=(128*128,1),input_shape=(128,128,1))(model_a_in)
    ft0a = Lambda(lambda v: tf.to_float(tf.spectral.rfft(v)))(re0a)

	
    conv_1a = Conv1D(24, kernel_size=latent_dim, activation='relu')(ft0a)
    pool_2a = MaxPooling1D(pool_size=latent_dim)(conv_1a)
    act_4a =Activation('relu')(pool_2a)

    ls5a= LSTM(latent_dim*latent_dim,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(act_4a)
    print('ls5 a shape is ', ls5a.shape) 
    
    conv_5a = Conv1D(48, kernel_size=latent_dim,  activation='relu')(ls5a)

    pool_6a=MaxPooling1D(pool_size=latent_dim)(conv_5a)
    act_7a = Activation('relu')(pool_6a)
    ift7a = Lambda(lambda v: tf.to_float(tf.spectral.irfft(tf.cast(v, dtype=tf.complex64))))(act_7a)

    re_7a = Reshape(target_shape=(255,94,1))(ift7a)
    pool_8a = MaxPooling2D((latent_dim//2,2))(re_7a)

    tr8a = Conv2DTranspose(1, kernel_size=(2,latent_dim), activation='relu', padding='valid')(pool_8a) 

    act_9a = Activation('relu')(tr8a)    # 2 x 25 x 48

    tr9a = Conv2DTranspose(1, kernel_size=(1,latent_dim), activation='relu', padding='valid')(act_9a) 

    act_9aa = Activation('relu')(tr9a)    # 2 x 25 x 48

    tr10a = Conv2DTranspose(1, kernel_size=(1,latent_dim//2), activation='relu', padding='valid')(act_9aa)     
    act_10aa = Activation('relu')(tr10a)
   
    re_10aa = Reshape(target_shape=(latent_dim*latent_dim, latent_dim*latent_dim))(act_10aa)

    flat12 = Flatten()(re_10aa)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=model_a_in, outputs=out)
    return model
    
    

model = buildModel()
 


#model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])


(trainx,trainy), (testx, testy) = importData()#.load_data()
    
train_data, train_labels = preprocess(trainx,trainy)
test_data, test_labels = preprocess(testx,testy)

train_labels = np.array(keras.utils.to_categorical(train_labels, totalLabel))
test_labels = np.array(keras.utils.to_categorical(test_labels, totalLabel))

model.summary()
print ('xtrain shape is ',train_data.shape)
print ('ytrain shape is ',train_labels.shape)

print ('xtest shape is ',test_data.shape)
print ('ytest shape is ',test_labels.shape)
model.fit(train_data,
        y=train_labels,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (test_data, test_labels)
        )
    
  
modelName = 'esc-spectral.hdf5'
model.save(modelName)


