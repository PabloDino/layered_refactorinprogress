from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential, Model,load_model
from keras.layers import Activation, Dense, Dropout, Conv1D,Conv2D, GlobalAveragePooling2D, InputLayer, \
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate,concatenate, Input, AveragePooling2D
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
from numpy import dstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import utils

from utils import importData, preprocess, recall_m, precision_m, f1_m 
from timeit import default_timer as timer



# model parameters for training
totalLabel = 50
batchSize = 128
epochs = 100
latent_dim=8
dataSize=128

# This function will import wav files by given data source path.
# And will extract wav file features using librosa.feature.melspectrogram.
# Class label will be extracted from the file name
# File name pattern: {WavFileName}-{ClassLabel}
# e.g. 0001-0 (0001 is the name for the wav and 0 is the class label)
# The program only interested in the class label and doesn't care the wav file name
from itertools import chain, combinations




def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def load_all_models(n_models):
        all_models = list()
        for i in range(n_models):
            #if i not in [1]:
            #if i not in [1,2]:
                # define filename for this ensemble
                filename = './models/Model.' + str(i + 1) + '.hdf5'
                #filename = '../models/SingleModel.' + str(i + 1) + '.h5'
                # load model from file
                print('loading ', filename)
                model = load_model(filename,  custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})
                model.built=True
                # add to list of members
                all_models.append(model)
                print('>loaded %s' % filename)
        return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
	
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression()
	model.fit(stackedX, inputy)
	return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat



(trainX,trainy), (testX, testy) = importData()


trainX, origtrainy = preprocess(trainX,trainy)
testX, origtesty = preprocess(testX,testy)

trainy = np.array(keras.utils.to_categorical(origtrainy, totalLabel))
testy = np.array(keras.utils.to_categorical(origtesty, totalLabel))
#print(trainX.shape, testX.shape)
# load all models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
        _, acc,f1,precision, recall = model.evaluate(testX, testy, verbose=0)
        print('Model Accuracy: %.3f ' % acc, f1,precision, recall )
        #print (acc,f1,precision, recall)
# fit stacked model using the ensemble
#print('test x shape is ', testX.shape)
#print('test y shape is ', testy.shape)
ps =powerset([1,2,3])
for s in ps:
   tempModel =[]
   for num in s:
      print (s)
      tempModel.append(members[num-1])
   if len(tempModel) >1:
      stackedModel = fit_stacked_model(tempModel, testX, origtesty)
      # evaluate model on test set
      yhat = stacked_prediction(tempModel, stackedModel, testX)
      modelsuffix =""
      for i in range(len(tempModel)):
          modelsuffix=modelsuffix+str(tempModel[i])+"_"
      acc = accuracy_score(origtesty, yhat)
      print('Stacked Test Accuracy:',s,': %.3f' % acc)
      #dot_img_file = 'stack'+modelsuffix+'.png'
      #keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

