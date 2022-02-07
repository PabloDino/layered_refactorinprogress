from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Conv1D,Conv2D, GlobalAveragePooling2D, InputLayer, \
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate,concatenate, Input, AveragePooling2D
from keras.layers.normalization import BatchNormalization                 
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


from timeit import default_timer as timer


# Your data source for wav files
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-base50/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-Base50p/'
baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-base50p/'
#baseFolder = '/home/paul/Downloads/ESC-50-tst2/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-Next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-next30p/'
nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-next30p/'
#nextFolder = '/home/paul/Downloads/ESC-50-tst2b/'
#lastFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-last20p/'
lastFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-last20p/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

dataSourceBase=baseFolder#lastFolder
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
#totalLabel = 10

# model parameters for training
#batchSize = 128
#epochs = 100
#latent_dim=8
dataSize=128

#timesteps = 128 # Length of your sequences
#input_dim = 128 

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


cb = TimingCallback()



def encPredict(enc, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   z_mean=[]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      #z_mean8, _, _ = enc.predict([[sample, sample]])
      z_mean8 = enc.predict(sample)
      z_mean.append(z_mean8)
      if (i==0):
         print('INPUT:::')
         print(sample.shape)
         print('OUTPUT:::')
         print(z_mean8.shape)
      #print('Sample ', i, ' shape ', sample.shape , ' converted to ', z_mean8.shape)
      #if (i==0):
      #  z_mean=z_mean8[0]
      #else:
      #  z_mean = np.concatenate((z_mean,z_mean8[0]))#, axis=0)
      #if True:#(i%200==0) and i>1:  
      #  print("enc stat",z_mean.shape)
   z_mean=np.array(z_mean) 
   return z_mean


def decPredict(dec, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      #sample = np.reshape(sample, (sample.shape[1], sample.shape[2]))
      #print(sample.shape)
      z_mean8 = dec.predict(sample)
      #z_mean8, _, _ = dec.predict([[sample, sample]])
      if (i==0):
        z_mean=z_mean8
      else:
        z_mean = np.concatenate((z_mean,z_mean8), axis=0)
      if (i%200==0) and i>1:  
        print("dec stat",z_mean.shape)
   return z_mean


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



def preprocess(array, labels):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    lookback = 1#latent_dim
    array=np.array(array)
    maxi=0
    #for i in range(array.shape[0]):
    #   if (maxi<np.max(array[i]):
    #       maxi= np.max(array[i])
    print("arrshape1:", array.shape)
    #print("labshape:", labels)
    #array, labels =  temporalize(array, labels, lookback)
    print("arrshape2:", array.shape)
    array = np.array(array).astype("float32") / np.max(array)
    array = np.reshape(array, (lookback*len(array), dataSize, dataSize,1))
    labels = np.array(labels) 
    return array, labels


def temporalize(X, y, lookback):
    '''
    Inputs
    X         A 3D numpy array ordered by time of shape: 
              (n_observations x steps_per_ob x n_features)
    y         A 1D numpy array with indexes aligned with 
              X, i.e. y[i] should correspond to X[i]. 
              Shape: n_observations.
    lookback  The window size to look back in the past 
              records. Shape: a scalar.

    Output
    output_X  A 4D numpy array of shape: 
              ((n_observations-lookback-1) x steps_per_ob x lookback x 
              n_features)
    output_y  A 1D array of shape: 
              (n_observations-lookback-1), aligned with X.
    '''
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        print('look', i, len(output_X), len(output_y))
        t=[]
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
            output_y.append(y[i + lookback + 1])
        output_X.append(t)
    #return np.array(output_X), np.array(output_y)
    return np.squeeze(np.array(output_X)), np.array(output_y)





# This function will import wav files by given data source path.
# And will extract wav file features using librosa.feature.melspectrogram.
# Class label will be extracted from the file name
# File name pattern: {WavFileName}-{ClassLabel}
# e.g. 0001-0 (0001 is the name for the wav and 0 is the class label)
# The program only interested in the class label and doesn't care the wav file name
def importData(setname='full'):
    dataSet = []
    lblmap ={}
    lblid=0
    totalCount = 0
    if (setname) == 'full':
        dataSourceBase=baseFolder
        totalLabel=50
   
    if (setname) == 'base':
        dataSourceBase=baseFolder
    if (setname) == 'next':
        dataSourceBase=nextFolder
    dirlist = os.listdir(dataSourceBase)
    for dr in dirlist:
      dataSource = os.path.join(dataSourceBase,dr)
      for root, _, files in os.walk(dataSource):
        for file in files:
            fileName, fileExtension = os.path.splitext(file)
            if fileExtension != '.wav': continue
            wavFilePath = os.path.join(root, file)
            y, sr = librosa.load(wavFilePath, duration=2.97)
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            if ps.shape != (128, 128): continue
            
            # extract the class label from the FileName
            label0 = dr.split('-')[1]
            if label0 not in lblmap:
               lblmap[label0] =lblid
               lblid+=1
            label=lblmap[label0]
            #label = dr#fileName.split('-')[1]
            print(fileName, label0, label)
            dataSet.append( (ps, label) )
            totalCount += 1
    f = open('dict50.csv','w')
    f.write("classID,class")
    for lb in lblmap:
       f.write(str(lblmap[lb])+','+lb)
    f.close()

    global totalRecordCount
    totalRecordCount = totalCount
    
    print('TotalCount: {}'.format(totalRecordCount))
    trainDataEndIndex = int(totalRecordCount*0.8)
    random.shuffle(dataSet)

    train = dataSet[:trainDataEndIndex]
    test = dataSet[trainDataEndIndex:]

    print('Total training data:{}'.format(len(train)))
    print('Total test data:{}'.format(len(test)))

    # Get the data (128, 128) and label from tuple
    print("train 0 shape is ",train[0][0].shape)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    

    return (X_train, y_train), (X_test, y_test)#dataSet

    
def getAggregates(logs):
   mx=0
   mn=10000
   for i in range(len(logs)):
       if (logs[i]>mx):
          mx=logs[i]
       if (logs[i]<mn):
          mn=logs[i]
   sumtime = sum(logs)
   avgtime = 1.0*sumtime/len(logs)

   return sumtime, avgtime, mx,mn    
    
    
def evaluateCheckPoints(prefix):
    files=[]
    for fle in os.listdir():
        if fle.startswith(prefix):
            files.append(fle)
    maxvacc=0
    maxdx=0
    for dx in range(len(files)):
        arr = files[dx].split("-")
        if  float(arr[3])>maxvacc:
            maxvacc = float(arr[3])
            maxdx = dx
    arr = files[maxdx].split("-")
    retloss = float(arr[2])
    retf1 = float(arr[4])
    retprecision = float(arr[5])
    retrecall = float(arr[6])
    acc = float(arr[7])
    for fle in files:
        os.remove(fle)
    return retloss,acc, maxvacc, retf1, retprecision, retrecall


def findLayer(lst, layer):
    pos=-1
    namecheck = layer.name.split('/')[0]
    for i in range(len(lst)):
        if (lst[i].name.split('/')[0]==namecheck):
           pos=i
    return pos
    
def mergeSets(dset,  nextdset):
    combSet =[]
    for i in range(max(len(dset),len(nextdset))):
        if (i<len(dset)):
           combSet.append(dset[i])
        if (i<len(nextdset)):
           combSet.append(nextdset[i])
    return combSet

'''
   
def getAggregates(logs):
   mx=0
   mn=10000
   for i in range(len(logs)):
       if (logs[i]>mx):
          mx=logs[i]
       if (logs[i]<mn):
          mn=logs[i]
   sumtime = sum(logs)
   avgtime = 1.0*sumtime/len(logs)

   return sumtime, avgtime, mx,mn    
    
    
def evaluateCheckPoints(prefix):
    files=[]
    for fle in os.listdir():
        if fle.startswith(prefix):
            files.append(fle)
    maxvacc=0
    maxdx=0
    for dx in range(len(files)):
        arr = files[dx].split("-")
        if  float(arr[3])>maxvacc:
            maxvacc = float(arr[3])
            maxdx = dx
    arr = files[maxdx].split("-")
    retloss = float(arr[2])
    retf1 = float(arr[4])
    retprecision = float(arr[5])
    retrecall = float(arr[6])
    acc = float(arr[7])
    for fle in files:
        os.remove(fle)
    return retloss,acc, maxvacc, retf1, retprecision, retrecall

'''
