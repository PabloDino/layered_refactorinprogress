from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
#from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Conv1D,Conv2D, GlobalAveragePooling2D, InputLayer, \
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.layers.normalization import BatchNormalization                      
import tensorflow as tf
import sys, os
import settings
import utils

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import denseBase
from denseBase import DenseBase

from time import time
import numpy as np
#import matplotlib.pyplot as plt
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




def encPredict(enc, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   z_mean=[]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      #z_mean8, _, _ = enc.predict([[sample, sample]])
      z_mean8 = enc.predict(sample)
      z_mean.append(z_mean8)
   z_mean=np.array(z_mean) 
   return z_mean


def decPredict(dec, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      z_mean8 = dec.predict(sample)
      if (i==0):
        z_mean=z_mean8
      else:
        z_mean = np.concatenate((z_mean,z_mean8), axis=0)
      if (i%200==0) and i>1:  
        print("dec stat",z_mean.shape)
   return z_mean


 
def mergeSets2(dset,  nextdset):
    combSet =[]
    for i in range(max(len(dset),len(nextdset))):
        if (i<len(dset)):
           combSet.append(dset[i])
        if (i<len(nextdset)):
           combSet.append(nextdset[i])
    return combSet
        


filepath = 'lstmrepCheck-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{val_f1_m:.2f}-{val_precision_m:.2f}-{val_recall_m:.2f}-{acc:.2f}-.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
cb = TimingCallback()

def fitCombined(xTrain50,xTrain20,xTest50,xTest20, y_train, y_test,combModel,fixedLayers):

    if len(xTrain50.shape)==5:
        xTrain50 = np.array([x.reshape( int(xTrain50.shape[2]), int(xTrain50.shape[3]), int(xTrain50.shape[4]) ) for x in xTrain50])
        xTest50 = np.array([x.reshape( int(xTest50.shape[2]), int(xTest50.shape[3]), int(xTest50.shape[4]))  for x in xTest50])
        xTrain20 = np.array([x.reshape( int(xTrain20.shape[2]), int(xTrain20.shape[3]), int(xTrain20.shape[4]) ) for x in xTrain20])
        xTest20 = np.array([x.reshape( int(xTest20.shape[2]), int(xTest20.shape[3]), int(xTest20.shape[4]))  for x in xTest20])
    else:
      if len(xTrain50.shape)==4:
        xTrain50 = np.array([x.reshape( int(xTrain50.shape[2]),int(xTrain50.shape[3]) ) for x in xTrain50])
        xTest50 = np.array([x.reshape( (int(xTest50.shape[2]), int(xTest50.shape[3])))  for x in xTest50])
        xTrain20 = np.array([x.reshape( int(xTrain20.shape[2]),int(xTrain20.shape[3]) ) for x in xTrain20])
        xTest20 = np.array([x.reshape( (int(xTest20.shape[2]), int(xTest20.shape[3])))  for x in xTest20])
      else:
        xTrain50 = np.array([x.reshape( int(xTrain50.shape[2]),1) for x in xTrain50])
        xTest50 = np.array([x.reshape( int(xTest50.shape[2]),1)  for x in xTest50])
        xTrain20 = np.array([x.reshape( int(xTrain20.shape[2]),1) for x in xTrain20])
        xTest20 = np.array([x.reshape( int(xTest20.shape[2]),1)  for x in xTest20])

    combModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

    initial_learning_rate = 0.01
    #epochs = 100
    drop = 0.75
    epochs_drop = 10.0
    decay = initial_learning_rate / epochs
    def lr_time_based_decay(epoch, lr):
       if epoch < 50:
            return initial_learning_rate
       else:
            lrate = initial_learning_rate * math.pow(drop,  
             math.floor((1+epoch)/epochs_drop))
       return lrate
 
    indata = [xTrain50,xTrain20]

    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([xTest50,xTest20], y_test),#,
        callbacks=[checkpoint, cb]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )
    print('finished fit at ', datetime.datetime.now())


    loss, acc, valacc,f1,precision, recall  = evaluateCheckPoints("lstmrepCheck")
    
    print('Loss for best accuracy:', loss)
    print('Best validation accuracy:', valacc)
    print('Best training accuracy:', acc)
    sumtime, avgtime, max_itertime,min_itertime = getAggregates(cb.logs)
    print('sumtime, avgtime, max_itertime,min_itertime :', sumtime, avgtime, max_itertime,min_itertime )
  


    outfile=open("lstmrep.perf.txt","a")
    outfile.write(str(fixedLayers)+","+ str(loss)+","+ str(acc)+","+ str(valacc) +","+str(f1)+","+str(precision)+","+str(recall)+","+str(sumtime)+","+str(avgtime)+","+str(max_itertime)+","+str(min_itertime)+"\n" )
    outfile.close()
    print('Model exported and finished')
    
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



def mergeModels(mod50p, mod20p,startLayer):

    input_shape_a=mod50p.layers[startLayer-1].get_output_at(0).shape#(128, 128,1)
    nextLyr=mod50p.layers[startLayer-1].get_output_at(0)
    inLyr50p=None
    inLyr20p=None
    if len(input_shape_a)==4:
        inLyr50p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        inLyr20p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
    else:#     
      if len(input_shape_a)==3:
        inLyr50p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        inLyr20p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
      else:
         oldshape=K.int_shape(nextLyr)
         inLyr50p = Input(shape=(oldshape[1],1))
         inLyr20p = Input(shape=(oldshape[1],1))
    nextLyr50p=inLyr50p
    nextLyr20p=inLyr20p
    nextLyr = Concatenate(name='inconCat',axis=1) ([nextLyr50p,nextLyr20p])
    if len(input_shape_a)==4:
       nextLyr = MaxPooling2D(pool_size=(2,1))(nextLyr)
    else:
       nextLyr = MaxPooling1D(pool_size=2)(nextLyr)
    
    numlyrs =len(mod50p.layers)
    currlyr=0
    for layer in mod50p.layers:
     if (currlyr >= startLayer):
      if not isinstance(nextLyr, keras.layers.Reshape):
          print(nextLyr.shape, '==>',str(layer.name), layer.get_output_at(0).shape)
      if currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.InputLayer):
           nextLyr = layer.get_output_at(0)
        if isinstance(layer, keras.layers.Conv2D):
           nextLyr = Conv2D(layer.filters,layer.kernel_size)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units)(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
     currlyr+=1 
    numdims=1
    for i in range(len(K.int_shape(nextLyr))):
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
           
    nextLyr =Reshape((numdims,))(nextLyr) 
    lastDense = Dense(totalLabel)(nextLyr)
    print('lastDense shape is ', lastDense.shape)
    out = Activation('softmax')(lastDense)
    newModel = Model(inputs= [inLyr50p, inLyr20p], outputs=out)
    newModel.summary()  
    return newModel 
    
def replicateListToMatch(inList, reqSize):
    outList =[]
    dx=0
    while (len(outList)<reqSize):
         print('dx=',dx)
         if (dx >=len(inList)):
            outList.append(inList[dx %len(inList)])
         else:
            outList.append(inList[dx])
         dx+=1
    return np.array(outList)

if __name__ == '__main__':
    #tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))
    dataset =  importData('base',train=True)#(train, test) =  
    print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    nextdataset =   importData('next',train=True)#(nextTrain,nextTest) = 
    print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    random.shuffle(dataset)
    availCount= int(totalRecordCount*0.5)
    availset = dataset[:availCount]
    #availset = replicateListToMatch(availset,len(nextdataset)/2)
    availset = mergeSets2(availset, nextdataset)#train, test, nextTrain, nextTest)
    availCount+=len(nextdataset)

    print('AvailCount: {}'.format(availCount))
    trainDataEndIndex = int(availCount*0.8)
    random.shuffle(availset)
    train = availset[:trainDataEndIndex]
    test = availset[trainDataEndIndex:]
    x, y = zip(*availset)
    x_train = x[:trainDataEndIndex]
    x_test = x[trainDataEndIndex:]   
    ycat = to_categorical(y)
    y_traincat = ycat[:trainDataEndIndex]
    y_testcat = ycat[trainDataEndIndex:]   

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255





    random.shuffle(nextdataset)
    nextRecordCount= len(nextdataset)

    print('nextRecordCount: {}'.format(nextRecordCount))
    nxttrainDataEndIndex = int(nextRecordCount*0.8)
    nx, ny = zip(*nextdataset)
    nx_train = nx[:nxttrainDataEndIndex]
    nx_test = nx[nxttrainDataEndIndex:]   
    nycat = to_categorical(ny)
    ny_traincat = nycat[:nxttrainDataEndIndex]
    ny_testcat = nycat[nxttrainDataEndIndex:]   

    nx_train = np.array(nx_train)
    nx_test = np.array(nx_test)
    nx_train = np.expand_dims(nx_train,-1)
    nx_test = np.expand_dims(nx_test,-1)
    nx_train = nx_train.astype('float32') / 255
    nx_test = nx_test.astype('float32') / 255



    image_size = nx_train[0].shape

    #encfine.built=True
    #enccoarse.built=True
    #print(os.getcwd())
    mod50p =keras.models.load_model('50p/Model.1.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
    mod20p =keras.models.load_model('20p/Model.1.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
    #modelbase.summary()
    #################################################################################
    for layer in mod50p.layers:
        layer.name = layer.name + str("_1")
           
    for layer in mod20p.layers:
        layer.name = layer.name + str("_2")
       
    orig50_in = mod50p.layers[0].get_output_at(0)
    orig20_in = mod20p.layers[0].get_output_at(0)
    
    

    for i in range(len(mod50p.layers[:-2]),0,-1):
      encPreModel = keras.models.clone_model(mod50p)
      
      encPreModel.build(orig50_in)

      for j in range( len(mod50p.layers)-i):
          encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)

      enc50p = keras.models.clone_model(encModel)
      X_train50p_encoded = encPredict(encModel,x_train)
      X_test50p_encoded = encPredict(encModel,x_test)
                  
      ###########20p#############################
      encPreModel = keras.models.clone_model(mod20p)
      
      encPreModel.build(orig20_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      for j in range( len(mod20p.layers)-i):
          encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)
      #encModel.summary()
      enc20p = keras.models.clone_model(encModel)
      X_train20p_encoded = encPredict(encModel,x_train)
      X_test20p_encoded = encPredict(encModel,x_test)

      startLyr =i
      
    
      combModel = mergeModels(enc20p, enc50p, startLyr)
      if True:#
        fitCombined(X_train50p_encoded, X_train20p_encoded, X_test50p_encoded, X_test20p_encoded, y_traincat, y_testcat, combModel, startLyr)
