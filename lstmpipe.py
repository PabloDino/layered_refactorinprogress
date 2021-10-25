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
import settings
import utils

sys.path.insert(1, os.path.join(sys.path[0], '..'))

filepath = 'lstmCheck-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{val_f1_m:.2f}-{val_precision_m:.2f}-{val_recall_m:.2f}-{acc:.2f}-.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
cb = TimingCallback()

def fitCombined(xTrain,xTest, y_train, y_test,combModel,fixedLayers):
    if len(xTrain.shape)==5:
        xTrain = np.array([x.reshape( int(xTrain.shape[2]), int(xTrain.shape[3]), int(xTrain.shape[4]) ) for x in xTrain])
        xTest = np.array([x.reshape( int(xTest.shape[2]), int(xTest.shape[3]), int(xTest.shape[4]))  for x in xTest])        
    else:
      if len(xTrain.shape)==4:
        xTrain = np.array([x.reshape( int(xTrain.shape[2]),int(xTrain.shape[3]) ) for x in xTrain])
        xTest = np.array([x.reshape( (int(xTest.shape[2]), int(xTest.shape[3])))  for x in xTest])
      else:
        xTrain = np.array([x.reshape( int(xTrain.shape[2])) for x in xTrain])
        xTest = np.array([x.reshape( int(xTest.shape[2]))  for x in xTest])

    combModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

    indata = [xTrain,xTrain]
    combModel.fit(xTrain,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (xTest, y_test),
        callbacks=[checkpoint, cb]
    )
    print('finished fit at ', datetime.datetime.now())



    loss, acc, valacc,f1,precision, recall  = evaluateCheckPoints("lstmCheck")
    
    print('Loss for best accuracy:', loss)
    print('Best validation accuracy:', valacc)
    print('Best training accuracy:', acc)
    sumtime, avgtime, max_itertime,min_itertime = getAggregates(cb.logs)
  


    outfile=open("lstm.perf.txt","a")
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
    maxacc=0
    maxdx=0
    for dx in range(len(files)):
        arr = files[dx].split("-")
        if  float(arr[3])>maxvacc:
            maxvacc = float(arr[3])
            maxdx = dx
        if  float(arr[7])>maxacc:
            maxacc = float(arr[7])
    retloss = float(arr[2])
    retf1 = float(arr[4])
    retprecision = float(arr[5])
    retrecall = float(arr[6])
    for fle in files:
        os.remove(fle)
    return retloss,maxacc, maxvacc, retf1, retprecision, retrecall



def cloneBranchedModel(modelbase, startLayer,totalLabel):
    input_shape_a=modelbase.layers[startLayer-1].get_output_at(0).shape#(128, 128,1)
    nextLyr=modelbase.layers[startLayer-1].get_output_at(0)
    inLyr=None
    branchIn=None
    if len(input_shape_a)==4:
        inLyr = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        branchIn = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
    else:#     
      if len(input_shape_a)==3:
        inLyr = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        branchIn = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
      else:
         oldshape=K.int_shape(nextLyr)
         inLyr = Input(shape=(oldshape[1],))
         branchIn = Input(shape=(oldshape[1],))
    nextLyr=inLyr
    nextBr=branchIn
    numlyrs =len(modelbase.layers)
    currlyr=0
    for layer in modelbase.layers:
     if (currlyr >= startLayer):
      print(currlyr,':layer name is ', layer.name, ' ; ', nextLyr.name)
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
  
    lastDense = Dense(totalLabel)(nextLyr)
    out = Activation('softmax')(lastDense)
    newModel = Model(inputs= inLyr, outputs=out)

    return newModel,origBranch,otherBranch
    


if __name__ == '__main__':
    dataset =  importData('base',train=True)#(train, test) =  
    nextdataset =   importData('next',train=True)#(nextTrain,nextTest) = 
    random.shuffle(dataset)
    availCount= int(totalRecordCount*0.5)
    availset = dataset[:availCount]
    availset = mergeSets2(availset, nextdataset)#train, test, nextTrain, nextTest)
    availCount+=len(nextdataset)
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


    modelbase =keras.models.load_model('50p/Model.1.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
    #################################################################################
    
    orig_in = modelbase.layers[0].get_output_at(0)    
    modelout = modelbase.layers[-1].get_output_at(0)
    denseOut=Dense(totalLabel,name='denseout')(modelout)
    out=Activation('softmax',name='actout')(denseOut)
    for i in range(len(modelbase.layers[:-2]),0,-1):
      encPreModel = keras.models.clone_model(modelbase)
      encPreModel.build(orig_in)
      for j in range( len(modelbase.layers)-i):
          encPreModel._layers.pop()          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)
      X_train_encoded = encPredict(encModel,x_train)
      X_test_encoded = encPredict(encModel,x_test)
      startLyr =i
          
      modelNew,origBranch,otherBranch= cloneBranchedModel(modelbase, startLyr, totalLabel)

      try:
         fitCombined(X_train_encoded, X_test_encoded,  y_traincat, y_testcat, modelNew, startLyr)  
      except:
         print("Error fitting layer ", startLyr)