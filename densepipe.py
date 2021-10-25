import utils
import settings

import denseBase
from denseBase import DenseBase


def fitCombined(xTrain,xTest, y_train, y_test,combModel,fixedLayers):
    print('xtrain shape1:',xTrain.shape)
    print('xTest shape1:',xTest.shape)
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
    initial_learning_rate = 0.01
    indata = [xTrain,xTrain]
    combModel.fit(xTrain,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (xTest, y_test),#,
        callbacks=[checkpoint,cb]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )
    loss, acc, valacc,f1,precision, recall = evaluateCheckPoints("DenseCheck")
    #return retloss,maxacc, maxvacc, retf1, retprecision, retrecall    
    #print(cb.logs)
    sumtime, avgtime, max_itertime,min_itertime = getAggregates(cb.logs)
    print('Loss for best accuracy:', loss)
    print('Best accuracy:', acc, str(acc))
    print('Best validation accuracy:', valacc)
    print('Average time:', avgtime)



    outfile=open("densepipe.perf.txt","a")#append_write)
    outfile.write(str(fixedLayers)+","+ str(acc)+","+ str(valacc) +","+str(f1)+","+str(precision)+","+str(recall)+","+str(sumtime)+","+str(avgtime)+","+str(max_itertime)+","+str(min_itertime)+","+ str(loss)+"\n" )
    outfile.close()
    print('Log completed for layer '+str(fixedLayers))
    


def cloneBranchedModel(modelbase, startLayer,totalLabel):
    inLyrs=[]
    lyrs=[]
    input_shape_a=modelbase.layers[startLayer].get_output_at(0).shape#(128, 128,1)
    nextLyr=modelbase.layers[startLayer-1].get_output_at(0)
    
    inLyr=None
    if len(input_shape_a)==4:
        inLyr = Input(shape=( int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
    else:#     
      if len(input_shape_a)==3:
        inLyr = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
      else:
        oldshape=K.int_shape(nextLyr)
        inLyr = Input(shape=(oldshape[1],))
    nextLyr=inLyr
    numlyrs =len(modelbase.layers)
    currlyr=0
    preBatchLyr =None
    for i in range (startLayer,len(modelbase.layers)-1):
     layer = modelbase.layers[i]
     if (currlyr >= startLayer):
      if True:#currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.Conv2D):
           nextLyr = Conv2D(filters=layer.filters,kernel_size=layer.kernel_size, padding=layer.padding, strides=layer.strides,  kernel_regularizer=layer.kernel_regularizer)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Concatenate):
           pos0 =findLayer(lyrs, layer.get_input_at(0)[0])
           pos1 =findLayer(lyrs, layer.get_input_at(0)[1])
           nextLyr = Concatenate()([inLyrs[pos0],inLyrs[pos1]])
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units,name ='dense_orig_'+str(i))(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
        if isinstance(layer, keras.layers.BatchNormalization):           
           if (preBatchLyr==None):
              nextLyr = BatchNormalization()(nextLyr)
           else:
              nextLyr = BatchNormalization()(preBatchLyr)
        if isinstance(layer, keras.layers.AveragePooling2D):
           nextLyr = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(nextLyr)
        lyrModel = Model(inLyr,nextLyr)
        preBatchLyr = lyrModel.layers[-1].get_output_at(0)
        moreinputs = True
        outdx=0
        lyrModel = Model(inLyr,nextLyr)
        outLyr = lyrModel.layers[-1].get_output_at(0)
        nextLyr = lyrModel.layers[-1].get_output_at(0)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)
     currlyr+=1 
  
    x = BatchNormalization()(nextLyr)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(totalLabel, activation='softmax')(x)
        


    newModel = Model(inLyr, prediction)
    return newModel
    
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


    modelbase =keras.models.load_model('50p/Model.3.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#,  custom_objects={'tf': tf})
    modelbase.summary()
    #################################################################################
    
    orig_in = modelbase.layers[0].get_output_at(0)
    
    
    

    print('ytraincat', y_traincat.shape)
    print('ytestcat', y_testcat.shape)

    
    modelout = modelbase.layers[-1].get_output_at(0)
    
    denseOut=Dense(totalLabel,name='denseout')(modelout)
    out=Activation('softmax',name='actout')(denseOut)
    for i in range( len(modelbase.layers[:-1])-2,1,-5):
      encPreModel = keras.models.clone_model(modelbase)
      modcopy  = keras.models.clone_model(modelbase)
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      for j in range( len(modelbase.layers)-i-1):
          encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)
      
      print('ENC MODEL', i)
      X_train_encoded = encPredict(encModel,x_train)
      X_test_encoded = encPredict(encModel,x_test)
      startLyr =i
      try:
        modelNew= cloneBranchedModel(modelbase, startLyr, totalLabel)
        encModel.summary()
        fitCombined(X_train_encoded, X_test_encoded,  y_traincat, y_testcat, modelNew, startLyr)  
        print('model cloned for layer ', i)
      except:
        print('error cloning model  layer ', i)
         
