import utils
import settings
import denseBase
from denseBase import DenseBase




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
 
    indata = [xTrain50,xTrain20]

    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([xTest50,xTest20], y_test),#,
        callbacks=[checkpoint, cb]
    )
    print('finished fit at ', datetime.datetime.now())


    loss , acc, valacc,f1,precision, recall  = evaluateCheckPoints("DenseRep")
    
    print('Loss for best accuracy:', loss)
    print('Best validation accuracy:', valacc)
    print('Best training accuracy:', acc)
    sumtime, avgtime, max_itertime,min_itertime = getAggregates(cb.logs)


    outfile=open("denserep.perf.txt","a")
    outfile.write(str(fixedLayers)+","+ str(acc)+","+ str(valacc) +","+str(f1)+","+str(precision)+","+str(recall)+","+str(sumtime)+","+str(avgtime)+","+str(max_itertime)+","+str(min_itertime)+","+ str(loss)+"\n" )
    outfile.close()
    print('Log completed for layer '+str(fixedLayers)) 
    

    
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
    print("b4 concat",nextLyr50p.shape)
    nextLyr = Concatenate(name='inconCat',axis=1) ([nextLyr50p,nextLyr20p])
    if len(input_shape_a)==4:
       nextLyr = MaxPooling2D(pool_size=(2,1))(nextLyr)
    else:
       nextLyr = MaxPooling1D(pool_size=2)(nextLyr)
    
    numlyrs =len(mod50p.layers)
    currlyr=0
    for layer in mod50p.layers:
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
    x = BatchNormalization()(nextLyr)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(totalLabel, activation='softmax')(x)
    newModel = Model(inputs= [inLyr50p, inLyr20p], outputs=prediction)

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

    random.shuffle(nextdataset)

    mod50p =keras.models.load_model('50p/Model.3.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
    mod20p =keras.models.load_model('20p/Model.3.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
    #modelbase.summary()
    #################################################################################
    for layer in mod50p.layers:
        layer.name = layer.name + str("_1")
           
    for layer in mod20p.layers:
        layer.name = layer.name + str("_2")
       
    orig50_in = mod50p.layers[0].get_output_at(0)
    orig20_in = mod20p.layers[0].get_output_at(0)
    
    for i in range( len(mod50p.layers[:-1])-2,1,-5):
    
      encPreModel = keras.models.clone_model(mod50p)
      
      encPreModel.build(orig50_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      for j in range( len(mod50p.layers)-i):
          encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)
      #encModel.summary()
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
      
    
      combModel = mergeModels(enc50p,enc20p, startLyr)

      try:
        fitCombined(X_train50p_encoded, X_train20p_encoded, X_test50p_encoded, X_test20p_encoded, y_traincat, y_testcat, combModel, startLyr)
      except: 
        print('next time @', startLyr)  
    

    
