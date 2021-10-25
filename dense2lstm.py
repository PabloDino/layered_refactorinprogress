import utils
import settings
import denseBase
from denseBase import DenseBase
sys.path.insert(1, os.path.join(sys.path[0], '..'))

    
filepath = 'dense2lstm-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{val_f1_m:.2f}-{val_precision_m:.2f}-{val_recall_m:.2f}-{acc:.2f}-.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

def fitCombined(xTrain50,xTrain20,xTrain50p3,xTest50,xTest20,xTest50p3, y_train, y_test,combModel,fixedLayers1,fixedLayers3 ):

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

    if len(xTrain50p3.shape)==5:
        xTrain50p3 = np.array([x.reshape( int(xTrain50p3.shape[2]), int(xTrain50p3.shape[3]), int(xTrain50p3.shape[4]) ) for x in xTrain50p3])
        xTest50p3 = np.array([x.reshape( int(xTest50p3.shape[2]), int(xTest50p3.shape[3]), int(xTest50p3.shape[4]))  for x in xTest50p3])
    else:
      if len(xTrain50.shape)==4:
        xTrain50p3 = np.array([x.reshape( int(xTrain50p3.shape[2]),int(xTrain50p3.shape[3]) ) for x in xTrain50p3])
        xTest50p3 = np.array([x.reshape( (int(xTest50p3.shape[2]), int(xTest50p3.shape[3])))  for x in xTest50p3])
      else:
        xTest50p3 = np.array([x.reshape( int(xTest50p3.shape[2]),1) for x in xTest50p3])
        xTest50p3 = np.array([x.reshape( int(xTest50p3.shape[2]),1)  for x in xTest50p3])

    combModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

 
    indata = [xTrain50,xTrain20,xTrain50p3]
    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([xTest50,xTest20,xTest50p3], y_test),#,
        callbacks=[checkpoint, cb]
    )
    print('finished fit at ', datetime.datetime.now())


    loss,acc, valacc,f1,precision, recall  = evaluateCheckPoints("dense2lstm")

    
    
    print('Loss for best accuracy:', loss)
    print('Best validation accuracy:', valacc)
    print('Best training accuracy:', acc)
    sumtime, avgtime, max_itertime,min_itertime = getAggregates(cb.logs)
  
    outfile=open("denselstm.perf.txt","a")
    outfile.write(str(fixedLayers3)+","+ str(fixedLayers1)+","+ str(loss)+","+ str(acc)+","+ str(valacc) +","+str(f1)+","+str(precision)+","+str(recall)+","+str(sumtime)+","+str(avgtime)+","+str(max_itertime)+","+str(min_itertime)+"\n" )
    outfile.close()
    print('Model exported and finished')


def copyLSTMBranch(modlev, mod):
    numlyrs =len(mod.layers)
    currlyr=modlev
    for i in range (modlev,len(mod50p.layers)-1):
     layer=mod.layers[currlyr]
     if (currlyr >= mod1lev):
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
     
    numdims=1
    for i in range(len(K.int_shape(nextLyr))):
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
           
    nextLyr =Reshape((numdims,))(nextLyr) 
    return nextLyr 



#########################################################
def copyDenseBranch(modlev, mod, inLyr):

    nextLyr= inLyr
    preBatchLyr =None
    preBatchBr=None
    currlyr= modlev
    lyrs=[]
    inLyrs=[]
    for i in range (modlev,len(mod.layers)-1):
     layer = mod.layers[i]
     if (currlyr >= mod):
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
        lyrModel = Model(modlev,nextLyr)
        nextLyr = lyrModel.layers[-1].get_output_at(0)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)
     currlyr+=1
     return nextLyr


def mergeModels(mod50p, mod20p, mod50p3, mod1lev, mod3lev):
    input_shape_a=mod50p.layers[mod1lev-1].get_output_at(0).shape#(128, 128,1)
    input_shape_b=mod50p3.layers[mod3lev-1].get_output_at(0).shape#(128, 128,1)
    inLyr50p=None
    inLyr50p3=None
    inLyr20p=None
    nextLyrmod1=mod50p.layers[mod1lev-1].get_output_at(0)
    nextLyrmod3=mod50p3.layers[mod3lev-1].get_output_at(0)

    if len(input_shape_a)==4:
        inLyr50p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        inLyr20p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
    else:#     
      if len(input_shape_a)==3:
        inLyr50p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        inLyr20p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
      else:
         oldshape1=K.int_shape(nextLyrmod1)
         inLyr50p = Input(shape=(oldshape1[1],1))
         inLyr20p = Input(shape=(oldshape1[1],1))
         
    if len(input_shape_b)==4:
        inLyr50p3 = Input(shape=(int(input_shape_b[1]),int(input_shape_b[2]),int(input_shape_b[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
    else:#     
      if len(input_shape_a)==3:
        inLyr50p3 = Input(shape=(int(input_shape_b[1]),int(input_shape_b[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
      else:
         oldshape3=K.int_shape(nextLyrmod3)
         inLyr50p3 = Input(shape=(oldshape3[1],1))
                  
    nextLyr50p=inLyr50p
    nextLyr50p3=inLyr50p3
    nextLyr20p=inLyr20p
    ##################################################################
    nextLyr = Concatenate(name='inconCat',axis=1) ([nextLyr50p,nextLyr20p])
    if len(input_shape_a)==4:
       nextLyr = MaxPooling2D(pool_size=(2,1))(nextLyr)
    else:
       nextLyr = MaxPooling1D(pool_size=2)(nextLyr)



    nextLyr1 = copyLSTMBranch(mod1lev, mod50p)
    #####################################################
    nextLyr = copyBranch(mod3lev, mod50p3, inLyr50p3)
   
 
    nextLyr = BatchNormalization()(nextLyr)
    #x = BatchNormalization()(nextLyr)
    nextLyr = Activation('relu')(nextLyr)
    print('nextLyr shape is ', nextLyr.shape)
    nextLyr3 = GlobalAveragePooling2D()(nextLyr)


    concatC = Concatenate(name='outconCat') ([nextLyr1,nextLyr3])
  
    prediction = Dense(totalLabel, activation='softmax')(concatC)
    ###################################################
    newModel = Model(inputs= [inLyr50p, inLyr20p, inLyr50p3], outputs=prediction)
    newModel.summary()  
    return newModel 
    



###############START MAIN####################
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
nextRecordCount= len(nextdataset)

mod50p1 =keras.models.load_model('50p/Model.1.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
mod20p1 =keras.models.load_model('20p/Model.1.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
mod50p3 =keras.models.load_model('50p/Model.3.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
#################################################################################
for layer in mod50p1.layers:
    layer.name = layer.name + str("_1")
           
for layer in mod20p1.layers:
    layer.name = layer.name + str("_2")
       
for layer in mod50p3.layers:
    layer.name = layer.name + str("_3")

orig_in50p1 = mod50p1.layers[0].get_input_at(0)
orig_in20p1 = mod20p1.layers[0].get_input_at(0)
orig_in50p3 = mod50p3.layers[0].get_input_at(0)
     
trimmed50p1out = mod50p1.layers[-3].get_output_at(0)
trimmed20p1out = mod20p1.layers[-3].get_output_at(0)
trimmed50p3out = mod50p3.layers[-1].get_output_at(0)
  
trimmed50p1=Model(inputs=orig_in50p1,outputs=trimmed50p1out)
trimmed20p1=Model(inputs=orig_in20p1,outputs=trimmed20p1out)
trimmed50p3=Model(inputs=orig_in50p3,outputs=trimmed50p3out)
mod50p3.summary()

numtrain = x_train.shape[0]
numtest = x_test.shape[0]
c50p1Count = len(trimmed50p1.layers)
c20p1Count = len(trimmed20p1.layers)
c50p3Count = len(trimmed50p3.layers)

prev50p1= 0
prev20p1= 0
prev50p3= 0
init50p1=False
init20p1=False
init50p3=False

init=False

for i in range(len(mod50p3.layers)-3, 0, -5):
  c50p3Lev=i
  c50p1Lev = int(i/c50p3Count*c50p1Count)
  if c50p1Lev<3:
      c50p1Lev=3
  c20p1Lev = c50p1Lev
    
  encPreModel = keras.models.clone_model(mod50p1)
  encPreModel.build(orig_in50p1)
  encPreModel.summary()
  print('PRESUMMARY',  c50p1Lev, c50p3Lev,init20p1,init20p1,init50p3)
  for j in range( len(mod50p1.layers)-c50p1Lev):
      encPreModel._layers.pop()
  encPreModel.summary()
  
  if  not c50p1Lev== prev50p1:       
    init50p1 =True
    encin = encPreModel.layers[0].get_output_at(0)
    encout = encPreModel.layers[-1].get_output_at(0)
    encModel=Model(inputs=encin,outputs=encout)
    enc50p1 = keras.models.clone_model(encModel)
    X_train50p1_encoded = encPredict(encModel,x_train)
    X_test50p1_encoded = encPredict(encModel,x_test)
                  
  ###########20p#############################
  encPreModel = keras.models.clone_model(mod20p1)
     
  encPreModel.build(orig_in20p1)
  #encPreModel.summary()
  print('PRESUMMARY',  c50p1Lev, c50p3Lev,init20p1,init20p1,init50p3)
  for j in range( len(mod20p1.layers)-c20p1Lev):
      encPreModel._layers.pop()
          
  if  not c20p1Lev== prev20p1:       
    init20p1 =True
    encin = encPreModel.layers[0].get_output_at(0)
    encout = encPreModel.layers[-1].get_output_at(0)
    encModel=Model(inputs=encin,outputs=encout)
    #encModel.summary()
    enc20p1 = keras.models.clone_model(encModel)
    X_train20p_encoded = encPredict(encModel,x_train)
    X_test20p_encoded = encPredict(encModel,x_test)
    ###########50p#############################
      
  encPreModel = keras.models.clone_model(mod50p3)
  encPreModel.build(orig_in50p3)
  print('PRESUMMARY',  c50p1Lev, c50p3Lev,init20p1,init20p1,init50p3)
  for j in range( len(mod50p3.layers)-c50p3Lev):
      encPreModel._layers.pop()
          
  if  (not c50p3Lev== prev50p3) and (c50p3Lev<= 126):       
    init50p3 =True
    encin = encPreModel.layers[0].get_output_at(0)
    encout = encPreModel.layers[-1].get_output_at(0)
    encModel=Model(inputs=encin,outputs=encout)
    enc50p3 = keras.models.clone_model(encModel)
    X_train50p3_encoded = encPredict(encModel,x_train)
    X_test50p3_encoded = encPredict(encModel,x_test)
   ###########50p#############################    


  init = init50p1 and init20p1 and init50p3    
  if init:   
     try:
         combModel = mergeModels(mod50p1, mod20p1, mod50p3, c50p1Lev, c50p3Lev)
         fitCombined(X_train50p1_encoded, X_train20p_encoded, X_train50p3_encoded, X_test50p1_encoded, X_test20p_encoded, X_test50p3_encoded, y_traincat, y_testcat, combModel, c50p3Lev,c50p1Lev)
     except: 
          print('next time @', c50p3Lev)  
    

    
