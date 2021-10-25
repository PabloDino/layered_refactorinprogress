import utils

import settings



def buildModel(X_train, X_test, y_train, y_test):
 
    Xb_train = X_train.copy()#np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    Xb_test = X_test.copy()#np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])

    

    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    model_b = Sequential()

    # Model Input
    l_input_shape=(128, 128,1,1)
    input_shape=(128, 128,1)


    model_b_in = Input(shape=input_shape)
    print(model_b_in.shape)

    conv_1b = Conv2D(24, (latent_dim,latent_dim), strides=(1, 1), input_shape=input_shape)(model_b_in)
    pool_2b = MaxPooling2D((latent_dim,latent_dim), strides=(latent_dim,latent_dim))(conv_1b)
    conv_3b = Conv2D(48, (latent_dim,latent_dim), strides=(1, 1), input_shape=input_shape)(pool_2b)
    act_3b =Activation('relu')(conv_3b)
    re_4b = Reshape(target_shape=(latent_dim*latent_dim,48),input_shape=(latent_dim,latent_dim,48))(act_3b)
    
    ls_5b= LSTM(latent_dim*latent_dim,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_4b)
    flat12 = Flatten()(ls_5b)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=model_b_in, outputs=out)  
    
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
    indata = [X_train]
    filepath = "lstm.{epoch:02d}-{loss:.2f}.hdf5"
    model.fit(X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test)
    )
    score = model.evaluate(X_test,
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'Incremental/20p/Model.1.'.format(timestr)
    modelName =modelName+".hdf5"
    model.save('{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    (train_data,train_labels), (test_data, test_labels) = importData()#.load_data()
    train_data, train_labels = preprocess(train_data,train_labels)
    test_data, test_labels = preprocess(test_data,test_labels)
    buildModel(train_data, test_data, train_labels, test_labels)
    
