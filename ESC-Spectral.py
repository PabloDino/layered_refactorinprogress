import utils
import settings

def buildModel(dataset):
    trainDataEndIndex = int(totalRecordCount*0.8)
    random.shuffle(dataset)

    train = dataset[:trainDataEndIndex]
    test = dataset[trainDataEndIndex:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])

    Xb_train = X_train.copy()#np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    Xb_test = X_test.copy()#np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])

    y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

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

    
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

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

    modelName = 'Incremental/20p/Model.2.final.hdf5'
    model.save(modelName)


    print('Model exported and finished')

if __name__ == '__main__':
    dataSet = importData()
    buildModel(dataSet)
    
