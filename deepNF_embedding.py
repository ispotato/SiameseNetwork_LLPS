from keras.optimizers import SGD
from keras.layers import Input, Dense, concatenate,Conv1D,MaxPooling1D,Flatten
import tensorflow as tf
from keras.regularizers import l1,l2
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.callbacks import EarlyStopping
import os
import numpy as np

def build_AE(input_dim, encoding_dims):
    """
    Function for building autoencoder.
    """
    # input layer
    input_layer = Input(shape=(input_dim, ))
    hidden_layer = input_layer
    for i in range(0, len(encoding_dims)):
        # generate hidden layer
        if i == int(len(encoding_dims)/2):
            hidden_layer = Dense(encoding_dims[i],
                                 activation='relu', #'sigmoid',
                                 activity_regularizer=l1(10e-6),
                                 kernel_regularizer=l1(1e-5),
                                 #bias_regularizer=l1(0.0001),
                                 name='middle_layer')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 activation='relu',
                                 activity_regularizer=l1(10e-6),
                                 #kernel_regularizer=l1(1e-5),
                                 #bias_regularizer=l1(0.0001),
                                 name='layer_' + str(i+1))(hidden_layer)

    # reconstruction of the input
    decoded = Dense(input_dim,activation='relu')(hidden_layer)

    # autoencoder model
    #sgd = SGD(lr=0.01, momentum=0.6, decay=0.0, nesterov=False)
    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer='Adam', loss='binary_crossentropy')
    print (model.summary())

    return model

def build_MDA(input_dims, encoding_dims):
    """
    Function for building multimodal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(int(encoding_dims[0]/len(input_dims)),
                                   activation='sigmoid')(input_layers[j]))

    # Concatenate layers
    if len(encoding_dims) == 1:
        hidden_layer = concatenate(hidden_layers, name='middle_layer')
    else:
        #hidden_layer = concatenate(hidden_layers)
        hidden_layer= tf.concat(hidden_layers, axis=3)

    # middle layers
    for i in range(1, len(encoding_dims)-1):
        if i == int(len(encoding_dims)/2):
            hidden_layer = Dense(encoding_dims[i],
                                 name='middle_layer',
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)

    if len(encoding_dims) != 1:
        # reconstruction of the concatenated layer
        hidden_layer = Dense(encoding_dims[0],
                             activation='sigmoid')(hidden_layer)

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(int(encoding_dims[-1]/len(input_dims)),
                                   activation='sigmoid')(hidden_layer))
    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   activation='sigmoid')(hidden_layers[j]))

    # autoencoder model
    sgd = SGD(lr=0.02, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model

def create_dir(path):
    # 创建单个目录
    try:
        os.mkdir(path)
    except FileExistsError:
        print('Directory already exists.')

def build_model(X, input_dims, arch, std=1.0, epochs=80, batch_size=64):
    model = build_AE(input_dims[0], arch)

    noise_factor = 0.5
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=0.2)
        X_train = []
        X_test = []
        for jj in range(0, len(Xs), 2):
            X_train.append(Xs[jj])
            X_test.append(Xs[jj+1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)
        for ii in range(0, len(X_train)):
            X_train_noisy[ii] = X_train_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train[ii].shape)
            X_test_noisy[ii] = X_test_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test[ii].shape)
            X_train_noisy[ii] = np.clip(X_train_noisy[ii], 0, 1)
            X_test_noisy[ii] = np.clip(X_test_noisy[ii], 0, 1)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2)
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
    # Fitting the model
    history = model.fit(X_train_noisy, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(X_test_noisy, X_test),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)],verbose=2)
    mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

    return mid_model, history
