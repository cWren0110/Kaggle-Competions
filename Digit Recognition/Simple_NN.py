# -*- coding: utf-8 -*-
"""
@author: Ethan
"""

from keras import regularizers
from keras.optimizers import SGD
from keras.models import Model
from keras.layers.core import Dense, Flatten
from keras.layers import Input
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_simple_model(input_shape,num_classes):
    defaults = dict(use_bias=True,
                    kernel_initializer='he_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(0.0005),
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    )
    
    data = Input(shape=input_shape, dtype=np.float32, name='data')
    flatten = Flatten(name='flatten')(data)
    hidden_neurons = 16
#----------------Change--------------------------------------------------------
#   By doubling the number of neurons (16=>32) we improve the results by ~1% 
#    hidden_neurons = 32
#------------------------------------------------------------------------------
    dense1 = Dense(hidden_neurons, activation='sigmoid', name='dense1', **defaults)(flatten)
    dense2 = Dense(hidden_neurons, activation='sigmoid', name='dense2', **defaults)(dense1)
#----------------Change--------------------------------------------------------
#    By changing the activation from the sigmoid to relu funtion we improve
#    the result by ~3%.  Test by commenting out above 2 lines and uncommenting
#    two lines below
#    dense1 = Dense(hidden_neurons, activation='relu', name='dense1', **defaults)(flatten)
#    dense2 = Dense(hidden_neurons, activation='relu', name='dense2', **defaults)(dense1)
#------------------------------------------------------------------------------ 
    output_probs = Dense(num_classes, activation='softmax',
                         name='predictions',**defaults)(dense2)
 
    model = Model(inputs=data, outputs=output_probs)
    loss = 'categorical_crossentropy'
    metrics=["accuracy"]
    optimizer = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
    
if __name__ == '__main__':
    #---------------Data Management--------------------------------------------
    # Load the data
    train = pd.read_csv("./Data/train.csv")
    test = pd.read_csv("./Data/test.csv")
    # labels
    Y_train = train["label"]
    # flattened images
    X_train = train.drop(labels = ["label"],axis = 1)
    # free some space
    del train
    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0
    # Reshape image in from (784,1) to (1,784)
    X_train = X_train.values.reshape(-1,1,784)
    test = test.values.reshape(-1,1,784)
    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = to_categorical(Y_train, num_classes = 10)
    
    # Set the random seed
    random_seed = 1234
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                      Y_train,
                                                      test_size = 0.1,
                                                      random_state=random_seed)
    #---------------Model Setup------------------------------------------------
    epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86
    input_shape = (1,784)
    num_classes = 10
    print('\n' + '~'*40)
    print('Creating and compiling model...')
    print('~'*40 + '\n')
    model = create_simple_model(input_shape=input_shape,
                                num_classes=num_classes)
    print(model.summary())
    #---------------Run Model--------------------------------------------------
    model.fit(X_train, Y_train, batch_size = batch_size,
              epochs = epochs, validation_data = (X_val, Y_val))
