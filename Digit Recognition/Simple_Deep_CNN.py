# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:59:25 2017

@author: Ethan
"""

from keras import regularizers
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Model
from keras.layers.core import Dense, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_CNN1_model(input_shape,num_classes): # 0.99385 score
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
    # ---------- First set of convolution and pooling ----------
    conv1_5x5 = Conv2D(filters=32, kernel_size=5, strides=1, activation='relu',
                       padding='same', name='conv1_5x5', **defaults)(data)
    pool1 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same', name='pool1')(conv1_5x5)
    # ---------- Second set of convolution and pooling ----------
    conv2_5x5 = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu',
                       padding='same', name='conv2_5x5', **defaults)(pool1)
    pool2 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same', name='pool2')(conv2_5x5)
    # ---------- Dropout and dense (a.k.a fully connected) layers ----------
    flatten = Flatten(name='flatten')(pool2)
    dense1 = Dense(units=2**9, activation="relu",
                   name='dense1', **defaults)(flatten)
    drop1 = Dropout(rate=0.5, name='drop1')(dense1)
    
    dense2 = Dense(units=2**9, activation="relu",
                   name='dense2', **defaults)(drop1)
    drop2 = Dropout(rate=0.5, name='drop2')(dense2)
    # ---------- Prediction layer ----------
    output_probs = Dense(num_classes, activation='softmax',
                         name='predictions',**defaults)(drop2)
    # create the model and train using SGD with momentum
    model = Model(inputs=data, outputs=output_probs)
    loss = 'categorical_crossentropy'
    metrics=["accuracy"]
#    optimizer = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def create_CNN2_model(input_shape,num_classes): # 0.99400 score
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
    # ---------- First set of convolution and pooling ----------
    conv1_5x5 = Conv2D(filters=32, kernel_size=5, strides=1, activation='relu',
                       padding='same', name='conv1_5x5', **defaults)(data)
    pool1 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same', name='pool1')(conv1_5x5)
    # ---------- Second set of convolution and pooling ----------
    conv1_3x3 = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu',
                       padding='same', name='conv1_3x3', **defaults)(pool1)
    pool2 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same', name='pool2')(conv1_3x3)
    # ---------- Dropout and dense (a.k.a fully connected) layers ----------
    flatten = Flatten(name='flatten')(pool2)
    dense1 = Dense(units=2**10, activation="relu",
                   name='dense1', **defaults)(flatten)
    drop1 = Dropout(rate=0.5, name='drop1')(dense1)
    
    dense2 = Dense(units=2**9, activation="relu",
                   name='dense2', **defaults)(drop1)
    drop2 = Dropout(rate=0.5, name='drop2')(dense2)
    # ---------- Prediction layer ----------
    output_probs = Dense(num_classes, activation='softmax',
                         name='predictions',**defaults)(drop2)
    # create the model and train using SGD with momentum
    model = Model(inputs=data, outputs=output_probs)
    loss = 'categorical_crossentropy'
    metrics=["accuracy"]
#    optimizer = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def create_CNN3_model(input_shape,num_classes): # 0.99514 score
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
    # ---------- First set of convolution and pooling ----------
    conv1_5x5 = Conv2D(filters=32, kernel_size=5, strides=1, activation='relu',
                       padding='same', name='conv1_5x5', **defaults)(data)
    pool1 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same', name='pool1')(conv1_5x5)
    norm1 = BatchNormalization(axis=3, name='norm1')(pool1)
    # ---------- Second set of convolution and pooling ----------
    conv1_3x3 = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu',
                       padding='same', name='conv1_3x3', **defaults)(norm1)
    pool2 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same', name='pool2')(conv1_3x3)
    norm2 = BatchNormalization(axis=3,name='norm2')(pool2)
    # ---------- Dropout and dense (a.k.a fully connected) layers ----------
    flatten = Flatten(name='flatten')(norm2)
    dense1 = Dense(units=2**10, activation="relu",
                   name='dense1', **defaults)(flatten)
    drop1 = Dropout(rate=0.5, name='drop1')(dense1)
    
    dense2 = Dense(units=2**9, activation="relu",
                   name='dense2', **defaults)(drop1)
    drop2 = Dropout(rate=0.5, name='drop2')(dense2)
    # ---------- Prediction layer ----------
    output_probs = Dense(num_classes, activation='softmax',
                         name='predictions',**defaults)(drop2)
    # create the model and train using SGD with momentum
    model = Model(inputs=data, outputs=output_probs)
    loss = 'categorical_crossentropy'
    metrics=["accuracy"]
#    optimizer = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
#    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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
    X_train = X_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)
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
    epochs = 50
    batch_size = 86
    input_shape = (28,28,1)
    num_classes = 10
    print('\n' + '~'*40)
    print('Creating and compiling model...')
    print('~'*40 + '\n')
    model = create_CNN3_model(input_shape=input_shape, num_classes=num_classes)
    print(model.summary())
    # With data augmentation to prevent overfitting.  Data augmentations allows
    # you to artificially inflat your data set
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
    train_generator = datagen.flow(X_train, Y_train,
                                   batch_size=batch_size, shuffle=True)
    # Set up rules for LR reduction
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', # which value to monitor
                                            patience=3, # how many epochs to wait
                                            verbose=1, # logging info, set to 0 for faster running
                                            factor=0.5, # what to do to the lr if things haven't changed
                                            min_lr=0.00001)
#    checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
#                                 monitor='val_acc',
#                                 save_best_only=True,
#                                 save_weights_only=True,
#                                 mode='max')
    #This will save off the weights of the best training iteration
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 monitor='val_acc',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')
    #---------------Run Model--------------------------------------------------
    history = model.fit_generator(train_generator,
                  epochs = epochs, validation_data = (X_val,Y_val),
                  verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size,
                  callbacks=[learning_rate_reduction, checkpoint])

    #---------------Test Model-------------------------------------------------
    model.load_weights('weights.hdf5') #load the best iteration of training
    results = model.predict(test)
    results = np.argmax(results,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv("cnn_mnist_predictions_6_60ep.csv",index=False)
    