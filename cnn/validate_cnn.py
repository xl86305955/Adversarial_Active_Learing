import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten

np.set_printoptions(threshold=np.inf)

def MNISTModel(restore):
    model = Sequential()

    layers = [
        Conv2D(32, (3, 3),
               input_shape=(28, 28, 1)),
        Activation('relu'),
        Conv2D(32, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation('relu'),
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(200),
        Activation('relu'),
        Dense(200),
        Activation('relu'),
        Dense(10),
    ]
    
    for layer in layers:
        model.add(layer)

    model.add(Activation("softmax"))
    
    model.load_weights(restore)
    return model

def CIFARModel(restore):
    model = Sequential()

    layers = [
        Conv2D(64, (3, 3),
                      input_shape=(32, 32, 3)),
        Activation('relu'),
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3)),
        Activation('relu'),
        Conv2D(128, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(256),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(10),
    ]

    for layer in layers:
        model.add(layer)
    model.add(Activation("softmax"))
    
    model.load_weights(restore)

    return model

def CIFAR_VGG16(restore):
    weight_decay = 0.0005

    model = Sequential()
    layers = [    
        Conv2D(64, (3, 3), padding='same',
               input_shape=(32,32,3), kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),


        Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),


        Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(512,kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),

        Dropout(0.5),
        Dense(10),
    ]
    
    for layer in layers:
        model.add(layer)
    model.add(Activation("softmax"))
    
    model.load_weights(restore)

    return model

def verify(X_test, Y_test, model, BATCH_SIZE=8, IMAGE_SIZE=28, NUM_CHANNELS=1):
    acc = 0.
    pred = model.predict(X_test)
    for i in range(len(Y_test)):
        pred_label = np.argmax(pred[i]) 
        true_label = np.argmax(Y_test[i]) 

        if pred_label == true_label:
            acc = acc + 1.

    print X_test.shape
    return acc / len(Y_test)

def cifar10(n=1000, m=500):
    from tensorflow.keras.datasets import cifar10
    (x_train, l_train), (x_test, l_test) = cifar10.load_data()
    mean = 120.707
    std = 64.15
    #x_train = x_train / 255.0
    #x_test = x_test / 255.0
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    
    y_train = np.zeros((l_train.shape[0], l_train.max()+1), dtype=np.float32)
    for i in range(l_train.shape[0]):
        y_train[i][l_train[i]] = 1
    y_test = np.zeros((l_test.shape[0], l_test.max()+1), dtype=np.float32)
    for i in range(l_test.shape[0]):
        y_test[i][l_test[i]] = 1

    extra_train = 10000
    extra_test = 1000
    validation = 5000

    # Training and testing set for substitue model, we use full dataset for substitute model
    X_val = x_train[:validation]
    Y_val = y_train[:validation]
    Xsub_train = x_train[validation:]
    Ysub_train = y_train[validation:]
    Xsub_test = x_test
    Ysub_test = y_test
    
    # Traing and testing set for target model
    idx_train = np.random.randint(len(y_train), size=n)
    idx_test = np.random.randint(len(y_test), size=m)
    
    X_train = x_train[idx_train]
    y_train = y_train[idx_train]
    X_test = x_test
    y_test = y_test

    return [X_train, X_test, X_val, y_train, y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test]


X_train, X_test, X_val, y_train, y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test =  cifar10(1000, 500)
model = CIFAR_VGG16('models/cifar10vgg.h5')
#model = CIFARModel('models/cifar')
acc = verify(X_test, y_test, model, BATCH_SIZE=128, IMAGE_SIZE=32, NUM_CHANNELS=3)

print 'acc', acc
