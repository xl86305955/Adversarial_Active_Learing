import time
import numpy as np
from random import shuffle
import pickle

def mnist10():
    from tensorflow.keras.datasets import mnist
    (x_train, l_train), (x_test, l_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
   
    y_train = np.zeros((l_train.shape[0], l_train.max()+1), dtype=np.float32)
    y_train[np.arange(l_train.shape[0]), l_train] = 1
    y_test = np.zeros((l_test.shape[0], l_test.max()+1), dtype=np.float32)
    y_test[np.arange(l_test.shape[0]), l_test] = 1

    extra_train = 10000
    extra_test = 1000
    validation = 5000

    # Training and testing set for substitue model, we use full dataset for substitute model
    X_val = x_train[-validation:]
    Y_val = y_train[-validation:]
    Xsub_train = x_train[:-validation]
    Ysub_train = y_train[:-validation]
    Xsub_test = x_test
    Ysub_test = y_test
    
    # Traing and testing set for target model
    idx_train = np.random.randint(len(y_train)-validation, size=len(y_train)-validation)
    #idx_test = np.random.randint(len(y_test), size=m)
    
    X_train = x_train[idx_train]
    y_train = y_train[idx_train]
    X_test = x_test
    y_test = y_test
    #X_test = x_test[idx_test]
    #y_test = y_test[idx_test]
    
    print 'X_train ', X_train.shape 
    print 'X_test ', X_test.shape 
    print 'Xsub_train ', Xsub_train.shape 
    print 'Xsub_test ', Xsub_test.shape 
    print 'X_val ', X_val.shape 
    print 'Y_train ', y_train.shape 

    return [X_train, X_test, X_val, y_train, y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test]

def cifar10():
    mean = 120.707
    std = 64.15
    
    from tensorflow.keras.datasets import cifar10
    (x_train, l_train), (x_test, l_test) = cifar10.load_data()
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
    X_val = x_train[-validation:]
    Y_val = y_train[-validation:]
    Xsub_train = x_train[:-validation]
    Ysub_train = y_train[:-validation]
    Xsub_test = x_test
    Ysub_test = y_test
    
    # Traing and testing set for target model
    idx_train = np.random.randint(len(y_train)-validation, size=len(y_train)-validation)
    
    X_train = x_train[idx_train]
    y_train = y_train[idx_train]
    X_test = x_test
    y_test = y_test
    #X_test = x_test[idx_test]
    #y_test = y_test[idx_test]
    
    #print 'X_train ', X_train.shape 
    #print 'X_test ', X_test.shape 
    #print 'Xsub_train ', Xsub_train.shape 
    #print 'Xsub_test ', Xsub_test.shape 
    #print 'X_val ', X_val.shape 
    #print 'Y_train ', y_train.shape 

    return [X_train, X_test, X_val, y_train, y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test]

def shuffle_data(X, Y):
    n = X.shape[0]
    X_new = np.zeros(X.shape)
    Y_new = np.zeros(Y.shape)
    index = np.array([i for i in range(n)])
    np.random.shuffle(index)
    for i in range(n):
        j = index[i]
        X_new[i, :] = X[j, :]
        Y_new[i] = Y[j]
    return [X_new, Y_new]
