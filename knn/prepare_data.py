import time
import numpy as np
from random import shuffle
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from eps_separation import find_eps_separated_set
from nn_attack_white_box import generate_adversarial_examples
import pickle

def mnist_1v7_data(n=1000, m=500):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    ones = [x_train[i].reshape(-1) for i in range(len(y_train)) if y_train[i] == 1]
    sevens = [x_train[i].reshape(-1) for i in range(len(y_train)) if y_train[i] == 7]
    onesTest = [x_test[i].reshape(-1) for i in range(len(y_test)) if y_test[i] == 1]
    sevensTest = [x_test[i].reshape(-1) for i in range(len(y_test)) if y_test[i] == 7]

    n_ones = len(ones)
    n_sevens = len(sevens)
    shuffle(ones)
    shuffle(sevens)
    sub_train_size=500
    sub_test_size=250
    X_train = np.array(ones[:n] + sevens[:n])
    X_test = np.array(onesTest[:m] + sevensTest[:m])
    X_extra = np.array(ones[n:2*n] + sevens[n:2*n])
    X_valid = np.array(onesTest[2*n:2*n+m] + sevensTest[2*n:2*n+m])
    Xsub_train = np.array(ones[:sub_train_size] + sevens[:sub_train_size]) # Training set for substitute model
    Xsub_test = np.array(onesTest[:sub_test_size] + sevensTest[:sub_test_size]) # Testing` set for substitute model
    y_train = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    y_test = np.array([1 for i in range(m)] + [-1 for i in range(m)])
    Ysub_train = np.array([1 for i in range(sub_train_size)] + [-1 for i in range(sub_train_size)])
    Ysub_test = np.array([1 for i in range(sub_test_size)] + [-1 for i in range(sub_test_size)])
    y_extra = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    y_valid = np.array([1 for i in range(m)] + [-1 for i in range(m)])
    assert (len(X_extra) == len(y_extra))
    return [X_train, X_test, X_extra, y_train, y_test, y_extra, X_valid, y_valid,  Xsub_train, Xsub_test, Ysub_train, Ysub_test]

def mnist_avb_data(a=1, b=7, n=1000, m=500):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    ones = [mnist.train.images[i] for i in range(len(mnist.train.labels)) if mnist.train.labels[i] == a]
    sevens = [mnist.train.images[i] for i in range(len(mnist.train.labels)) if mnist.train.labels[i] == b]
    onesTest = [mnist.validation.images[i] for i in range(len(mnist.validation.labels)) if mnist.validation.labels[i] == a]
    sevensTest = [mnist.validation.images[i] for i in range(len(mnist.validation.labels)) if mnist.validation.labels[i] == b]
    
    
    n_ones = len(ones)
    n_sevens = len(sevens)
    shuffle(ones)
    shuffle(sevens)
    X_train = np.array(ones[:n] + sevens[:n])
    X_test = np.array(onesTest[:m] + sevensTest[:m])
    X_extra = np.array(ones[n:2*n] + sevens[n:2*n])
    y_train = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    y_test = np.array([1 for i in range(m)] + [-1 for i in range(m)])
    y_extra = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    assert (len(X_extra) == len(y_extra))
    return [X_train, X_test, X_extra, y_train, y_test, y_extra]

def mnist10(n=1000, m=500):
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
    X_val = x_train[:validation]
    Y_val = y_train[:validation]
    Xsub_train = x_train[validation:]
    Ysub_train = y_train[validation:]
    Xsub_test = x_test
    Ysub_test = y_test
    
    # Traing and testing set for target model
    idx_train = np.random.randint(len(y_train), size=n)
    #idx_test = np.random.randint(len(y_test), size=m)
    
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
    print 'Y_train ', y_train.shape 

    return [X_train, X_test, X_val, y_train, y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test]

def cifar10(n=1000, m=500):
    from tensorflow.keras.datasets import cifar10
    (x_train, l_train), (x_test, l_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
 
        
    
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
    #X_test = x_test[idx_test]
    #y_test = y_test[idx_test]
    
    #print 'X_train ', X_train.shape 
    #print 'X_test ', X_test.shape 
    #print 'Xsub_train ', Xsub_train.shape 
    #print 'Xsub_test ', Xsub_test.shape 
    #print 'X_val ', X_val.shape 
    #print 'Y_train ', y_train.shape 

    return [X_train, X_test, X_val, y_train, y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test]

from sklearn import datasets

def sample(n, sigma):
    [X, Y] = datasets.make_moons(n, True, sigma)
    Y = np.array([1 if i==1 else -1 for i in Y])
    return [X, Y]

def halfmoon_data(n=500, m=500, sigma=0.2):
    sub_train_size=500
    sub_test_size=500
    [X_train, y_train] = sample(n, sigma)
    [X_test, y_test] = sample(m, sigma)
    [X_extra, y_extra] = sample(n, sigma)
    [X_valid, y_valid] = sample(m, sigma)
    [Xsub_train, Ysub_train] = sample(sub_train_size, sigma)
    [Xsub_test, Ysub_test] = sample(sub_test_size, sigma)
    return [X_train, X_test, X_extra, y_train, y_test, y_extra, X_valid, y_valid, Xsub_train, Xsub_test, Ysub_train, Ysub_test]


def abalone_data(n=500,m=100):
    data = np.genfromtxt('abalone.data', dtype='str', delimiter=',')
    data = [data[i] for i in range(len(data)) if data[i][0] == 'F']
    X = [data[i][1:8] for i in range(len(data))]
    Y = [int(data[i][8]) for i in range(len(data))]
    X = [map(float, X[i]) for i in range(len(X))]
    s = sum([Y[i] >= 11 for i in range(len(Y))])
    # half of the abalones are 11 years old and above, so the classification task is whether age >= 11
    Y = [int(Y[i] >= 11) for i in range(len(Y))]
    Y = [1 if i==1 else -1 for i in Y]
    Z = zip(X, Y)
    shuffle(Z)
    sub_train_size=500
    sub_test_size=100
    test = Z[:m]
    train = Z[m:m+n]
    extra = Z[m+n:m+2*n]
    valid = Z[m+2*n:2*m+2*n]
    sub_train = Z[:sub_train_size] # Training set for substitute model 
    sub_test = Z[sub_train_size:sub_train_size+sub_test_size]  # Testing set for substitute model
    Xtrain = np.array([train[i][0] for i in range(len(train))])
    Ytrain = np.array([train[i][1] for i in range(len(train))])
    Xtest = np.array([test[i][0] for i in range(len(test))])
    Ytest = np.array([test[i][1] for i in range(len(test))])
    Xextra = np.array([extra[i][0] for i in range(len(extra))])
    Yextra = np.array([extra[i][1] for i in range(len(extra))])
    Xvalid = np.array([test[i][0] for i in range(len(valid))])
    Yvalid = np.array([test[i][1] for i in range(len(valid))])
    Xsub_train = np.array([sub_train[i][0] for i in range(len(sub_train))])
    Ysub_train = np.array([sub_train[i][1] for i in range(len(sub_train))])
    Xsub_test = np.array([sub_test[i][0] for i in range(len(sub_test))])
    Ysub_test = np.array([sub_test[i][1] for i in range(len(sub_test))])
    return [Xtrain, Xtest, Xextra, Ytrain, Ytest, Yextra, Xvalid, Yvalid, Xsub_train, Xsub_test, Ysub_train, Ysub_test]

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
