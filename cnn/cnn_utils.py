import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ProjectedGradientDescent

from cleverhans.compat import softmax_cross_entropy_with_logits

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

def train(X_train, Y_train, X_val, Y_val, DATASET, num_epochs=50, batch_size=128, train_temp=1):
    tf.reset_default_graph()
    """
    Standard neural network training procedure.
    """
   
    model = Sequential()
    if DATASET == 'mnist10':
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
    elif DATASET == 'cifar10':
        weight_decay = 0.0005
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

    #model = Sequential()
    #
    #model.add(Conv2D(params[0], (3, 3),
    #                        input_shape=X_train.shape[1:]))
    #model.add(Activation('relu'))
    #model.add(Conv2D(params[1], (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(params[2], (3, 3)))
    #model.add(Activation('relu'))
    #model.add(Conv2D(params[3], (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Flatten())
    #model.add(Dense(params[4]))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(params[5]))
    #model.add(Activation('relu'))
    #model.add(Dense(10))
    

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss=fn,
                  optimizer=opt,
                  metrics=['accuracy'])

    checkpoint_filepath = './tmp/checkpoint'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              validation_data=(X_val, Y_val),
              nb_epoch=num_epochs,
              shuffle=True,
              callbacks=[model_checkpoint_callback])

    return model

def verify(X_test, Y_test, model):
    acc = 0.
    pred = model.predict(X_test)
    for i in range(len(Y_test)):
        pred_label = np.argmax(pred[i]) 
        true_label = np.argmax(Y_test[i]) 

        if pred_label == true_label:
            acc = acc + 1.

    return acc / len(Y_test)

def query(X_test, Y_train, DATASET, num_classes, ONE_HOT=True):
    tf.reset_default_graph()
    if DATASET == 'mnist10':
        restore = "models/mnist" 
        victim_model = MNISTModel(restore) 
    elif DATASET == 'cifar10':
        #restore = "models/cifar" 
        #victim_model = CIFARModel(restore) 
        restore = "models/cifar10vgg.h5" 
        victim_model = CIFAR_VGG16(restore) 

    pred = victim_model.predict(X_test)

    Y_test = np.zeros((X_test.shape[0],num_classes), dtype=np.float32)
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        Y_test[i][label] = 1.

    diff = 0
    for i in range(Y_test.shape[0]):
        if np.argmax(Y_train[i]) != np.argmax(Y_test[i]):
            diff = diff + 1
    print 'diff', diff

    return Y_test

def cnn_gen_adv(X_train, Y_train, ATTACK, DATASET, image_size, num_channels, nb_classes):
    tf.set_random_seed(1234)
    tf.reset_default_graph()
    sess = tf.Session()
    tf.keras.backend.set_session(sess)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))


    if DATASET == 'mnist10':
        model = MNISTModel("models/mnist") 
    elif DATASET == 'cifar10':
        #model = CIFARModel("models/cifar")
        model = CIFAR_VGG16("models/cifar10vgg.h5")

    model_wrap = KerasModelWrapper(model)
    if ATTACK == 'CW' or ATTACK == 'PGD':
        preds = model_wrap.get_logits(x)
    else:
        preds = model_wrap.get_probs(x)

    def evaluate():
        pass

    # Generate adversarial example with differnet attack method        
    if ATTACK == 'FGSM':
        print 'Attack with FGSM'
        fgsm_params = {'eps': 0.1,                                 
                       'ord': 2}                                          
        fgsm = FastGradientMethod(model_wrap, sess=sess)                    
        adv_x = fgsm.generate(x, **fgsm_params)                    
    elif ATTACK == 'DEEPFOOL':                                     
        print 'Attack with DeepFool'
        deepfool_params = {'nb_candidate': 10,                             
                           'overshoot': 0.2,                      
                           'max_iter': 100,                        
                           'clip_min': 0.,                         
                           'clip_max': 1.}                         
        deepfool = DeepFool(model_wrap, sess=sess)                      
        adv_x = deepfool.generate(x, **deepfool_params)            
    elif ATTACK == 'CW':                                           
        print 'Attack with CW'
        cw_l2_params = {'y_target': None,                          
                        'batch_size': 10,                           
                        'confidence': 0,                           
                        'learning_rate': 5e-3,                     
                        'binary_search_steps': 10,                 
                        'max_iterations': 1000,                    
                        'abort_early': True,                       
                        'initial_const': 1e-2,                     
                        'clip_min': 0.,                            
                        'clip_max': 1.}                            
        cw_l2 = CarliniWagnerL2(model_wrap, sess=sess)                  
        adv_x = cw_l2.generate(x, **cw_l2_params)                  
    elif ATTACK == 'PGD':
        print 'Attack with PGD'
        pgd_params =  {'eps':0.3,
                       'eps_iter':0.05,
                       'nb_iter':10,
                       'y':None,
                       'ord': 2,                                   
                       'loss_fn':softmax_cross_entropy_with_logits,
                       'clip_min':None,
                       'clip_max':None,
                       'y_target':None,
                       'rand_init':None,
                       'rand_init_eps':None,
                       'clip_grad':False,
                       'sanity_checks':True}
        pgd = ProjectedGradientDescent(model_wrap, sess=sess)                  
        adv_x = pgd.generate(x, **pgd_params)                         

    X_adv = sess.run(adv_x, feed_dict={x: X_train})                            
    #print ' X_adv shape' , X_adv.shape
    #print ' X_train shape' , X_train.shape

    confidence_score = model_wrap.get_probs(X_adv)
    confidence_score = sess.run(confidence_score, feed_dict={x:X_adv})
    #print 'Confidence score: ' , confidence_score

    #conf_avg=0
    #for i in range(len(confidence_score)):
    #    if confidence_score[i][0] >= confidence_score[i][1]:
    #        conf_avg = conf_avg + confidence_score[i][0]
    #    else:
    #        conf_avg = conf_avg + confidence_score[i][1]
    #conf_avg = conf_avg / len(confidence_score)
    #print 'Average confidence score: ' , conf_avg
    
    sess.close()
    del sess

    return X_adv, confidence_score

def valid_list(confidence_score):
    valid = np.arange(confidence_score.shape[0])
    #threshhold = 0
    #valid = []
    #for i in range(confidence_score.shape[0]):
    #    if(np.max(confidence_score[i]) >= threshhold):
    #        valid.append(i)
    print 'Num valid:', len(valid)
    #valid = np.array(valid)

    return valid

def prune_sample(X_adv, Xsub_train, Ysub_train, size, valid):
    idx = np.random.choice(valid, size, replace=False)
    rand_idx = np.random.choice(np.arange(Ysub_train.shape[0]), size, replace=False) 

    X_padv = X_adv[idx] 
    Y_padv = Ysub_train[idx]
    X_pcln = Xsub_train[idx]
    Y_pcln = Ysub_train[idx]
    X_rand = Xsub_train[rand_idx]
    Y_rand = Ysub_train[rand_idx]

    return X_padv, Y_padv, X_pcln, Y_pcln, X_rand, Y_rand

# Return the sorted argument list with min perturbation
def sort_perturb(x, x_adv):
    perturb = np.absolute(x_adv - x)
    perturb = np.reshape(perturb, (perturb.shape[0], -1))
    perturb = np.sum(perturb, axis=1)
    pert_arglist = np.argsort(perturb)
    return pert_arglist  

# Return the sorted argument list with max confidence score 
def sort_conf(confidence_score):
    print 'conf shape', confidence_score.shape
    max_score = np.zeros(shape=confidence_score.shape[0])
    for i in range(confidence_score.shape[0]):
        max_score[i] = np.max(confidence_score[i])

    conf_arglist = np.argsort(-1.0 * max_score)
    print 'conf_arglist shape', conf_arglist.shape

    return conf_arglist

# Analyze each time sample classes distribution  
def anal_class(label):
    anal_list = np.zeros(10, dtype=int)
    for i in range(label.shape[0]):
        idx = np.argmax(label[i])
        anal_list[idx] = anal_list[idx] + 1

    return anal_list

