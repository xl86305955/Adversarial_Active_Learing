from __future__ import division
import time
import sys
sys.path.insert(0, "./cleverhans")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
import numpy as np
from random import shuffle
from prepare_data import shuffle_data, mnist10, cifar10
from cnn_utils import *


def cbadv_cnn(DATASET, X_train, X_test, X_val, Y_train, Y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test, ep):
    fgsm_cnn_acc = []
    fgsm_aug_cnn_acc = []
    pgd_cnn_acc = []
    pgd_aug_cnn_acc = []
    deepfool_cnn_acc = []
    deepfool_aug_cnn_acc = []
    cw_cnn_acc = []
    cw_aug_cnn_acc = []
    
    if DATASET == 'mnist10':
        print 'Training mnist with adversarial examples'
        
        batch_size = 128
        image_size = 28
        num_channels = 1
        nb_classes = 10
        epochs = 100
        model_arch = [32, 32, 64, 64, 200, 200]

        # Setup training data
        X_train = X_train.reshape(X_train.shape[0], image_size, image_size, num_channels)
        X_test = X_test.reshape(X_test.shape[0], image_size, image_size, num_channels)
        X_val = X_val.reshape(X_val.shape[0], image_size, image_size, num_channels)
        Xsub_train = Xsub_train.reshape(Xsub_train.shape[0], image_size, image_size, num_channels)

    elif DATASET == 'cifar10':
        print 'Training cifar10 with adversarial examples'
        
        batch_size = 128
        image_size = 32
        num_channels = 3
        nb_classes = 10
        epochs = 200
        model_arch = [64, 64, 128, 128, 256, 256] 


    # Train with clean sample
    model = train(Xsub_train, Ysub_train, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
    cln_acc = verify(X_test, Y_test, model)
    np.full(len(ep), cln_acc)
    print 'cln acc', cln_acc
    
    # Generate FGSM adversarial example
    print 'Start creating FGSM adversarial example'
    idx = 0
    batch_adv = 1000
    rnd = Xsub_train.shape[0] / 1000
    X_adv_fgsm = np.empty((0, image_size, image_size, num_channels))
    confidence_score = np.empty((0, nb_classes))
    for i in range(int(rnd)):    # Prevent tensorflow out of memory
        x_adv, cf_score = cnn_gen_adv(Xsub_train[idx:idx+batch_adv], Ysub_train, 'FGSM', DATASET, image_size, num_channels, nb_classes)
        X_adv_fgsm = np.vstack((X_adv_fgsm, x_adv))
        confidence_score = np.vstack((confidence_score, cf_score))
        idx = idx + batch_adv
    valid = valid_list(confidence_score)   
    Y_adv_fgsm = Ysub_train

    x_fgsm_at = np.concatenate((X_adv_fgsm, Xsub_train)) 
    y_fgsm_at = np.concatenate((Y_adv_fgsm, Ysub_train))
    x_fgsm_at, y_fgsm_at = shuffle_data(x_fgsm_at, y_fgsm_at)

    # Adversarial Training with FGSM
    model = train(x_fgsm_at, y_fgsm_at, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
    fgsm_at_acc = verify(X_test, Y_test, model)
    print 'fgsm at acc', fgsm_at_acc
    fgsm_at_acc = np.full(len(ep), fgsm_at_acc)
    
    # Training with FGSM adversarial example
    for i in range(len(ep)):
        eps = ep[i]
        print 'eps', eps
        
        print 'Query for adversarial examples labels'
        perturb = X_adv_fgsm - Xsub_train
        x_padv =  Xsub_train + eps * perturb 
        y_padv = query(x_padv, Ysub_train, DATASET, nb_classes, ONE_HOT=True) 
        print 'Training with adversarial examples'
        adv_model = train(x_padv, y_padv, X_val, Y_val, DATASET, num_epochs=epochs)
        fgsm_acc = verify(X_test, Y_test, adv_model)
        fgsm_cnn_acc.append(fgsm_acc)
        print 'fgsm acc', fgsm_acc
        # Train with both augment adversarial example and clean sample
        x_aug = np.concatenate((x_padv, Xsub_train)) 
        y_aug = np.concatenate((y_padv, Ysub_train))
        #y_aug = np.concatenate((Ysub_train, Ysub_train))
        x_aug, y_aug = shuffle_data(x_aug, y_aug)
        model = train(x_aug, y_aug, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
        fgsm_aug_acc = verify(X_test, Y_test, model)
        fgsm_aug_cnn_acc.append(fgsm_aug_acc)
        print 'fgsm aug acc', fgsm_aug_acc
    
    # Free fgsm adversarial examples
    X_adv_fgsm = None
    Y_adv_fgsm = None
    x_fgsm_at = None
    y_fgsm_at = None

    # Generate PGD adversarial examples
    print 'Start creating PGD adversarial example'
    idx = 0
    batch_adv = 1000
    rnd = Xsub_train.shape[0] / 1000
    X_adv_pgd = np.empty((0, image_size, image_size, num_channels))
    confidence_score = np.empty((0, nb_classes))
    for i in range(int(rnd)):    # Prevent tensorflow out of memory
        x_adv, cf_score = cnn_gen_adv(Xsub_train[idx:idx+batch_adv], Ysub_train, 'PGD', DATASET, image_size, num_channels, nb_classes)
        X_adv_pgd = np.vstack((X_adv_pgd, x_adv))
        confidence_score = np.vstack((confidence_score, cf_score))
        idx = idx + batch_adv
    valid = valid_list(confidence_score)   
    Y_adv_pgd = Ysub_train

    x_pgd_at = np.concatenate((X_adv_pgd, Xsub_train)) 
    y_pgd_at = np.concatenate((Y_adv_pgd, Ysub_train))
    x_pgd_at, y_pgd_at = shuffle_data(x_pgd_at, y_pgd_at)
    
    # Adversarial Training with PGD
    model = train(x_pgd_at, y_pgd_at, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
    pgd_at_acc = verify(X_test, Y_test, model)
    print 'pgd at acc', pgd_at_acc
    pgd_at_acc = np.full(len(ep), pgd_at_acc)

    # Train with PGD adversarial example
    for i in range(len(ep)):
        eps = ep[i]
        print 'eps', eps
        
        print 'Query for adversarial examples labels'
        perturb = X_adv_pgd - Xsub_train
        x_padv =  Xsub_train + eps * perturb 
        y_padv = query(x_padv, Ysub_train, DATASET, nb_classes, ONE_HOT=True) 
        print 'Training with adversarial examples'
        adv_model = train(x_padv, y_padv, X_val, Y_val, DATASET, num_epochs=epochs)
        pgd_acc = verify(X_test, Y_test, adv_model)
        pgd_cnn_acc.append(pgd_acc)
        print 'pgd acc', pgd_acc
        # Train with both augment adversarial example and clean sample
        x_aug = np.concatenate((x_padv, Xsub_train)) 
        y_aug = np.concatenate((y_padv, Ysub_train))
        #y_aug = np.concatenate((Ysub_train, Ysub_train))
        x_aug, y_aug = shuffle_data(x_aug, y_aug)
        model = train(x_aug, y_aug, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
        pgd_aug_acc = verify(X_test, Y_test, model)
        pgd_aug_cnn_acc.append(pgd_aug_acc)
        print 'pgd aug acc', pgd_aug_acc

    # Free pgd adversarial examples
    X_adv_pgd = None
    Y_adv_pgd = None
    x_pgd_at = None
    y_pgd_at = None


    # Generate DeepFool adversarial examples
    print 'Start creating DeepFool adversarial example'
    idx = 0
    batch_adv = 1000
    rnd = Xsub_train.shape[0] / 1000
    X_adv_deepfool = np.empty((0, image_size, image_size, num_channels))
    confidence_score = np.empty((0, nb_classes))
    for i in range(int(rnd)):    # Prevent tensorflow out of memory
        x_adv, cf_score = cnn_gen_adv(Xsub_train[idx:idx+batch_adv], Ysub_train, 'DEEPFOOL', DATASET, image_size, num_channels, nb_classes)
        X_adv_deepfool = np.vstack((X_adv_deepfool, x_adv))
        confidence_score = np.vstack((confidence_score, cf_score))
        idx = idx + batch_adv
    valid = valid_list(confidence_score)   
    Y_adv_deepfool = Ysub_train
   
    x_deepfool_at = np.concatenate((X_adv_deepfool, Xsub_train)) 
    y_deepfool_at = np.concatenate((Y_adv_deepfool, Ysub_train))
    x_deepfool_at, y_deepfool_at = shuffle_data(x_deepfool_at, y_deepfool_at)
    
    # Adversarial Training with DeepFool
    model = train(x_deepfool_at, y_deepfool_at, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
    deepfool_at_acc = verify(X_test, Y_test, model)
    print 'deepfool at acc', deepfool_at_acc
    deepfool_at_acc = np.full(len(ep), deepfool_at_acc)

    # Train with adversarial example
    for i in range(len(ep)):
        eps = ep[i]
        print 'eps', eps
   
        print 'Query for adversarial examples labels'
        perturb = X_adv_deepfool - Xsub_train
        x_padv =  Xsub_train + eps * perturb 
        y_padv = query(x_padv, Ysub_train, DATASET, nb_classes, ONE_HOT=True) 
        print 'Training with adversarial examples'
        adv_model = train(x_padv, y_padv, X_val, Y_val, DATASET, num_epochs=epochs)
        deepfool_acc = verify(X_test, Y_test, adv_model)
        deepfool_cnn_acc.append(deepfool_acc)
        print 'deepfool acc', deepfool_acc
        # Train with both augment adversarial example and clean sample
        x_aug = np.concatenate((x_padv, Xsub_train)) 
        y_aug = np.concatenate((y_padv, Ysub_train))
        #y_aug = np.concatenate((Ysub_train, Ysub_train))
        x_aug, y_aug = shuffle_data(x_aug, y_aug)
        model = train(x_aug, y_aug, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
        deepfool_aug_acc = verify(X_test, Y_test, model)
        deepfool_aug_cnn_acc.append(deepfool_aug_acc)
        print 'deepfool aug acc', deepfool_aug_acc
    
    # Free DeepFool adversarial examples
    X_adv_deepfool = None
    Y_adv_deepfool = None
    x_deepfool_at = None
    y_deepfool_at = None

    # Generate CW adversarial examples
    print 'Start creating CW adversarial example'
    idx = 0
    batch_adv = 1000
    rnd = Xsub_train.shape[0] / 1000
    X_adv_cw = np.empty((0, image_size, image_size, num_channels))
    confidence_score = np.empty((0, nb_classes))
    for i in range(int(rnd)):    # Prevent tensorflow out of memory
        x_adv, cf_score = cnn_gen_adv(Xsub_train[idx:idx+batch_adv], Ysub_train, 'CW', DATASET, image_size, num_channels, nb_classes)
        X_adv_cw = np.vstack((X_adv_cw, x_adv))
        confidence_score = np.vstack((confidence_score, cf_score))
        idx = idx + batch_adv
    valid = valid_list(confidence_score)   
    Y_adv_cw = Ysub_train
    
    x_cw_at = np.concatenate((X_adv_cw, Xsub_train)) 
    y_cw_at = np.concatenate((Y_adv_cw, Ysub_train))
    x_cw_at, y_cw_at = shuffle_data(x_cw_at, y_cw_at)
    
    # Adversarial Training with CW
    model = train(x_cw_at, y_cw_at, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
    cw_at_acc = verify(X_test, Y_test, model)
    print 'cw at acc', cw_at_acc
    cw_at_acc = np.full(len(ep), cw_at_acc)

    # Train with adversarial example
    for i in range(len(ep)):
        eps = ep[i]
        print 'eps', eps

        print 'Query for adversarial examples labels'
        perturb = X_adv_cw - Xsub_train
        x_padv =  Xsub_train + eps * perturb 
        y_padv = query(x_padv, Ysub_train, DATASET, nb_classes, ONE_HOT=True) 
        print 'Training with adversarial examples'
        adv_model = train(x_padv, y_padv, X_val, Y_val, DATASET, num_epochs=epochs)
        cw_acc = verify(X_test, Y_test, adv_model)
        cw_cnn_acc.append(cw_acc)
        print 'cw acc', cw_acc
        # Train with both augment adversarial example and clean sample
        x_aug = np.concatenate((x_padv, Xsub_train)) 
        #y_aug = np.concatenate((y_padv, Ysub_train))
        y_aug = np.concatenate((Ysub_train, Ysub_train))
        x_aug, y_aug = shuffle_data(x_aug, y_aug)
        model = train(x_aug, y_aug, X_val, Y_val, DATASET, num_epochs=epochs, batch_size=batch_size)
        cw_aug_acc = verify(X_test, Y_test, model)
        cw_aug_cnn_acc.append(cw_aug_acc)
        print 'cw aug acc', cw_aug_acc

    acc = [cln_acc, fgsm_at_acc, pgd_at_acc, deepfool_at_acc, cw_at_acc, fgsm_cnn_acc, fgsm_aug_cnn_acc, pgd_cnn_acc, pgd_aug_cnn_acc,deepfool_cnn_acc, deepfool_aug_cnn_acc, cw_cnn_acc, cw_aug_cnn_acc]
    return acc

def iter_al_cnn(DATASET, X_train, X_test, X_val, Y_train, Y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test, ep):
    eps_list = [0.1, 0.01, 0.001]

    acc_all = []
    random_acc = []
    minperturb_acc = []
    minperturb_e3_acc = []
    minperturb_e2_acc = []
    minperturb_e1_acc = []
    maxconfidence_acc = []

    if DATASET == 'mnist10':
        print 'Training mnist with adversarial examples'
        
        batch_size = 128
        image_size = 28
        num_channels = 1
        nb_classes = 10
        restore = "models/mnist" 
        model_arch = [32, 32, 64, 64, 200, 200]
        num_epoch = 50

        budgets = [100, 400, 300, 200]
        unlabeled_poolsize = [55000, 30000, 10000, 5000, 2500, 1500]

        # Setup training data
        X_train = X_train.reshape(X_train.shape[0], image_size, image_size, num_channels)
        X_test = X_test.reshape(X_test.shape[0], image_size, image_size, num_channels)
        X_val = X_val.reshape(X_val.shape[0], image_size, image_size, num_channels)
        Xsub_train = Xsub_train.reshape(Xsub_train.shape[0], image_size, image_size, num_channels)

    elif DATASET == 'cifar10':
        print 'Training cifar10 with adversarial examples'
        
        batch_size = 128
        image_size = 32
        num_channels = 3
        nb_classes = 10
        restore = "models/cifar10vgg.h5" 
        model_arch = [64, 64, 128, 128, 256, 256] 
        num_epoch = 100
        
        #unlabeled_poolsize = [45000, 40000, 35000]
        unlabeled_poolsize = [40000]
        budgets = [10000, 10000, 10000]

    # Generate adversarial example
    print 'Start creating adversarial example'
    idx = 0
    batch_adv = 1000
    rnd = Xsub_train.shape[0] / 1000
    X_adv = np.empty((0, image_size, image_size, num_channels))
    confidence_score = np.empty((0, nb_classes))
    for i in range(int(rnd)):    # Prevent tensorflow out of memory
        x_adv, cf_score = cnn_gen_adv(Xsub_train[idx:idx+batch_adv], Ysub_train, 'DEEPFOOL', DATASET, image_size, num_channels, nb_classes)
        X_adv = np.vstack((X_adv, x_adv))
        confidence_score = np.vstack((confidence_score, cf_score))
        idx = idx + batch_adv
    
    # Start Active learning training process
    for poolsize in unlabeled_poolsize:
        num_samples = 0
       
        # Random Select Unlabeled Pool
        tmp_sub_idx = np.arange(Xsub_train.shape[0])
        np.random.shuffle(tmp_sub_idx)
        tmp_sub_idx = tmp_sub_idx[:poolsize]
        
        xsub_train = Xsub_train[tmp_sub_idx]
        ysub_train = Ysub_train[tmp_sub_idx]
        
        xsub_adv = X_adv[tmp_sub_idx]
        ysub_adv = Ysub_train[tmp_sub_idx]

        # Random Sample list
        tmp_rand_idx = np.arange(poolsize)
        np.random.shuffle(tmp_rand_idx)

        # Argument list sorted by min perturbation
        pert_arglist = sort_perturb(xsub_train, xsub_adv)

        # Argument list sorted by max confidence score
        conf_arglist = sort_conf(confidence_score)

        # Initialized training samples and labels 
        X_rand = np.empty((0, image_size, image_size, num_channels))
        Y_rand = np.empty((0, nb_classes))

        X_minpert = np.empty((0, image_size, image_size, num_channels))
        Y_minpert = np.empty((0, nb_classes))
        
        X_minpert_e3 = np.empty((0, image_size, image_size, num_channels))
        X_minpert_e2 = np.empty((0, image_size, image_size, num_channels))
        X_minpert_e1 = np.empty((0, image_size, image_size, num_channels))
        
        X_maxconf = np.empty((0, image_size, image_size, num_channels))
        Y_maxconf = np.empty((0, nb_classes))
        
        for budget in budgets:
            # Random methods
            rand_idx = tmp_rand_idx[num_samples:num_samples+budget]
            x_rand = xsub_train[rand_idx] 
            X_rand = np.vstack((X_rand, x_rand))
            
            y_rand = ysub_train[rand_idx]
            y_qrand = query(x_rand, y_rand, DATASET, nb_classes, ONE_HOT=True) 
            Y_rand = np.vstack((Y_rand, y_qrand)) 

            rand_model = train(X_rand, Y_rand, X_val, Y_val, DATASET, num_epoch, batch_size=batch_size)
            rand_acc = verify(X_test, Y_test, rand_model)
            random_acc.append(rand_acc)
            print '#Samples', num_samples+budget, ',Random Acc:', rand_acc

            # Using Min perturbation of adversarial examples to select samples for training 
            minpert_idx = pert_arglist[num_samples:num_samples+budget]
            x_cln_minpert = xsub_train[minpert_idx] 
            x_adv_minpert = xsub_adv[minpert_idx]

            X_minpert = np.vstack((X_minpert, x_cln_minpert))
            X_minpert = np.vstack((X_minpert, x_adv_minpert))
            
            y_minpert = ysub_train[minpert_idx]
            y_qminpert = query(x_cln_minpert, y_minpert, DATASET, nb_classes, ONE_HOT=True) 
            Y_minpert = np.vstack((Y_minpert, y_qminpert)) 
            Y_minpert = np.vstack((Y_minpert, y_qminpert)) 

            minpert_model = train(X_minpert, Y_minpert, X_val, Y_val, DATASET, num_epoch, batch_size=batch_size)
            minpert_acc = verify(X_test, Y_test, minpert_model)
            minperturb_acc.append(minpert_acc)    
            print '#Samples', num_samples+budget, ',Minpert Acc:', minpert_acc

            # Using Min perturbation of adversarial examples with epsilon to select samples for training 
            perturb = x_adv_minpert - x_cln_minpert 
            x_adv_e3 = x_cln_minpert + eps_list[2] * perturb
            x_adv_e2 = x_cln_minpert + eps_list[1] * perturb
            x_adv_e1 = x_cln_minpert + eps_list[0] * perturb
            
            X_minpert_e3 = np.vstack((X_minpert_e3, x_cln_minpert))
            X_minpert_e3 = np.vstack((X_minpert_e3, x_adv_e3))
            X_minpert_e2 = np.vstack((X_minpert_e2, x_cln_minpert))
            X_minpert_e2 = np.vstack((X_minpert_e2, x_adv_e2))
            X_minpert_e1 = np.vstack((X_minpert_e1, x_cln_minpert))
            X_minpert_e1 = np.vstack((X_minpert_e1, x_adv_e1))
            
            minpert_e3_model = train(X_minpert_e3, Y_minpert, X_val, Y_val, DATASET, num_epoch, batch_size=batch_size)
            minpert_e3_acc = verify(X_test, Y_test, minpert_e3_model)
            minperturb_e3_acc.append(minpert_e3_acc)    
            print '#Samples', num_samples+budget, ',Minpert e3 Acc:', minpert_e3_acc

            minpert_e2_model = train(X_minpert_e2, Y_minpert, X_val, Y_val, DATASET, num_epoch, batch_size=batch_size)
            minpert_e2_acc = verify(X_test, Y_test, minpert_e2_model)
            minperturb_e2_acc.append(minpert_e2_acc)    
            print '#Samples', num_samples+budget, ',Minpert e2 Acc:', minpert_e2_acc

            minpert_e1_model = train(X_minpert_e1, Y_minpert, X_val, Y_val, DATASET, num_epoch, batch_size=batch_size)
            minpert_e1_acc = verify(X_test, Y_test, minpert_e1_model)
            minperturb_e1_acc.append(minpert_e1_acc)    
            print '#Samples', num_samples+budget, ',Minpert e1 Acc:', minpert_e1_acc

            # Using Max perturbation of adversarial examples to select samples for training 
            maxconf_idx = pert_arglist[num_samples:num_samples+budget]
            x_cln_maxconf = xsub_train[maxconf_idx] 
            x_adv_maxconf = xsub_adv[maxconf_idx]

            X_maxconf = np.vstack((X_maxconf, x_cln_maxconf))
            X_maxconf = np.vstack((X_maxconf, x_adv_maxconf))
            
            y_maxconf = ysub_train[maxconf_idx]
            y_qmaxconf = query(x_cln_maxconf, y_maxconf, DATASET, nb_classes, ONE_HOT=True) 
            Y_maxconf = np.vstack((Y_maxconf, y_qmaxconf)) 
            Y_maxconf = np.vstack((Y_maxconf, y_qmaxconf)) 

            maxconf_model = train(X_maxconf, Y_maxconf, X_val, Y_val, DATASET, num_epoch, batch_size=batch_size)
            maxconf_acc = verify(X_test, Y_test, maxconf_model)
            maxconfidence_acc.append(maxconf_acc)    
            print '#Samples', num_samples+budget, ',Maxconf Acc:', maxconf_acc

            num_samples = num_samples + budget

    acc_all.append(random_acc)
    acc_all.append(minperturb_acc)
    acc_all.append(minperturb_e3_acc)
    acc_all.append(minperturb_e2_acc)
    acc_all.append(minperturb_e1_acc)
    acc_all.append(maxconfidence_acc)

    return acc_all


if __name__ == "__main__":
    ### main script for the experiment.
    t = time.time()
    DATASET = sys.argv[1]     # which dataset
    TASK = sys.argv[2] # which experiment
    '''
        Valid datasets are:
        * mnist10
        * cifar10
        Valid tasks are:
        * nl # normal training
        * al # active learning
    '''

    t = time.time()
    if DATASET == 'mnist10':
        [X_train, X_test, X_val, Y_train, Y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test] = mnist10()
        ep = [0.01*(i) for i in range(41)]
    elif DATASET == 'cifar10':
        [X_train, X_test, X_val, Y_train, Y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test] = cifar10()
        ep = [0.001, 0.01, 0.1]

    if TASK == 'nl':
        # Run all experiments together included cln at aug
        acc = cbadv_cnn(DATASET, X_train, X_test, X_val, Y_train, Y_test, Y_val, Xsub_train, Xsub_test, Ysub_train, Ysub_test, ep)
        data_folder = "./cnn_results/"
        f_name = data_folder+'acc'
        np.save(f_name, acc)
    else:
        # Active Learning experiments
        print 'Start Active Learning Experiments'
        acc_all = iter_al_cnn(DATASET, X_train, X_test, X_val, Y_train, Y_test, Y_val,
                          Xsub_train, Xsub_test, Ysub_train, Ysub_test, ep)
        
        print 'Random', acc_all[0]
        print 'Min pert', acc_all[1]
        print 'Min pert e3', acc_all[2]
        print 'Min pert e2', acc_all[3]
        print 'Min pert e1', acc_all[4]
        print 'Max conf', acc_all[5]
