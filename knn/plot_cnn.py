import numpy as np
import matplotlib.pyplot as plt
import sys

data_folder = './cnn_results/'
task = sys.argv[1]  # Dataset
attack = sys.argv[2] # Attack method
size = sys.argv[3] # Attack method
flags = ['nn']

for flag in flags:
    fname = data_folder+task+'_'+flag+'_'+attack+'_'+str(size)+'.npy'
    [standard, at, adv] = np.load(fname)
    ep = [0.01*i for i in range(91)]
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    axes = plt.gca()
    ymin = 0.1
    ymax = 1
    axes.set_ylim([ymin,ymax])
    l1 = ax.plot(ep, standard, marker = 's', label = 'StandardNN')
    l2 = ax.plot(ep, at, marker = 'o', label = 'ATNN')
    l3 = ax.plot(ep, adv, marker = 'o', label = 'ADVNN')
    legend = ax.legend(loc = 'lower left', fontsize = 12)
    ax.set_ylabel('Classification Accuracy', fontsize = 18)
    ax.set_xlabel('Max $l_2$ Norm of Adv. Perturbation', fontsize = 18)
    ax.set_title(task, fontsize = 20)
    fig.tight_layout()
    plt.savefig(data_folder+task+'_'+attack+str(size)+'.png')
#    plt.show()
#    plt.close()
