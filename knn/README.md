# k-NN Experiments
This repo is modified from "Analyzing the Robustness of Nearest Neighbors to Adversarial Examples" accepted by ICML 2018. The paper can be found on arXiv at https://arxiv.org/abs/1706.03922

## Required Environment
   1. standard numpy and matlibplot packages
   2. tensorflow with gpu
   3. hopcroftkarp module of finding maximum matching.  (can be added using pip install hopcroftkarp)
   4. cleverhans adversarial attack package. (can be found at https://github.com/tensorflow/cleverhans)

### Usage 
#### Arguments
* Datasets
    * halfmoon
    * abalone
    * mnist (mnist1v7)
* Substitute Models
    * nn (neural network)
* Adversarial Attack
    * FGSM
    * PGD
    * DeepFool
    * C&W
* Training Size    

```
make halfmoon
```

For more experiment, please check the Makefile
