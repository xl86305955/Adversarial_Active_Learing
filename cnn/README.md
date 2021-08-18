# Adversarial Active Learning CNN Experiment

## Environments
1. Python 2.7
2. Tensorflow 2.1.0 with cuda 10.1
3. Cleverhans packages

For more details, plaese check the requirements.txt. 

## File Desrciptions
1. run_experiment.py: Main function.
2. prepare_data.py: For dataset preprocessing.
3. cnn_utils.py: Function with cnn related.

## Usage

### Arguments
* Datasets
    * MNIST10
    * Cifar10
* Tasks
    * nl: Normal CNN training process with query labels
    * al: Active Learning with sample selecting methods

```
make minist10 nl
```

or 


```
make minist10 al
```
