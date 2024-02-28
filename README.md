# Federated-Learning (PyTorch)

Implementation of the vanilla federated learning paper: 
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://arxiv.org/abs/1602.05629

Experiments are produced on MNIST and CIFAR10 (IID). 
In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the 
quality inference in federated learning paradigm, only simple models such as MLP and CNN are used.

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Options
You can change the default values of other parameters to simulate different conditions. 
The default values for various paramters parsed to the experiment are given in ```options.py```:
    
* ```--seed:```       Default: 1
* ```--dataset:```    Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--num_users:```  Number of users. Default is 5.
* ```--frac:```       Fraction of users to be used for federated updates. Default is 0.4.
* ```--model:```      Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--epochs:```     Number of rounds of training.

* ```--weight:```     Weighting of users. Default is no weights, set to 1 for weighting.
* ```--noise_type:``` Default: 'no', Options: 'linear', 'some', 'attack', 'freeride'
* ```--noise_size:``` Default: 0. Options: 'linear' (max noise level) 'rest' (num. of noisy par. with noise 0.5)
* ```--aggregate:```  Default: ''. Options: 'avg', 'tm', 'KRUM', 'RR'
* ```--robustness:``` Default: 0. Trimming parameter of TrimmedMean (tm)
** ```--r_check:```   Default: 0. Round number when QI testing takes place in cross_silo setting.
* ```--anon_set:```   Default: 1. Anonymity set of clients during testing.
* ```--groupping:```  Default: ''. Technique to decide which coalitions should be tested.

* ```--optimizer:```  Default: 'sgd'. Options: 'sgd', 'Adam'
* ```--lr:```         Default: 0.01
* ```--do:```         Default: 0.5
* ```--norm:```       Default: 'None'. Options: 'batch_norm', 'layer_norm', 'None'
* ```--local_ep:```   Default: 1
* ```--local_bs:```   Default: 32
* ```--gpu:```        Default: None. Options: Specific GPU ID