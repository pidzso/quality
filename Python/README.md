# Federated-Learning (PyTorch)

Implementation of the vanilla federated learning paper: 
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://arxiv.org/abs/1602.05629

Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). 
In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the 
quality inference in federated learning paradigm, only simple models such as MLP and CNN are used.

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Options
You can change the default values of other parameters to simulate different conditions. 
The default values for various paramters parsed to the experiment are given in ```options.py```:
    
* ```--instance:```   Name the the run.
* ```--dataset:```    Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--iid:```        Distribution of data amongst users. Default is IID, set to 0 for non-IID.
* ```--num_users:```  Number of users. Default is 5.
* ```--frac:```       Fraction of users to be used for federated updates. Default is 0.4.
* ```--model:```      Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--epochs:```     Number of rounds of training.
* ```--optimizer:```   Default: 'sgd'. Options: 'sgd', 'Adam'
* ```--lr:```         Default: 0.01
* ```--drop_out:```   Default: 0.5
* ```--norm:```       Default: 'None'. Options: 'batch_norm', 'layer_norm', 'None'
* ```--weight:```     Weighting of users. Default is no weights, set to 1 for weighting.
* ```--noise_type:``` Default: 'no', Options: 'linear', 'some', 'attack', 'freeride'
* ```--noise_size:``` Default: 0. Options: 'linear' (max noise level) 'rest' (num.r of noisy par. with noise 0.5)
* ```--gpu:```        Default: None. Options: Specific GPU ID
* ```--seed:```       Default: 1
* ```--local_ep:```   Default: 1
* ```--local_bs:```   Default: 32
