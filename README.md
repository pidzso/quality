Modified implementation of the vanilla federated learning: https://github.com/AshwinRJ/Federated-Learning-PyTorch

Since the purpose of these experiments are to illustrate the effectiveness of the quality inference, only simple models such as MLP and CNN are used on MNIST, and CIFAR10.

The added options are:

--noise_type	- freeride / attack / linear
--noise_size	- noise parameter (either the number of cheaters or the max noise for linear)
--weight		- weight participants