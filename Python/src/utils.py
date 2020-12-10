import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:  # Sample IID user data from CIFAR
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:  # Sample non-IID user data from CIFAR
            user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            # transforms.Lambda(lambda x: AddGaussianNoise(x)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        if args.iid:  # sample training data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:  # Sample Non-IID user data from Mnist
            user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    # Returns the average of the weights
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Instance  : {args.instance}')
    print(f'    Dataset   : {args.dataset}')
    print(f'    IID       : {args.iid}')
    print(f'    Model     : {args.model}')
    print(f'    Rounds    : {args.epochs}\n')
    print(f'    Weights   : {args.weight}')

    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')

    print(f'    Noise Type         : {args.noise_type}')
    if args.noise_type == 'linear':
        print(f'    Max Noise Level    : {args.noise_size}')
    else:
        print(f'    Noisy Participants : {args.noise_size}')

    return
