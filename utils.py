import copy
import torch
from scipy import stats
from torchvision import datasets, transforms
from sampling import mnist_iid, cifar_iid


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
        user_groups = cifar_iid(train_dataset, args.num_users)

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

        # sample training data from Mnist
        user_groups = mnist_iid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    # Returns the average of the weights
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def median_weights(w, trim, device):
    # Returns the median of the weights
    w_tm = copy.deepcopy(w[0])
    for key in w_tm.keys():
        if 'bias' in key:  # 1 dimension
            for i in range(0, len(w_tm[key])):
                w_tm[key][i] = torch.tensor([stats.trim_mean([w[client][key][i].to("cpu") for client in range(0, len(w))], trim)], device=device)
        elif 'conv' in key:  # 4 dimension
            for i in range(0, len(w_tm[key])):
                for j in range(0, len(w_tm[key][i])):
                    for k in range(0, len(w_tm[key][i][j])):
                        for l in range(0, len(w_tm[key][i][j][k])):
                           w_tm[key][i][j][k][l] = torch.tensor([stats.trim_mean([w[client][key][i][j][k][l].to("cpu") for client in range(0, len(w))], trim)], device=device)
        else:  # 2 dimension
            for i in range(0, len(w_tm[key])):
                for j in range(0, len(w_tm[key][i])):
                    w_tm[key][i][j] = torch.tensor([stats.trim_mean([w[client][key][i][j].to("cpu") for client in range(0, len(w))], trim)], device=device)
    return w_tm


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Rounds    : {args.epochs}')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Seed  : {args.seed}\n')

    print(f'    Weights   : {args.weight}')
    print(f'    Aggregation method : {args.aggregate}')
    print(f'    Robustness level   : {args.robustness}\n')

    print(f'    Noise Type  : {args.noise_type}')
    print(f'    Noise Size : {args.noise_size}')

    return
