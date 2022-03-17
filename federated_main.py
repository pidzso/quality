import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNCifar
from utils import get_dataset, average_weights, median_weights, exp_details


if __name__ == '__main__':
    args = args_parser()
    exp_details(args)

    # define paths
    path_project = os.path.abspath('..')
    path = args.model + '_' + args.dataset + '_' + str(args.num_users) + '/' + \
           args.noise_type + '_' + str(args.noise_size) + '_' + str(args.weight) + '_' + str(args.aggregate) + '_' + str(args.robustness) + \
           '/' + str(args.seed) + '/'
    Path(path_project + '/save/' + path).mkdir(parents=True, exist_ok=True)

    if args.gpu:
        # Nvidia card ID = 0
        torch.cuda.set_device(0)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # inject noise
    noise = torch.zeros([args.num_users])
    if args.noise_type == 'linear':
        # 1st participant: noise_size noise, last participant: 0 noise, in-between linearly decreasing noise level
        for n in range(args.num_users):
            noise[n] = args.noise_size - (n / (args.num_users - 1) * args.noise_size)
        ran = torch.randint(10, [len(train_dataset.targets)])
        for n in range(args.num_users):
            ber = torch.bernoulli(torch.add(torch.zeros(len(train_dataset.targets)), noise[n]))
            tmp = user_groups.copy()
            tmp.pop(n, None)
            for key in tmp.keys():
                ber[list(tmp[key])] = 0
            tmp = torch.as_tensor(train_dataset.targets)
            train_dataset.targets = torch.where(ber > 0, ran, tmp)

    # set cheat to correspond to method/parameter
    cheat = np.zeros([args.num_users])
    deviants = []
    if args.noise_type == 'freeride' or args.noise_type == 'attack':
        # first few are cheating (data split is random)
        deviants = [i for i in range(int(args.noise_size))]
        for n in deviants:
            if args.noise_type == 'attack':
                cheat[n] = -1
            elif args.noise_type == 'freeride':
                cheat[n] = 1
            else:
                cheat[n] = 0

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes, dropout=args.do)

    # Set the model to train and send it to device
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # contribution measuring variables
    contributors = np.zeros([args.epochs, int(args.num_users * args.frac)])
    test_improvement = np.zeros(args.epochs)
    weight = np.ones([args.epochs + 1, args.num_users])

    # initial model's accuracy
    test_acc = test_inference(args, global_model, test_dataset)[0]
    old_test_acc = test_acc

    # save starting accuracy
    with open(path_project + '/save/' + path + '/start.npy', 'wb') as f:
        np.save(f, np.array([100 * round(test_acc, 6)]))

    for epoch in tqdm(range(args.epochs)):

        local_weights = []
        global_model.train()
        m = max(round(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        count = 0

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w = local_model.update_weights(model=copy.deepcopy(global_model))[0]

            # inverting the gradient
            if cheat[idx] == -1:
                for key in w:
                    w[key] = 2 * global_model.state_dict()[key] - w[key]

            # weightening the contribution
            # ToDo weight aggregate instead of individual updates
            if args.weight != 0.:
                for key in w:
                    w[key] = global_model.state_dict()[key] + (w[key] - global_model.state_dict()[key]) * weight[epoch, idx]

            # always except freeriding add the new weights to the list
            if cheat[idx] != 1:
                local_weights.append(copy.deepcopy(w))

            # mark the contributors
            contributors[epoch, count] = idx
            count = count + 1

        # update global weights
        if not (args.noise_type == 'freeride' and len(np.intersect1d(idxs_users, deviants)) == round(args.num_users * args.frac)):
            if args.aggregate == 'avg':
                global_weights = average_weights(local_weights)
                global_model.load_state_dict(global_weights)
            elif args.aggregate == 'tm':
                global_weights = median_weights(local_weights, (1 - (args.num_users - 2 * args.robustness) / args.num_users) / 2, device)
                global_model.load_state_dict(global_weights)
            elif args.aggregate == 'KRUM':
                # ToDo
                global_weights = average_weights(local_weights)
                global_model.load_state_dict(global_weights)
            elif args.aggregate == 'RR':
                # ToDo
                global_weights = average_weights(local_weights)
                global_model.load_state_dict(global_weights)

        # measure the model test improvement epoch-wise
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_improvement[epoch] = 100 * round(test_acc - old_test_acc, 6)
        old_test_acc = test_acc

        # update participant weights
        if args.weight != 0. and epoch > 0:
            weight[epoch + 1] = weight[epoch]
            if test_improvement[epoch] < 0:
                for contributor in contributors[epoch]:
                    weight[epoch + 1, int(contributor)] = weight[epoch + 1, int(contributor)] * (1 - args.weight)
            if test_improvement[epoch] > test_improvement[epoch - 1]:
                for contributor in contributors[epoch]:
                    weight[epoch + 1, int(contributor)] = weight[epoch + 1, int(contributor)] * (1 + args.weight)
                for contributor in contributors[epoch - 1]:
                    weight[epoch + 1, int(contributor)] = weight[epoch + 1, int(contributor)] * (1 - args.weight)

    # Test inference after completion of training
    test_acc = test_inference(args, global_model, test_dataset)[0]

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # save results
    with open(path_project + '/save/' + path + '/contributor.npy', 'wb') as f:
        np.save(f, np.array(contributors))
    with open(path_project + '/save/' + path + '/weight.npy', 'wb') as f:
        np.save(f, np.array(weight))
    with open(path_project + '/save/' + path + '/deviant.npy', 'wb') as f:
        np.save(f, np.array(deviants))
    with open(path_project + '/save/' + path + '/test.npy', 'wb') as f:
        np.save(f, np.array(test_improvement))