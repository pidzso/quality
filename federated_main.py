import os
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        # Nvidia card ID = 0
        torch.cuda.set_device(0)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0

    # contribution measuring variables
    deviants = []
    contributors = np.zeros([args.epochs, int(args.num_users * args.frac)])
    improvement = np.zeros(args.epochs)
    weight = np.zeros([args.epochs + 1, args.num_users]) + 1
    score = np.zeros([args.epochs, args.num_users])
    norm = np.zeros([args.epochs])

    # initial model's accuracy
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    old_test_acc = test_acc

    # save starting accuracy
    file_string = path_project + '/save/result' + args.instance + '.txt'
    res = open(file_string, "w")
    print(f'Starting:\n {100 * round(old_test_acc, 6)}', file=res)
    res.close()

    noise = np.zeros([args.num_users])  # explicit noise
    cheat = np.zeros([args.num_users])  # for freeriding/attacking

    # set noise/cheat to correspond to method/parameter
    if args.noise_type == 'linear':
        # 1st participant: noise_size noise, last participant: 0 noise
        # in-between linearly decreasing noise level
        for n in range(args.num_users):
            noise[n] = args.noise_size - (n / (args.num_users - 1) * args.noise_size)
    elif args.noise_type == 'freeride' or args.noise_type == 'attack':
        deviants = [i for i in range(int(args.noise_size))]  # first few are cheating
        for n in deviants:
            if args.noise_type == 'attack':
                cheat[n] = -1
            elif args.noise_type == 'freeride':
                cheat[n] = 1
            else:
                cheat[n] = 0

    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        count = 0

        for idx in idxs_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, noise=noise[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model))

            # inverting the gradient
            if cheat[idx] == -1:
                for key in w:
                    w[key] = 2 * global_model.state_dict()[key] - w[key]

            # weightening the contribution
            if args.weight == 1:
                for key in w:
                    w[key] = global_model.state_dict()[key] + (w[key] - global_model.state_dict()[key]) * weight[epoch, idx]

            # always except freeriding add the new weights to the list
            if cheat[idx] != 1:
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))  # TODO loss is meaningless as of now if attack

            # mark the contributors
            contributors[epoch, count] = idx
            count = count + 1

        # update global weights
        if not (args.noise_type == 'freeride' and
                len([i for i in set(idxs_users) if i in deviants]) == round(args.num_users * args.frac)):
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

        # measure the model improvement epoch-wise
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        improvement[epoch] = 100 * round(test_acc - old_test_acc, 6)
        old_test_acc = test_acc

        # update participant weights
        kappa = 0.95
        if args.weight == 1 and epoch > 0:
            score[epoch] = score[epoch - 1]
            weight[epoch + 1] = weight[epoch]
            for contributor in contributors[epoch]:
                norm[epoch] = norm[epoch] + weight[epoch, int(contributor)] / (args.num_users * args.frac)
            if improvement[epoch] < 0:
                for contributor in contributors[epoch]:
                    weight[epoch, int(contributor)] = weight[epoch, int(contributor)] * kappa
            if improvement[epoch] > improvement[epoch - 1]:
                for contributor in contributors[epoch]:
                    weight[epoch, int(contributor)] = weight[epoch, int(contributor)] * 1 / kappa
                for contributor in contributors[epoch - 1]:
                    weight[epoch, int(contributor)] = weight[epoch, int(contributor)] * kappa

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # save run results
    np.set_printoptions(threshold=10000)
    res = open(file_string, "a+")
    print(f'Deviants:\n {deviants}', file=res)
    print(f'Participants:\n {contributors}', file=res)
    print(f'Improvements:\n {improvement}', file=res)
    print(f'Weights:\n {weight[args.epochs]}', file=res)
    res.close()
    print('\n Total Run Time: {0:0.0f}'.format(time.time()-start_time))
