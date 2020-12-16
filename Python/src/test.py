from q_inf import aggregate, test, quality_inf, catch, optimize, test, highest, avg, readOLD
import numpy as np
import matplotlib.pyplot as plt


def create_fig(model, data, participants, noise_type, noise_size, weight, rounds, instance_num):

    train = np.zeros((instance_num, rounds + 1))
    test = np.zeros((instance_num, rounds + 1))
    path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + noise_type + '_' + str(noise_size) + '_' + str(weight) + '/'

    for i in np.arange(instance_num):
        train[i] = np.cumsum(np.insert(np.load(path + str(i + 1) + '/train.npy'), 0, np.load(path + str(i + 1) + '/start.npy')))
        test[i] = np.cumsum(np.insert(np.load(path + str(i + 1) + '/test.npy'), 0, np.load(path + str(i + 1) + '/start.npy')))

    mins = np.transpose(test).min(axis=1)
    maxes = np.transpose(test).max(axis=1)
    means = np.transpose(test).mean(axis=1)
    std = np.transpose(test).std(axis=1)

    if noise_type == 'linear':
        plt.errorbar(np.arange(rounds + 1), means, std, fmt='ok', lw=3)
        plt.errorbar(np.arange(rounds + 1), means, [means - mins, maxes - means],
                     fmt='.k', ecolor='gray', lw=1)
        plt.xlim(-1, 101)
    elif noise_type == 'no':
        avg_tr = np.average(train, axis=0)
        avg_te = np.average(test, axis=0)

        plt.plot(np.arange(rounds + 1), avg_tr, 'r', label='Train')
        plt.plot(np.arange(rounds + 1), avg_te, 'b', label='Test')

    plt.legend()
    plt.title(model + '_' + data + '_' + str(participants))
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.savefig(path + 'accuracy.png')
    plt.show()
    plt.close()


def weight_diff(model, data, participants, noise_type, noise_size, weights, rounds, instance_num):

    test = np.zeros((len(weights), instance_num, rounds + 1))
    bl = np.zeros((instance_num, rounds + 1))
    AVGbl = np.zeros((rounds + 1))
    AVGtest = np.zeros((len(weights), rounds + 1))
    color = ['r', 'b', 'y']
    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/'

    for w in np.arange(len(weights)):
        path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + noise_type + '_' + str(noise_size) + '_' + str(weights[w]) + '/'
        pathBL = '../save/' + model + '_' + data + '_' + str(participants) + '/no_0.0_0.0/'
        for i in np.arange(instance_num):
            test[w][i] = np.cumsum(np.insert(np.load(path + str(i + 1) + '/test.npy'), 0, np.load(path + str(i + 1) + '/start.npy')))
            bl[i] = np.cumsum(np.insert(np.load(pathBL + str(i + 1) + '/test.npy'), 0, np.load(pathBL + str(i + 1) + '/start.npy')))

        AVGtest[w] = np.average(test[w], axis=0)
        AVGbl = np.average(bl, axis=0)
        plt.plot(np.arange(rounds + 1), AVGtest[w], color[w], label='Weight=' + str(weights[w]))

    plt.plot(np.arange(rounds + 1), AVGbl, 'g', label='No Noise')

    plt.legend()
    plt.title(model + '_' + data + '_' + str(participants))
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.savefig(path0 + 'weight.png')
    plt.show()
    plt.close()


#create_fig('mlp', 'mnist', 5, 'linear', 1.0, 0.1, 100, 9)
#create_fig('mlp', 'cifar', 5, 'linear', 1.0, 0.1, 100, 9)
#create_fig('cnn', 'mnist', 5, 'linear', 1.0, 0.1, 100, 9)
#create_fig('cnn', 'cifar', 5, 'linear', 1.0, 0.1, 100, 9)

weight_diff('mlp', 'mnist', 5, 'linear', 1.0, [0.0, 0.05], 100, 9)
weight_diff('mlp', 'cifar', 5, 'linear', 1.0, [0.0, 0.05], 100, 9)
weight_diff('cnn', 'mnist', 5, 'linear', 1.0, [0.0, 0.05], 100, 9)
weight_diff('cnn', 'cifar', 5, 'linear', 1.0, [0.0, 0.05], 100, 9)
