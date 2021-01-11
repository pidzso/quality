import numpy as np
import matplotlib.pyplot as plt
from q_inf import test, quality_inf


def accuracy_boxplot(model, data, participants, noise_type, noise_size, weight, rounds, instance_num):

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
    plt.close()


def weight_diff(model, data, participants, noise_type, noise_size, weights, rounds, instance_num):

    test = np.zeros((len(weights), instance_num, rounds + 1))
    AVGtest = np.zeros((len(weights), rounds + 1))
    color = ['r', 'b', 'y', 'g']
    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/'

    for w in np.arange(len(weights)):
        path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + noise_type + '_' + str(noise_size) + '_' + str(weights[w]) + '/'
        for i in np.arange(instance_num):
            test[w][i] = np.cumsum(np.insert(np.load(path + str(i + 1) + '/test.npy'), 0, np.load(path + str(i + 1) + '/start.npy')))

        AVGtest[w] = np.average(test[w], axis=0)
        plt.plot(np.arange(rounds + 1), AVGtest[w], color[w], label='Weight=' + str(weights[w]))

    plt.legend()
    plt.title(model + '_' + data + '_' + str(participants))
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.savefig(path0 + 'weight.png')
    plt.close()


def weight_change(model, data, participants, noise_type, noise_size, weight, rounds, instance_num):

    test = np.zeros((instance_num, rounds + 1, participants))
    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/'
    path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + noise_type + '_' + str(noise_size) + '_' + str(weight) + '/'
    for i in np.arange(instance_num):
        test[i] = np.load(path + str(i + 1) + '/weight.npy')

    AVGtest = np.transpose(np.average(test, axis=0))
    for i in np.arange(participants):
        plt.plot(np.arange(rounds + 1), AVGtest[i], color=str(i / participants))

    plt.title(model + '_' + data + '_' + str(participants))
    plt.xlabel('Rounds')
    plt.ylabel('Weights')
    plt.savefig(path0 + 'change.png')
    plt.close()


def score(model, data, participants, instance):

    scores = np.zeros((instance, participants))
    mins = np.zeros(participants)
    maxes = np.zeros(participants)
    means = np.zeros(participants)
    std = np.zeros(participants)

    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '/'
    for i in np.arange(instance):
        scores[i] = test(path0 + str(i + 1), ['neg', 'inc', 'help'], 'count', 0, 0, 0)  # (what, how, option, ignorefirst, ignorelast, treshold)

    mins = np.min(scores, axis=0)
    maxes = np.max(scores, axis=0)
    means = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)

    plt.errorbar(np.arange(participants), means, std, fmt='ok', lw=3)
    plt.errorbar(np.arange(participants), means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)
    plt.xlim(-1, participants)
    plt.title(model + '_' + data + '_score')
    plt.xlabel('Participant')
    plt.ylabel('Score')
    plt.savefig(path0 + 'score.png')
    plt.close()


#score('mlp', 'mnist', 100, 3)
#score('mlp', 'cifar', 100, 3)
#score('cnn', 'mnist', 100, 3)
#score('cnn', 'cifar', 100, 3)

#weight_change('mlp', 'mnist', 100, 'linear', 1.0, 0.05, 100, 3)
#weight_change('mlp', 'mnist', 100, 'linear', 1.0, 0.1, 100, 3)
#weight_change('mlp', 'mnist', 100, 'linear', 1.0, 0.2, 100, 3)

#weight_change('mlp', 'cifar', 100, 'linear', 1.0, 0.05, 100, 3)
#weight_change('mlp', 'cifar', 100, 'linear', 1.0, 0.1, 100, 3)
#weight_change('mlp', 'cifar', 100, 'linear', 1.0, 0.2, 100, 3)

#weight_change('cnn', 'mnist', 100, 'linear', 1.0, 0.05, 100, 3)
#weight_change('cnn', 'mnist', 100, 'linear', 1.0, 0.1, 100, 3)
#weight_change('cnn', 'mnist', 100, 'linear', 1.0, 0.2, 100, 3)

#weight_change('cnn', 'cifar', 100, 'linear', 1.0, 0.05, 100, 3)
#weight_change('cnn', 'cifar', 100, 'linear', 1.0, 0.1, 100, 3)
#weight_change('cnn', 'cifar', 100, 'linear', 1.0, 0.2, 100, 3)

#accuracy_boxplot('mlp', 'mnist', 100, 'no', 0.0, 0.0, 100, 3)
#accuracy_boxplot('mlp', 'cifar', 100, 'no', 0.0, 0.0, 100, 3)
#accuracy_boxplot('cnn', 'mnist', 100, 'no', 0.0, 0.0, 100, 3)
#accuracy_boxplot('cnn', 'cifar', 100, 'no', 0.0, 0.0, 100, 3)

#accuracy_boxplot('mlp', 'mnist', 100, 'linear', 1.0, 0.2, 100, 3)
#accuracy_boxplot('mlp', 'cifar', 100, 'linear', 1.0, 0.2, 100, 3)
#accuracy_boxplot('cnn', 'mnist', 100, 'linear', 1.0, 0.2, 100, 3)
#accuracy_boxplot('cnn', 'cifar', 100, 'linear', 1.0, 0.2, 100, 3)

#accuracy_boxplot('mlp', 'mnist', 100, 'linear', 1.0, 0.1, 100, 3)
#accuracy_boxplot('mlp', 'cifar', 100, 'linear', 1.0, 0.1, 100, 3)
#accuracy_boxplot('cnn', 'mnist', 100, 'linear', 1.0, 0.1, 100, 3)
#accuracy_boxplot('cnn', 'cifar', 100, 'linear', 1.0, 0.1, 100, 3)

#accuracy_boxplot('mlp', 'mnist', 100, 'linear', 1.0, 0.05, 100, 3)
#accuracy_boxplot('mlp', 'cifar', 100, 'linear', 1.0, 0.05, 100, 3)
#accuracy_boxplot('cnn', 'mnist', 100, 'linear', 1.0, 0.05, 100, 3)
#accuracy_boxplot('cnn', 'cifar', 100, 'linear', 1.0, 0.05, 100, 3)

#accuracy_boxplot('mlp', 'mnist', 100, 'linear', 1.0, 0.0, 100, 3)
#accuracy_boxplot('mlp', 'cifar', 100, 'linear', 1.0, 0.0, 100, 3)
#accuracy_boxplot('cnn', 'mnist', 100, 'linear', 1.0, 0.0, 100, 3)
#accuracy_boxplot('cnn', 'cifar', 100, 'linear', 1.0, 0.0, 100, 3)

#weight_diff('mlp', 'mnist', 100, 'linear', 1.0, [0.0, 0.05, 0.1, 0.2], 100, 3)
#weight_diff('mlp', 'cifar', 100, 'linear', 1.0, [0.0, 0.05, 0.1, 0.2], 100, 3)
#weight_diff('cnn', 'mnist', 100, 'linear', 1.0, [0.0, 0.05, 0.1, 0.2], 100, 3)
#weight_diff('cnn', 'cifar', 100, 'linear', 1.0, [0.0, 0.05, 0.1, 0.2], 100, 3)
