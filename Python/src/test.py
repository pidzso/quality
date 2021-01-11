import numpy as np
import matplotlib.pyplot as plt
from q_inf import test
from scipy import stats
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def accuracy_boxplot(model, data, participants, noise_type, noise_size, weight, rounds, instance_num):
    '''
    creates boxplot of round-wise accuracies when 'linear'
    creates round-wise accuracies based on test and train when 'no'
    '''

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
        plt.xlim(-1, 100)
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
    '''
    creates average round-wise accuracies for different weights
    '''

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
    '''
    creates round-wise the participants' weight changes
    '''

    test = np.zeros((instance_num, rounds + 1, participants))
    path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + noise_type + '_' + str(noise_size) + '_' + str(weight) + '/'
    for i in np.arange(instance_num):
        test[i] = np.load(path + str(i + 1) + '/weight.npy')

    AVGtest = np.transpose(np.average(test, axis=0))
    for i in np.arange(participants):
        plt.plot(np.arange(rounds + 1), AVGtest[i], color=str(i / participants))

    plt.title(model + '_' + data + '_' + str(participants))
    plt.xlabel('Rounds')
    plt.ylabel('Weights')
    plt.savefig(path + 'change.png')
    plt.close()


def score(model, data, participants, instance):
    '''
    creates boxplot of participant-wise quality scores
    '''

    scores = np.zeros((instance, participants))

    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '/'
    for i in np.arange(instance):
        scores[i] = test(path0 + str(i + 1), ['neg', 'inc', 'help'], 'count', 0,           0,          0)
                      #  what,               how,                     option, ignorefirst, ignorelast, treshold

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


def order(model, data, participants, instance, ignorefirst, ignorelast, treshold, type):
    '''
    calculates the quality inference's accuracy instance-wise
    '''

    scores = np.zeros((instance, participants))
    sp = np.zeros(instance)
    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '/'

    for i in np.arange(instance):
        scores[i] = test(path0 + str(i + 1), ['neg', 'inc', 'help'], type, ignorefirst, ignorelast, treshold)
        sp[i] = stats.spearmanr(np.arange(participants), scores[i])[0]

    return sp


def find_opt(model, data, participants, instance):
    '''
    finds the optimal parameters to maximize the average score
    '''

    ignorefirst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    treshold = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    best = [0, 0.0, 'count']
    for ty in ['count', 'value']:
        for ig in ignorefirst:
            for th in treshold:
                if np.mean(order(model, data, participants, instance, best[0], 0, best[1], best[2])) < np.mean(order(model, data, participants, instance, ig, 0, th, ty)):
                    best = [ig, th, ty]
    return best


def order_show(what, opt):
    '''
    creates boxplot of experiment-wise quality inference accuracies
    with default parameters / with default and optimal parameters
    '''

    mm5 = order('mlp', 'mnist', 5, 9, 0, 0, 0, 'count')
    mm25 = order('mlp', 'mnist', 25, 9, 0, 0, 0, 'count')
    #mm100 = order('mlp', 'mnist', 100, 3, 0, 0, 0, 'count')
    mc5 = order('mlp', 'cifar', 5, 9, 0, 0, 0, 'count')
    mc25 = order('mlp', 'cifar', 25, 9, 0, 0, 0, 'count')
    #mc100 = order('mlp', 'cifar', 100, 3, 0, 0, 0, 'count')
    cm5 = order('cnn', 'mnist', 5, 9, 0, 0, 0, 'count')
    cm25 = order('cnn', 'mnist', 25, 9, 0, 0, 0, 'count')
    #cm100 = order('cnn', 'mnist', 100, 3, 0, 0, 0, 'count')
    cc5 = order('cnn', 'cifar', 5, 9, 0, 0, 0, 'count')
    cc25 = order('cnn', 'cifar', 25, 9, 0, 0, 0, 'count')
    #cc100 = order('cnn', 'cifar', 100, 3, 0, 0, 0, 'count')

    opt_mm5 = order('mlp', 'mnist', 5, 9, 1, 0, 0.16, 'count')
    opt_mm25 = order('mlp', 'mnist', 25, 9, 0, 0, 0.16, 'count')
    #opt_mm100 = order('mlp', 'mnist', 100, 3, 0, 0, 0, 'count')
    opt_mc5 = order('mlp', 'cifar', 5, 9, 0, 0, 0.02, 'count')
    opt_mc25 = order('mlp', 'cifar', 25, 9, 2, 0, 00.16, 'count')
    #opt_mc100 = order('mlp', 'cifar', 100, 3, 0, 0, 0, 'count')
    opt_cm5 = order('cnn', 'mnist', 5, 9, 0, 0, 1.28, 'count')
    opt_cm25 = order('cnn', 'mnist', 25, 9, 8, 0, 0.32, 'count')
    #opt_cm100 = order('cnn', 'mnist', 100, 3, 0, 0, 0, 'count')
    opt_cc5 = order('cnn', 'cifar', 5, 9, 0, 0, 0.08, 'count')
    opt_cc25 = order('cnn', 'cifar', 25, 9, 2, 0, 0.64, 'count')
    #opt_cc100 = order('cnn', 'cifar', 100, 3, 0, 0, 0, 'count')

    #opt2_mc5 = order('mlp', 'cifar', 5, 9, 1, 0, 0, 'value')
    #opt2_cm5 = order('cnn', 'mnist', 5, 9, 1, 0, 0, 'value')
    #opt2_mm25 = order('mlp', 'mnist', 25, 9, 1, 0, 0.01, 'value')

    # add 100
    if opt == 1:
        l = 8
        mins =  {'5': np.min( [mm5, opt_mm5, mc5, opt_mc5, cm5, opt_cm5, cc5, opt_cc5], axis=1),
                 '25': np.min( [mm25, opt_mm25, mc25, opt_mc25, cm25, opt_cm25, cc25, opt_cc25], axis=1),
                 '100': np.min( [])}
        maxes = {'5': np.max( [mm5, opt_mm5, mc5, opt_mc5, cm5, opt_cm5, cc5, opt_cc5], axis=1),
                 '25': np.max( [mm25, opt_mm25, mc25, opt_mc25, cm25, opt_cm25, cc25, opt_cc25], axis=1),
                 '100': np.max( [])}
        means = {'5': np.mean([mm5, opt_mm5, mc5, opt_mc5, cm5, opt_cm5, cc5, opt_cc5], axis=1),
                 '25': np.mean([mm25, opt_mm25, mc25, opt_mc25, cm25, opt_cm25, cc25, opt_cc25], axis=1),
                 '100': np.mean([])}
        std =   {'5': np.std( [mm5, opt_mm5, mc5, opt_mc5, cm5, opt_cm5, cc5, opt_cc5], axis=1),
                 '25': np.std( [mm25, opt_mm25, mc25, opt_mc25, cm25, opt_cm25, cc25, opt_cc25], axis=1),
                 '100': np.std( [])}
    else:
        l = 4
        mins =  {'5': np.min( [mm5, mc5, cm5, cc5], axis=1),
                 '25': np.min( [mm25, mc25, cm25, cc25], axis=1),
                 '100': np.min( [])}
        maxes = {'5': np.max( [mm5, mc5, cm5, cc5], axis=1),
                 '25': np.max( [mm25, mc25, cm25, cc25], axis=1),
                 '100': np.max( [])}
        means = {'5': np.mean([mm5, mc5, cm5, cc5], axis=1),
                 '25': np.mean([mm25, mc25, cm25, cc25], axis=1),
                 '100': np.mean([])}
        std =   {'5': np.std( [mm5, mc5, cm5, cc5], axis=1),
                 '25': np.std( [mm25, mc25, cm25, cc25], axis=1),
                 '100': np.std( [])}

    for par in what:
        fig, ax = plt.subplots()
        plt.errorbar(np.arange(l), means[par], std[par], fmt='ok', lw=3)
        plt.errorbar(np.arange(l), means[par], [means[par] - mins[par], maxes[par] - means[par]], fmt='.k', ecolor='gray', lw=1)
        plt.xlim(-1, l)
        plt.title('Quality Inference')
        plt.ylabel('QI Accuracy')
        if opt == 1:
            ax.set_xticklabels(['', 'mm' + par, 'OPTmm' + par, 'mc' + par, 'OPTmc' + par, 'cm' + par, 'OPTcm' + par, 'cc' + par, 'OPTcc' + par, ''], rotation=45)
            plt.savefig('../save/qi_' + par + '_opt.png')
        else:
            ax.set_xticklabels(['', 'mm' + par, 'mc' + par, 'cm' + par, 'cc' + par, ''], rotation=45)
            plt.savefig('../save/qi_' + par + '.png')
        plt.close()




#mm5 = find_opt('mlp', 'mnist', 5, 9)    # ig=1, th=.16
#mc5 = find_opt('mlp', 'cifar', 5, 9)    # ig=0, th=.02   value(1/0)
#cm5 = find_opt('cnn', 'mnist', 5, 9)    # ig=0, th=1.28  value(1/0)
#cc5 = find_opt('cnn', 'cifar', 5, 9)    # ig=0, th=.08
#mm25 = find_opt('mlp', 'mnist', 25, 9)  # ig=0, th=.16   value(1/.01)
#mc25 = find_opt('mlp', 'cifar', 25, 9)  # ig=2, th=.16
#cm25 = find_opt('cnn', 'mnist', 25, 9)  # ig=8, th=.32
#cc25 = find_opt('cnn', 'cifar', 25, 9)  # ig=2, th=.64

# do it for ['count'] and ['count', 'value']
#mm100 = find_opt('mlp', 'mnist', 100, 3)
#mc100 = find_opt('mlp', 'cifar', 100, 3)
#cm100 = find_opt('cnn', 'mnist', 100, 3)
#cc100 = find_opt('cnn', 'cifar', 100, 3)

#order_show(['100'], 0)
#order_show(['100'], 1)

# do it for all weights as well + no
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
