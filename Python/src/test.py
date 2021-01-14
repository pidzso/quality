import numpy as np
import matplotlib
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

    plt.title(model + '_' + data + '_' + str(participants), fontsize=20)
    plt.xlabel('Rounds', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
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

    plt.legend(fontsize=20)
    plt.title(model + '_' + data + '_' + str(participants), fontsize=20)
    plt.xlabel('Rounds', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.savefig('../save/weight_' + model[0] + data[0] + str(participants) + '.png')
    plt.close()


weight_diff('mlp', 'mnist', 5, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)
weight_diff('mlp', 'cifar', 5, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)
weight_diff('cnn', 'mnist', 5, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)
weight_diff('cnn', 'cifar', 5, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)

weight_diff('mlp', 'mnist', 25, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)
weight_diff('mlp', 'cifar', 25, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)
weight_diff('cnn', 'mnist', 25, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)
weight_diff('cnn', 'cifar', 25, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 9)

weight_diff('mlp', 'mnist', 100, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 3)
weight_diff('mlp', 'cifar', 100, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 3)
weight_diff('cnn', 'mnist', 100, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 3)
weight_diff('cnn', 'cifar', 100, 'linear', 1.0, [0.0, 0.05, 0.10, 0.20], 100, 3)


def weight_change(model, data, participants, noise_type, noise_size, weight, rounds, instance):
    '''
    creates round-wise the participants' weight changes
    '''

    test = np.zeros((instance, rounds + 1, participants))
    path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + noise_type + '_' + str(noise_size) + '_' + str(weight) + '/'
    for i in np.arange(instance):
        test[i] = np.load(path + str(i + 1) + '/weight.npy')

    AVGtest = np.transpose(np.average(test, axis=0))
    for i in np.arange(participants):
        plt.plot(np.arange(rounds + 1), AVGtest[i], color=str(i / participants))

    plt.title(model + '_' + data + '_' + str(participants), fontsize=20)
    plt.xlabel('Rounds', fontsize=20)
    plt.ylabel('Weights', fontsize=20)
    plt.savefig(path + 'change.png')
    plt.close()


def score_change(model, data, participants, noise_type, noise_size, rounds, instance):
    '''
    creates round-wise the participants' score changes
    '''

    scores = np.zeros((rounds, instance, participants))

    path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '/'
    for i in np.arange(instance):
        for r in np.arange(rounds):
            scores[r][i] = test(path + str(i + 1), ['neg', 'inc', 'help'], 'count', 0, 99 - r, 0)

    AVGscores = np.transpose(np.average(scores, axis=1))
    for i in np.arange(participants):
        plt.plot(np.arange(rounds), AVGscores[i], color=str(i / participants))

    plt.title(model + '_' + data + '_' + str(participants), fontsize=20)
    plt.xlabel('Rounds', fontsize=20)
    plt.ylabel('Scores', fontsize=20)
    plt.savefig('../save/change_score_' + model[0] + data[0] + str(participants) + '.png')
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
    plt.title(model + '_' + data + '_' + str(participants), fontsize=20)
    plt.xlabel('Participant', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig('../save/score_' + model[0] + data[0] + str(participants) + '.png')
    plt.close()


def order(model, data, participants, instance, ignorefirst, ignorelast, treshold, type, tests):
    '''
    calculates the quality inference's accuracy instance-wise
    '''

    scores = np.zeros((instance, participants))
    sp = np.zeros(instance)
    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '/'

    for i in np.arange(instance):
        scores[i] = test(path0 + str(i + 1), tests, type, ignorefirst, ignorelast, treshold)
        sp[i] = stats.spearmanr(np.arange(participants), scores[i])[0]
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

    return sp


def find_opt(model, data, participants, instance, tests):
    '''
    finds the optimal parameters to maximize the average score
    '''

    ignorefirst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    treshold = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    best = [0, 0.0, 'count']
    for ty in ['count', 'value']:
        for ig in ignorefirst:
            for th in treshold:
                if np.mean(order(model, data, participants, instance, best[0], 0, best[1], best[2], tests)) < np.mean(order(model, data, participants, instance, ig, 0, th, ty)):
                    best = [ig, th, ty]
    return best


def order_show(participants):
    '''
    creates boxplot of experiment-wise quality inference accuracies
    with default parameters / with default and optimal parameters
    '''

    mins = {}
    maxes = {}
    means = {}
    std = {}

    OPTmins = {}
    OPTmaxes = {}
    OPTmeans = {}
    OPTstd = {}

    tests = [['help', 'inc', 'neg']]#, ['help', 'inc'], ['help', 'neg'],['inc', 'neg'], ['help'], ['inc'], ['neg']]

    for i in np.arange(len(tests)):
        mm5   = order('mlp', 'mnist', 5, 9, 0, 0, 0, 'count', tests[i])
        mm25  = order('mlp', 'mnist', 25, 9, 0, 0, 0, 'count', tests[i])
        mm100 = order('mlp', 'mnist', 100, 3, 0, 0, 0, 'count', tests[i])
        mc5   = order('mlp', 'cifar', 5, 9, 0, 0, 0, 'count', tests[i])
        mc25  = order('mlp', 'cifar', 25, 9, 0, 0, 0, 'count', tests[i])
        mc100 = order('mlp', 'cifar', 100, 3, 0, 0, 0, 'count', tests[i])
        cm5   = order('cnn', 'mnist', 5, 9, 0, 0, 0, 'count', tests[i])
        cm25  = order('cnn', 'mnist', 25, 9, 0, 0, 0, 'count', tests[i])
        cm100 = order('cnn', 'mnist', 100, 3, 0, 0, 0, 'count', tests[i])
        cc5   = order('cnn', 'cifar', 5, 9, 0, 0, 0, 'count', tests[i])
        cc25  = order('cnn', 'cifar', 25, 9, 0, 0, 0, 'count', tests[i])
        cc100 = order('cnn', 'cifar', 100, 3, 0, 0, 0, 'count', tests[i])

        OPTmm5   = order('mlp', 'mnist', 5, 9, 1, 0, 0.16, 'count', tests[i])
        OPTmm25  = order('mlp', 'mnist', 25, 9, 1, 0, 0.01, 'value', tests[i])
        OPTmm100 = order('mlp', 'mnist', 100, 3, 1, 0, 0.32, 'count', tests[i])
        OPTmc5   = order('mlp', 'cifar', 5, 9, 1, 0, 0, 'value', tests[i])
        OPTmc25  = order('mlp', 'cifar', 25, 9, 2, 0, 0.16, 'count', tests[i])
        OPTmc100 = order('mlp', 'cifar', 100, 3, 6, 0, 0.08, 'count', tests[i])
        OPTcm5   = order('cnn', 'mnist', 5, 9, 1, 0, 0, 'value', tests[i])
        OPTcm25  = order('cnn', 'mnist', 25, 9, 8, 0, 0.32, 'count', tests[i])
        OPTcm100 = order('cnn', 'mnist', 100, 3, 9, 0, 0.16, 'count', tests[i])
        OPTcc5   = order('cnn', 'cifar', 5, 9, 0, 0, 0.08, 'count', tests[i])
        OPTcc25  = order('cnn', 'cifar', 25, 9, 2, 0, 0.64, 'count', tests[i])
        OPTcc100 = order('cnn', 'cifar', 100, 3, 4, 0, 1.28, 'count', tests[i])

        mins['5' + str(i)]    = np.min([mm5, mc5, cm5, cc5], axis=1)
        mins['25' + str(i)]   = np.min([mm25, mc25, cm25, cc25], axis=1)
        mins['100' + str(i)]  = np.min([mm100, mc100, cm100, cc100], axis=1)
        maxes['5' + str(i)]   = np.max([mm5, mc5, cm5, cc5], axis=1)
        maxes['25' + str(i)]  = np.max([mm25, mc25, cm25, cc25], axis=1)
        maxes['100' + str(i)] =  np.max([mm100, mc100, cm100, cc100], axis=1)
        means['5' + str(i)]   = np.mean([mm5, mc5, cm5, cc5], axis=1)
        means['25' + str(i)]  = np.mean([mm25, mc25, cm25, cc25], axis=1)
        means['100' + str(i)] = np.mean([mm100, mc100, cm100, cc100], axis=1)
        std['5' + str(i)]     = np.std([mm5, mc5, cm5, cc5], axis=1)
        std['25' + str(i)]    = np.std([mm25, mc25, cm25, cc25], axis=1)
        std['100' + str(i)]   = np.std([mm100, mc100, cm100, cc100], axis=1)

        OPTmins['5' + str(i)]    = np.min( [mm5,   OPTmm5,   mc5,   OPTmc5,   cm5,   OPTcm5,   cc5,   OPTcc5],   axis=1)
        OPTmins['25' + str(i)]   = np.min( [mm25,  OPTmm25,  mc25,  OPTmc25,  cm25,  OPTcm25,  cc25,  OPTcc25],  axis=1)
        OPTmins['100' + str(i)]  = np.min( [mm100, OPTmm100, mc100, OPTmc100, cm100, OPTcm100, cc100, OPTcc100], axis=1)
        OPTmaxes['5' + str(i)]   = np.max( [mm5,   OPTmm5,   mc5,   OPTmc5,   cm5,   OPTcm5,   cc5,   OPTcc5],   axis=1)
        OPTmaxes['25' + str(i)]  = np.max( [mm25,  OPTmm25,  mc25,  OPTmc25,  cm25,  OPTcm25,  cc25,  OPTcc25],  axis=1)
        OPTmaxes['100' + str(i)] =  np.max([mm100, OPTmm100, mc100, OPTmc100, cm100, OPTcm100, cc100, OPTcc100], axis=1)
        OPTmeans['5' + str(i)]   = np.mean([mm5,   OPTmm5,   mc5,   OPTmc5,   cm5,   OPTcm5,   cc5,   OPTcc5],   axis=1)
        OPTmeans['25' + str(i)]  = np.mean([mm25,  OPTmm25,  mc25,  OPTmc25,  cm25,  OPTcm25,  cc25,  OPTcc25],  axis=1)
        OPTmeans['100' + str(i)] = np.mean([mm100, OPTmm100, mc100, OPTmc100, cm100, OPTcm100, cc100, OPTcc100], axis=1)
        OPTstd['5' + str(i)]     = np.std( [mm5,   OPTmm5,   mc5,   OPTmc5,   cm5,   OPTcm5,   cc5,   OPTcc5],   axis=1)
        OPTstd['25' + str(i)]    = np.std( [mm25,  OPTmm25,  mc25,  OPTmc25,  cm25,  OPTcm25,  cc25,  OPTcc25],  axis=1)
        OPTstd['100' + str(i)]   = np.std( [mm100, OPTmm100, mc100, OPTmc100, cm100, OPTcm100, cc100, OPTcc100], axis=1)

    for par in participants:
        fig, ax = plt.subplots()
        for i in np.arange(len(tests)):
            plt.errorbar(np.arange(i, 8*len(tests), len(tests)), OPTmeans[par + str(i)], OPTstd[par + str(i)], fmt='ok', lw=3)  #, ecolor=['red', 'green', 'blue', 'brown'])
            plt.errorbar(np.arange(i, 8*len(tests), len(tests)), OPTmeans[par + str(i)], [OPTmeans[par + str(i)] - OPTmins[par + str(i)], OPTmaxes[par + str(i)] - OPTmeans[par + str(i)]], fmt='.k', lw=1, ecolor='gray')#ecolor=['red', 'green', 'blue', 'brown'])
        plt.xlim(-1, 8*len(tests))
        plt.title(par, fontsize=20)
        plt.ylabel('QI Accuracy', fontsize=20)
        ax.set_xticklabels(['', 'MM' + par, 'oMM' + par, 'MC' + par, 'oMC' + par, 'CM' + par, 'oCM' + par, 'CC' + par, 'oCC' + par, ''], rotation=90, fontsize=20)
        plt.savefig('../save/qi_' + par + '_OPT.png')
        plt.close()


def compare_order(model, data, participants, instance):
    '''
    compare the quality inference's accuracy case-wise
    '''

    scores = np.zeros((7, instance, participants))
    sp = np.zeros((7, instance))
    path0 = '../save/' + model + '_' + data + '_' + str(participants) + '/' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '/'
    tests = [['neg', 'inc', 'help'], ['inc', 'help'], ['neg', 'help'], ['neg', 'inc'], ['help'], ['inc'], ['neg']]
    for k in np.arange(7):
        for i in np.arange(instance):
            scores[k][i] = test(path0 + str(i + 1), tests[k], 'count', 0, 0, 0)
            sp[k][i] = stats.spearmanr(np.arange(participants), scores[k][i])[0]

    mins =  np.min(sp, axis=1)
    maxes = np.max(sp, axis=1)
    means = np.mean(sp, axis=1)
    std =   np.std(sp, axis=1)

    fig, ax = plt.subplots()
    plt.errorbar(np.arange(7), means, std, fmt='ok', lw=3)
    plt.errorbar(np.arange(7), means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)
    plt.xlim(-1, 7)
    plt.title('Quality Inference', fontsize=20)
    plt.ylabel('QI Accuracy')
    ax.set_xticklabels(['', 'NIH', 'IH', 'NH', 'NI', 'H', 'I', 'N', ''], rotation=45)
    plt.savefig('../save/' + model + '_' + data + '_' + str(participants) + '/QI.png')
    plt.close()


#      neg           inc           help         inchelp       neghelp       neginc        neginchelp
#mm5   ig=0 th=.08   ig=3 th=.16   ig=7 th=.02  ig=1 th=.04   ig=0 th=.02   ig=0 th=.08   ig=1 th=.16
#mc5   ig=9 th=.08   ig=9 th=.04   ig=0 th=.04  ig=10 th=.32  ig=9 th=.02   ig=8 th=.32   ig=0 th=.02
#cm5   ig=0 th=2.56  ig=0 th=.64   ig=0 th=0.2  ig=5 th=1.28  ig=6 th=2.56  ig=0 th=.64   ig=0 th=1.2
#cc5   ig=0 th=2.56  ig=4 th=.08   ig=7 th=0    ig=4 th=.43   ig=3 th=0     ig=0 th=0     ig=0 th=.08
#mm25  ig=0 th=.16   ig=0 th=.64   ig=8 th=.16  ig=1 th=.16   ig=8 th=.16   ig=0 th=.64   ig=0 th=.16
#mc25  ig=2 th=.16   ig=4 th=.32   ig=0 th=.08  ig=2 th=.16   ig=0 th=.16   ig=4 th=.16   ig=2 th=.16
#cm25  ig=4 th=.32   ig=0 th=.64   ig=10 th=0   ig=7 th=.32   ig=7 th=.02   ig=0 th=.64   ig=8 th=.32
#cc25  ig=7 th=.16   ig=4 th=.64   ig=4 th=.08  ig=4 th=.64   ig=4 th=.08   ig=4 th=.64   ig=2 th=.64
#mm100 ig=0 th=0     ig=0 th=.32   ig=8 th=.32  ig=1 th=.32   ig=1 th=.32   ig=0 th=.04   ig=1 th=.32
#mc100 ig=5 th=.01   ig=3 th=.16   ig=8 th=.04  ig=6 th=.08   ig=8 th=.01   ig=3 th=.08   ig=6 th=.08
#cm100 ig=0 th=.16   ig=9 th=.32   ig=0 th=.02  ig=9 th=.16   ig=8 th=.16   ig=9 th=.32   ig=9 th=.16
#cc100 ig=8 th=.64   ig=2 th=1.28  ig=6 th=.32  ig=7 th=0     ig=7 th=.04   ig=3 th=1.28  ig=4 th=1.28

#VALUE  neg           inc           inchelp      neginc       neginchelp
#mc5    ig=0 th=.64  ig=0 th=.64   ig=1 th=0    ig=0 th=.64   ig=1 th=0
#cm5    ig=0 th=0    ig=0 th=0                  ig=0 th=0     ig=1 th=0
#cc5    ig=0 th=0    ig=2 th=2.56               ig=0 th=2.56
#mm25   ig=0 th=.08                             ig=0 th=.08  ig=1 th=.01
#mc25   ig=2 th=.04  ig=6 th=.04                ig=5 th=0
#cm25   ig=8 th=.16  ig=9 th=.32   ig=7 th=.02  ig=8 th=.08
#cc25   ig=0 th=.08  ig=10 th=.16               ig=2 th=.08
#mm100               ig=0 th=0
#mc100               ig=5 th=.16
#cm100               ig=9 th=.32
#cc100               ig=2 th=1.28

