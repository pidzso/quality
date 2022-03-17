import os
from scipy import stats
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
rcParams.update({'figure.autolayout': True})

from q_inf import test, position, read, QI_gen, dist, cheat_score


def accuracy_boxplot(model, data, participants, noise_type, noise_size, weight, rounds, instance_num):
    '''
    creates boxplot of round-wise accuracies when 'linear'
    creates round-wise accuracies based on test and train when 'no'
    '''

    test = np.zeros((instance_num, rounds + 1))
    path = '../save/' + model + '_' + data + '_' + str(participants) + '/' + noise_type + '_' + str(noise_size) + '_' + str(weight) + '/'

    for i in np.arange(instance_num):
        test[i] = np.cumsum(np.insert(np.load(path + str(i + 1) + '/test.npy'), 0, np.load(path + str(i + 1) + '/start.npy')))

    mins = np.transpose(test).min(axis=1)
    maxes = np.transpose(test).max(axis=1)
    means = np.transpose(test).mean(axis=1)
    std = np.transpose(test).std(axis=1)

    if noise_type == 'linear':
        plt.errorbar(np.arange(rounds + 1), means, std, fmt='ok', lw=3)
        plt.errorbar(np.arange(rounds + 1), means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)
        plt.xlim(-1, rounds)
    elif noise_type == 'no':
        avg_te = np.average(test, axis=0)
        plt.plot(np.arange(rounds + 1), avg_te, 'b')

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
    plt.savefig('../save/weight_' + model[0] + data[0] + str(participants) + '(1).png')
    plt.close()


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


def score_change(model, data, participants, rounds, instance):
    '''
    creates round-wise the participants' score changes for Shapley experiments
    '''

    scores = {'QI': np.zeros((instance, rounds + 1, participants)),
              'LO': np.zeros((instance, rounds + 1, participants)),
              'DS': np.zeros((instance, rounds + 1, participants))}

    path = os.path.abspath('..') + '\\save\\Shapley\\' + model + '_' + data + '_' + str(participants) +'\\' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '\\'

    # create figs for Shapley
    for i in np.arange(instance):
        start, deviant, scores['LO'][i][1:], scores['DS'][i][1:], scores['QI'][i][1:], weight, contributor, test_imp = read(path + '780' + str(i))

    AVGscores = {'QI': np.transpose(np.average(scores['QI'], axis=0)),
                 'LO': np.transpose(np.average(scores['LO'], axis=0)),
                 'DS': np.transpose(np.average(scores['DS'], axis=0))}

    for key in AVGscores.keys():
        for i in np.arange(participants):
            plt.plot(np.arange(rounds+1), AVGscores[key][i], color=str(i / participants))

        plt.title(model + '_' + data + '_' + str(participants), fontsize=20)
        plt.xlabel('Rounds', fontsize=20)
        plt.ylabel('Scores_' + str(key), fontsize=20)
        plt.savefig(path + model[0] + data[0] + str(participants) + str(key) + '.png')
        plt.close()


def score(model, data, participants, rounds, instance):
    '''
    creates boxplot of participant-wise quality scores for Shapley experiment
    '''

    scores = {'QI': np.zeros((instance, rounds, participants)),
              'LO': np.zeros((instance, rounds, participants)),
              'DS': np.zeros((instance, rounds, participants))}

    path = os.path.abspath('..') + '\\save\\Shapley\\' + model + '_' + data + '_' + str(participants) +'\\' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '\\'
    for i in np.arange(instance):
        start, deviant, scores['LO'][i], scores['DS'][i], scores['QI'][i], weight, contributor, test_imp = read(path + '780' + str(i))

    AVGscores = {'QI': np.transpose(np.average(scores['QI'], axis=1)),
                 'LO': np.transpose(np.average(scores['LO'], axis=1)),
                 'DS': np.transpose(np.average(scores['DS'], axis=1))}

    for key in scores.keys():
        mins  = np.min( AVGscores[key], axis=1)
        maxes = np.max( AVGscores[key], axis=1)
        means = np.mean(AVGscores[key], axis=1)
        std   = np.std( AVGscores[key], axis=1)

        plt.errorbar(np.arange(participants), means, std, fmt='ok', lw=3)
        plt.errorbar(np.arange(participants), means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)
        plt.xlim(-1, participants)
        plt.title(model + '_' + data + '_' + str(participants) + '_' + str(key), fontsize=20)
        plt.xlabel('Participant', fontsize=20)
        plt.ylabel('Score_' + str(key), fontsize=20)
        plt.savefig(path + 'score_' + model[0] + data[0] + str(participants) + str(key) + '.png')
        plt.close()


def order(model, data, participants, instance, rounds, aggregate=''):
    '''
    calculates the quality inference's accuracy instance-wise
    for both Shapley and RobustRand
    '''

    #path = os.path.abspath('..') + '\\save\\Shapley\\' + model + '_' + data + '_' + str(participants) +'\\' + 'linear' + '_' + str(1.0) + '_' + str(0.0) + '\\'
    path = os.path.abspath('..') + '\\save\\RobustRand\\' + model + '_' + data + '_' + str(participants) +'\\' + 'no' + '_' + str(0.0) + '_' + str(0.0) + aggregate + '\\'

    scores = {'QI': np.zeros((instance, rounds, participants)),
              'LO': np.zeros((instance, rounds, participants)),
              'DS': np.zeros((instance, rounds, participants))}
    sp = {'QI': np.zeros((instance, rounds)),
          'LO': np.zeros((instance, rounds)),
          'DS': np.zeros((instance, rounds))}

    for i in np.arange(instance):
        start, deviant, tmp1, tmp2, scores['QI'][i], weight, contributor, test_imp = read(path + str(i + 1))
        #start, deviant, scores['LO'][i], scores['DS'][i], scores['QI'][i], weight, contributor, test_imp = read(path + '780' + str(i))

    for key in scores.keys():
        for i in np.arange(instance):
            for r in np.arange(rounds):
                sp[key][i][r] = stats.spearmanr(np.arange(participants), scores[key][i][r])[0]

    #plt.plot(np.arange(rounds), np.average(sp['QI'], axis=0), color='red',   label='QI')
    #plt.plot(np.arange(rounds), np.average(sp['LO'], axis=0), color='blue',  label='LO')
    #plt.plot(np.arange(rounds), np.average(sp['DS'], axis=0), color='green', label='DS')

    #plt.legend()
    #plt.ylim(0, 1)
    #plt.title('order_' + model + '_' + data + '_' + str(participants), fontsize=20)
    #plt.xlabel('Rounds', fontsize=20)
    #plt.ylabel('Spearman', fontsize=20)
    #plt.savefig(path + 'order_' + model[0] + data[0] + str(participants) + '.png')
    #plt.close()

    return sp

def generate_RobustRand_figs():
    '''
    generate average QI values for RobustRand honest and cheating participants
    '''

    for da in ['mnist', 'cifar']:
        for mo in ['mlp', 'cnn']:
            for de in ['attack', 'freeride']:
                for num in [1.0, 3.0, 7.0, 15.0]:
                    s0  = cheat_score(mo, da, 32, 3, 100, de, num, 'avg', 0)
                    s3  = cheat_score(mo, da, 32, 3, 100, de, num, 'tm', 3)
                    s7  = cheat_score(mo, da, 32, 3, 100, de, num, 'tm', 7)
                    s15 = cheat_score(mo, da, 32, 3, 100, de, num, 'tm', 15)

                    dev = [s0[0], s3[0], s7[0], s15[0]]
                    hon = [s0[1], s3[1], s7[1], s15[1]]

                    plt.plot(np.arange(4), dev, color='red',   label='Deviant')
                    plt.plot(np.arange(4), hon, color='green',  label='Honest')

                    plt.legend()
                    plt.title(mo + '_' + da + '_' + de + '_' + str(num), fontsize=20)
                    plt.xlabel('Trimming Size [0, 3, 7, 15]', fontsize=20)
                    plt.ylabel('QI Scores', fontsize=20)
                    plt.savefig(os.path.abspath('..') + '\\save\\RobustRand\\' + mo + '_' + da + '_' + de + '_' + str(num) + '.png')
                    plt.close()

def generate_Shapley_figs():
    '''
    generate QI vs LO values for Shapley
    '''

    mean  = np.zeros(16)
    maxes = np.zeros(16)
    mins  = np.zeros(16)
    std   = np.zeros(16)
    i = 0
    fig, ax = plt.subplots()
    plt.xlim(-1, 16)
    plt.ylim(0, 1)
    plt.title('TrimMean', fontsize=20)
    plt.title('QI vs LOO', fontsize=20)
    plt.ylabel('Spearman', fontsize=20)
    for p in [5, 25]:
        for d in ['mnist', 'cifar']:
            for m in ['mlp', 'cnn']:
                for key in ['QI', 'LO']:
                    tmp = order(m, d, p, 10, 100)
                    mean[i]  = np.mean(np.transpose(tmp[key])[99])
                    maxes[i] = np.max( np.transpose(tmp[key])[99])
                    mins[i]  = np.min( np.transpose(tmp[key])[99])
                    std[i]   = np.std( np.transpose(tmp[key])[99])
                    i = i + 1

    plt.errorbar(np.arange(16), mean, std, fmt='ok', lw=3)
    plt.errorbar(np.arange(16), mean, [mean - mins, maxes - mean], fmt='.k', ecolor='gray', lw=1)
    plt.savefig(os.path.abspath('..') + '\\save\\Shapley\\QI-LO.png')
    plt.close()


def pos(participants):
    '''
    shows the positions of cheaters instance-wise for all usecase
    '''

    p = position(['neg', 'inc', 'help'], ['mlp', 'cnn'], ['mnist', 'cifar'], participants, 0, 0, 0, 'count')
    mins  = [np.min(p['a'], axis=2).flatten(), np.min(p['f'], axis=2).flatten()]
    maxes = [np.max(p['a'], axis=2).flatten(), np.max(p['f'], axis=2).flatten()]
    means = [np.mean(p['a'], axis=2).flatten(), np.mean(p['f'], axis=2).flatten()]
    std   = [np.std(p['a'], axis=2).flatten(), np.std(p['f'], axis=2).flatten()]

    for i in [0, 1]:
        if i == 0:
            s = 'Attacker'
        elif i == 1:
            s = 'Freerider'
        fig, ax = plt.subplots()
        plt.errorbar(np.arange(4), means[i], std[i], fmt='ok', lw=3)
        plt.errorbar(np.arange(4), means[i], [means[i] - mins[i], maxes[i] - means[i]], fmt='.k', ecolor='gray', lw=1)
        plt.xlim(-1, 4)
        plt.ylim(0, int(participants) - 1)
        plt.title(s + '\'s Position', fontsize=20)
        plt.ylabel('Position', fontsize=20)
        ax.set_xticklabels(['', 'MM' + participants, 'MC' + participants, 'CM' + participants, 'CC' + participants, ''], rotation=45, fontsize=20)
        plt.savefig('../save/' + participants + s + '.png')
        plt.close()


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
        mm100 = order('mlp', 'mnist', 100, 9, 0, 0, 0, 'count', tests[i])
        mc5   = order('mlp', 'cifar', 5, 9, 0, 0, 0, 'count', tests[i])
        mc25  = order('mlp', 'cifar', 25, 9, 0, 0, 0, 'count', tests[i])
        mc100 = order('mlp', 'cifar', 100, 9, 0, 0, 0, 'count', tests[i])
        cm5   = order('cnn', 'mnist', 5, 9, 0, 0, 0, 'count', tests[i])
        cm25  = order('cnn', 'mnist', 25, 9, 0, 0, 0, 'count', tests[i])
        cm100 = order('cnn', 'mnist', 100, 9, 0, 0, 0, 'count', tests[i])
        cc5   = order('cnn', 'cifar', 5, 9, 0, 0, 0, 'count', tests[i])
        cc25  = order('cnn', 'cifar', 25, 9, 0, 0, 0, 'count', tests[i])
        cc100 = order('cnn', 'cifar', 100, 9, 0, 0, 0, 'count', tests[i])

        OPTmm5   = order('mlp', 'mnist', 5, 9, 1, 0, 0.16, 'count', tests[i])
        OPTmm25  = order('mlp', 'mnist', 25, 9, 1, 0, 0.01, 'value', tests[i])
        OPTmm100 = order('mlp', 'mnist', 100, 9, 1, 0, 0.04, 'value', tests[i])
        OPTmc5   = order('mlp', 'cifar', 5, 9, 1, 0, 0, 'value', tests[i])
        OPTmc25  = order('mlp', 'cifar', 25, 9, 2, 0, 0.16, 'count', tests[i])
        OPTmc100 = order('mlp', 'cifar', 100, 9, 1, 0, 0.04, 'value', tests[i])
        OPTcm5   = order('cnn', 'mnist', 5, 9, 1, 0, 0, 'value', tests[i])
        OPTcm25  = order('cnn', 'mnist', 25, 9, 8, 0, 0.32, 'count', tests[i])
        OPTcm100 = order('cnn', 'mnist', 100, 9, 0, 0, 0.16, 'count', tests[i])
        OPTcc5   = order('cnn', 'cifar', 5, 9, 0, 0, 0.08, 'count', tests[i])
        OPTcc25  = order('cnn', 'cifar', 25, 9, 2, 0, 0.64, 'count', tests[i])
        OPTcc100 = order('cnn', 'cifar', 100, 9, 7, 0, 0.32, 'count', tests[i])

        mins['5' + str(i)]    = np.min([mm5, mc5, cm5, cc5], axis=1)
        mins['25' + str(i)]   = np.min([mm25, mc25, cm25, cc25], axis=1)
        mins['100' + str(i)]  = np.min([mm100, mc100, cm100, cc100], axis=1)
        maxes['5' + str(i)]   = np.max([mm5, mc5, cm5, cc5], axis=1)
        maxes['25' + str(i)]  = np.max([mm25, mc25, cm25, cc25], axis=1)
        maxes['100' + str(i)] = np.max([mm100, mc100, cm100, cc100], axis=1)
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

    # mean and variance of scoring rules among each other
    #print(np.argsort([means['50'][0], means['51'][0], means['52'][0], means['53'][0], means['54'][0], means['55'][0], means['56'][0]]))
    #print(np.argsort([means['50'][1], means['51'][1], means['52'][1], means['53'][1], means['54'][1], means['55'][1], means['56'][1]]))
    #print(np.argsort([means['50'][2], means['51'][2], means['52'][2], means['53'][2], means['54'][2], means['55'][2], means['56'][2]]))
    #print(np.argsort([means['50'][3], means['51'][3], means['52'][3], means['53'][3], means['54'][3], means['55'][3], means['56'][3]]))

    #print(np.argsort([means['250'][0], means['251'][0], means['252'][0], means['253'][0], means['254'][0], means['255'][0], means['256'][0]]))
    #print(np.argsort([means['250'][1], means['251'][1], means['252'][1], means['253'][1], means['254'][1], means['255'][1], means['256'][1]]))
    #print(np.argsort([means['250'][2], means['251'][2], means['252'][2], means['253'][2], means['254'][2], means['255'][2], means['256'][2]]))
    #print(np.argsort([means['250'][3], means['251'][3], means['252'][3], means['253'][3], means['254'][3], means['255'][3], means['256'][3]]))

    #print(np.argsort([means['1000'][0], means['1001'][0], means['1002'][0], means['1003'][0], means['1004'][0], means['1005'][0], means['1006'][0]]))
    #print(np.argsort([means['1000'][1], means['1001'][1], means['1002'][1], means['1003'][1], means['1004'][1], means['1005'][1], means['1006'][1]]))
    #print(np.argsort([means['1000'][2], means['1001'][2], means['1002'][2], means['1003'][2], means['1004'][2], means['1005'][2], means['1006'][2]]))
    #print(np.argsort([means['1000'][3], means['1001'][3], means['1002'][3], means['1003'][3], means['1004'][3], means['1005'][3], means['1006'][3]]))

    for par in participants:
        fig, ax = plt.subplots()
        for i in np.arange(len(tests)):
            plt.errorbar(np.arange(i, 4*len(tests), len(tests)), means[par + str(i)], std[par + str(i)], fmt='ok', lw=3) #, ecolor=['red', 'green', 'blue', 'brown'])
            plt.errorbar(np.arange(i, 4*len(tests), len(tests)), means[par + str(i)], [means[par + str(i)] - mins[par + str(i)], maxes[par + str(i)] - means[par + str(i)]], fmt='.k', lw=1, ecolor='gray')  #ecolor=['red', 'green', 'blue', 'brown'])
        plt.xlim(-1, 4*len(tests))
        plt.ylim(0, 1)
        plt.title(par, fontsize=20)
        plt.ylabel('QI Accuracy', fontsize=20)
        ax.set_xticklabels(['', 'MM' + par, 'MC' + par, 'CM' + par, 'CC' + par, ''], rotation=45, fontsize=20)
        plt.savefig('../save/qi_' + par + '.png')
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
    plt.ylim(0, 1)
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
#mm100 ig=0 th=0     ig=0 th=.32   ig=8 th=.32  ig=1 th=.32   ig=1 th=.32   ig=0 th=.04   ig=1 th=0
#mc100 ig=5 th=.01   ig=3 th=.16   ig=8 th=.04  ig=6 th=.08   ig=8 th=.01   ig=3 th=.08   ig=6 th=.64
#cm100 ig=0 th=.16   ig=9 th=.32   ig=0 th=.02  ig=9 th=.16   ig=8 th=.16   ig=9 th=.32   ig=0 th=.16
#cc100 ig=8 th=.64   ig=2 th=1.28  ig=6 th=.32  ig=7 th=0     ig=7 th=.04   ig=3 th=1.28  ig=7 th=0.32

#VALUE  neg           inc           inchelp      neginc       neginchelp
#mc5    ig=0 th=.64  ig=0 th=.64   ig=1 th=0    ig=0 th=.64   ig=1 th=0
#cm5    ig=0 th=0    ig=0 th=0                  ig=0 th=0     ig=1 th=0
#cc5    ig=0 th=0    ig=2 th=2.56               ig=0 th=2.56
#mm25   ig=0 th=.08                             ig=0 th=.08   ig=1 th=.01
#mc25   ig=2 th=.04  ig=6 th=.04                ig=5 th=0
#cm25   ig=8 th=.16  ig=9 th=.32   ig=7 th=.02  ig=8 th=.08
#cc25   ig=0 th=.08  ig=10 th=.16               ig=2 th=.08
#mm100               ig=0 th=0                               ig=1 th=.04
#mc100               ig=5 th=.16                             ig=6 th=.04
#cm100               ig=9 th=.32
#cc100               ig=2 th=1.28

