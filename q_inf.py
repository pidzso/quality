import numpy as np
import os

# reading the result file
def read(instance):
    start       = np.load(instance + '\\' + 'start.npy')
    deviant     = np.load(instance + '\\' + 'deviant.npy')
    weight      = np.load(instance + '\\' + 'weight.npy')
    contributor = np.load(instance + '\\' + 'contributor.npy').astype(int)
    test_imp    = np.load(instance + '\\' + 'test.npy')

    # handling non existing files
    scores = {'qi': [], 'loo': [], 'dshapley': []}
    for score in ['qi', 'loo', 'dshapley']:
        if os.path.exists(instance + '\\' + score + '.npy'):
            scores[score] = np.load(instance + '\\' + score + '.npy')
        else:
            scores[score] = np.array([])
    return start, deviant, scores['loo'], scores['dshapley'], scores['qi'], weight, contributor, test_imp


# computing the QI scores
def test(what, how, option, ignorefirst, ignorelast, treshold):
    # ingorelast - skipping the last iterations
    # treshold - ignoring changes below
    # option - count or actual

    start, deviant, loo, dshapley, qi, weight, contributors, test_imp = read(what)

    score = np.zeros(np.amax(contributors) + 1)
    if 'neg' in how:  # if big negative, than -
        for round in range(ignorefirst, len(test_imp) - ignorelast):
            if test_imp[round] < -treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        score[contributors[round][contributor]] = score[contributors[round][contributor]] - 1
                    else:
                        score[contributors[round][contributor]] = score[contributors[round][contributor]] + test_imp[round]

    if 'inc' in how:  # if big improvement next than -
        for round in range(ignorefirst, len(test_imp) - 1 - ignorelast):
            if test_imp[round] < test_imp[round + 1] - treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        score[contributors[round][contributor]] = score[contributors[round][contributor]] - 1
                    else:
                        score[contributors[round][contributor]] = score[contributors[round][contributor]] + test_imp[round] - test_imp[round + 1]

    if 'help' in how:  # if big improvement compared to last, than +
        for round in range(ignorefirst, len(test_imp) - ignorelast):
            if test_imp[round] > test_imp[round - 1] + treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        score[contributors[round][contributor]] = score[contributors[round][contributor]] + 1
                    else:
                        score[contributors[round][contributor]] = score[contributors[round][contributor]] + test_imp[round] - test_imp[round - 1]
    return score


# determine the success rate of identifying the cheater with different test combinations
def catch(tests, instances, models, Dsets, option, ignorefirst, ignorelast, treshold, number):  # number - check worst x
    # success[NoiseType][ModelID][DatasetID][TestCase][0/1] (0: successful case, 1: all case)
    success = {'a': np.zeros([len(models), len(Dsets), len(tests), 2]),
               'f': np.zeros([len(models), len(Dsets), len(tests), 2])}
    final = {'a': np.zeros([len(models), len(Dsets), len(tests)]),
             'f': np.zeros([len(models), len(Dsets), len(tests)])}

    for model in range(len(models)):  # 0:mlp, 1:cnn
        for dataset in range(len(Dsets)):  # 0:mnist, 1:cifar
            for type in ['a', 'f']:  # attack/freeride
                files = [str(model) + str(dataset) + str(instance) + type for instance in range(instances)]
                for file in range(len(files)):
                    for test_iter in range(len(tests)):
                        dev, score = test(files[file], tests[test_iter], option, ignorefirst, ignorelast, treshold)

                        # select the smallest 'number's
                        aux = list(set(score))
                        aux.sort()
                        find = np.array([])
                        small_iter = 0
                        while len(find) < number:
                            last = np.where(score == aux[small_iter])[0]
                            find = np.append(find, last)
                            small_iter = small_iter + 1

                        # probability that the deviants are in the smallest 'number's
                        # in case of N deviants, one corresponds to 1/N-th of the probability (e.,g. 100% catching all)
                        hit0 = sum([int(find[i] <= max(dev)) for i in range(len(find) - len(last))])
                        hit1 = sum([int(i <= max(dev)) for i in last])
                        if len(find) > number:
                            hit1 = hit1 * sum([int(i <= number) for i in last]) / len(last)
                        success[type][model][dataset][test_iter][0] = success[type][model][dataset][test_iter][0] + hit0 + hit1
                        success[type][model][dataset][test_iter][1] = success[type][model][dataset][test_iter][1] + len(dev)

                final[type][model][dataset] = [round(success[type][model][dataset][i][0] /
                                                     success[type][model][dataset][i][1], 2) for i in range(len(tests))]
    return final


# determine the position of the cheater
def position(tests, models, Dsets, participants, ignorefirst, ignorelast, treshold, option):

    # final[NoiseType][ModelID][DatasetID][9]
    final = {'a': np.zeros([len(models), len(Dsets), 9]), 'f': np.zeros([len(models), len(Dsets), 9])}

    for model in range(len(models)):  # 0:mlp, 1:cnn
        for dataset in range(len(Dsets)):  # 0:mnist, 1:cifar
            for type in ['a', 'f']:  # attack/freeride
                files = [participants + str(model) + str(dataset) + str(instance) + type for instance in range(9)]
                for file in range(len(files)):
                    score = test(files[file], tests, option, ignorefirst, ignorelast, treshold)
                    final[type][model][dataset][file] = sorted(score).index(score[0])
    return final


def QI_gen(model, data, participants, rounds, instance, exp_type, exp_size, agr, source):
    '''
    generating QI scores participant-wise and instance-wise
    '''

    scores = np.zeros((instance, rounds + 1, participants))
    if source == 'RobustRand':
        path = os.path.abspath('..') + '\\save\\' + source + '\\' + model + '_' + data + '_' + str(participants) + '\\' + exp_type + '_' + str(exp_size) + '_' + str(0.0) + '_' + agr + '\\'
    elif source == 'Shapley':
        path = os.path.abspath('..') + '\\save\\' + source + '\\' + model + '_' + data + '_' + str(participants) + '\\' + exp_type + '_' + str(exp_size) + '_' + str(0.0) + '\\'
    else:
        path = os.path.abspath('..') + '\\save\\' + model + '_' + data + '_' + str(participants) + '\\' + exp_type + '_' + str(exp_size) + '_' + str(0.0) + '\\'

    for i in np.arange(instance):
        if source == 'Shapley':
            tmp = path + '780' + str(i)  # specific seed used for Shapley experiments
        else:
            tmp = path + str(i + 1)

        for r in np.arange(rounds):
            scores[i][r + 1] = test(path + str(i + 1), ['neg', 'inc', 'help'], 'count', 0, rounds - 1 - r, 0)
        with open(tmp + '\\qi.npy', 'wb') as f:
            np.save(f, np.array(scores[i][1:]))


def dist(model, data, participants, rounds, instance):
    '''
    RMSE of LO and QI scores
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

    QI = np.mean(AVGscores['QI'], axis=1)
    DS = np.mean(AVGscores['DS'], axis=1)
    LO = np.mean(AVGscores['LO'], axis=1)

    QI = (QI - np.min(QI)) / (np.max(QI) - np.min(QI))
    DS = (DS - np.min(DS)) / (np.max(DS) - np.min(DS))
    LO = (LO - np.min(LO)) / (np.max(LO) - np.min(LO))

    return print(np.sqrt(np.mean((QI-LO)**2)))


def cheat_score(model, data, participants, instance, rounds, dev_type, dev_size, agg_type, agr_size):
    '''
    calculates the average score of cheaters vs non-cheaters
    '''

    path = os.path.abspath('..') + '\\save\\RobustRand\\' + model + '_' + data + '_' + str(participants) +'\\' + dev_type + '_' + str(dev_size) + '_' + str(0.0) + '_' + agg_type + '_' + str(agr_size) + '\\'

    scores = np.zeros((instance, rounds, participants))
    avg_scores = np.zeros((rounds, participants))

    for i in np.arange(instance):
        start, deviant, alma, korte, scores[i], weight, contributor, test_imp = read(path + str(i + 1))

    avg_scores = np.mean(scores, axis=0)

    dev, hon = 0, 0
    for d in deviant:
        dev = dev + avg_scores[rounds - 1][d] / len(deviant)
    for h in np.setdiff1d(np.arange(participants), deviant):
        hon = hon + avg_scores[rounds - 1][h] / (participants - len(deviant))

    return dev, hon
