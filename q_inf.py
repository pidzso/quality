import numpy as np
import os


# reading the result file
def read(instance):
    if instance[-1] not in '/\\':
        instance = instance + '\\'
    start       = np.load(instance + 'start.npy')
    deviant     = np.load(instance + 'deviant.npy')
    weight      = np.load(instance + 'weight.npy')
    contributor = np.load(instance + 'contributor.npy').astype(int)
    test_imp    = np.load(instance + 'test.npy')

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

    score = np.zeros([(len(test_imp) - ignorelast - ignorefirst), np.amax(contributors) + 1])
    if 'neg' in how:  # if big negative, than -
        for round in range(ignorefirst, len(test_imp) - ignorelast):
            if test_imp[round] < -treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        for r in range(round, len(test_imp) - ignorelast):
                            score[r][contributors[round][contributor]] = score[r][contributors[round][contributor]] - 1
                    else:
                        for r in range(round, len(test_imp) - ignorelast):
                            score[r][contributors[round][contributor]] = score[r][contributors[round][contributor]] + test_imp[round]

    if 'inc' in how:  # if big improvement next than -
        for round in range(ignorefirst, len(test_imp) - 1 - ignorelast):
            if test_imp[round] < test_imp[round + 1] - treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        for r in range(round, len(test_imp) - ignorelast):
                            score[r][contributors[round][contributor]] = score[r][contributors[round][contributor]] - 1
                    else:
                        for r in range(round, len(test_imp) - ignorelast):
                            score[r][contributors[round][contributor]] = score[r][contributors[round][contributor]] + test_imp[round] - test_imp[round + 1]

    if 'help' in how:  # if big improvement compared to last, than +
        for round in range(ignorefirst, len(test_imp) - ignorelast):
            if test_imp[round] > test_imp[round - 1] + treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        for r in range(round, len(test_imp) - ignorelast):
                            score[r][contributors[round][contributor]] = score[r][contributors[round][contributor]] + 1
                    else:
                        for r in range(round, len(test_imp) - ignorelast):
                            score[r][contributors[round][contributor]] = score[r][contributors[round][contributor]] + test_imp[round] - test_imp[round - 1]

    # save QI scores
    with open(what + '/qi.npy', 'wb') as f:
        np.save(f, score)

    return score

#for i in range(1, 10):
#    test(os.path.abspath('..') + '\\save\\mlp_mnist_5\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\mlp_cifar_5\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\cnn_mnist_5\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\cnn_cifar_5\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\mlp_mnist_25\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\mlp_cifar_25\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\cnn_mnist_25\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\cnn_cifar_25\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\mlp_mnist_100\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\mlp_cifar_100\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\cnn_mnist_100\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)
#    test(os.path.abspath('..') + '\\save\\cnn_cifar_100\\linear_1.0_0.0\\' + str(i), ['neg', 'inc', 'help'], 'count', 0, 0, 0)



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


def QI_gen(model, data, participants, rounds, instance, exp_type, exp_size, weight, agr_type, rob_size, source):
    '''
    OBSOLETE: functionality is built in the federated run
    generating QI scores participant-wise and instance-wise
    '''

    scores = np.zeros((instance, rounds + 1, participants))
    if source == 'RobustRand':
        path = os.path.abspath('..') + '\\save\\' + source + '\\' + model + '_' + data + '_' + str(participants) + \
               '\\' + exp_type + '_' + str(exp_size) + '_' + str(weight) + '_' + agr_type + str(rob_size) + '\\'
    elif source == 'Shapley':
        path = os.path.abspath('..') + '\\save\\' + source + '\\' + model + '_' + data + '_' + str(participants) + \
               '\\' + exp_type + '_' + str(exp_size) + '_' + str(weight) + '\\'
    else:
        path = os.path.abspath('..') + '\\save\\' + model + '_' + data + '_' + str(participants) + \
               '\\' + exp_type + '_' + str(exp_size) + '_' + str(weight) + '\\'

    for i in np.arange(instance):
        if source == 'Shapley':
            tmp = path + '780' + str(i)  # specific seed used for Shapley experiments
        else:
            tmp = path + str(i)  # str(i+1) if there is no 0 instance

        for r in np.arange(rounds):
            scores[i][r + 1] = test(tmp, ['neg', 'inc', 'help'], 'count', 0, rounds - 1 - r, 0)
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
        start, deviant, alma, korte, scores[i], weight, contributor, test_imp = read(path + str(i))  # str(i+1 when there is no 0 instance)

    mins = np.min(scores, axis=0)
    maxes = np.max(scores, axis=0)
    avg = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)

    dev, hon = [0, 0, 0, 0], [0, 0, 0, 0]
    for d in deviant:
        dev[0] = dev[0] + avg[rounds - 1][d] / len(deviant)
        dev[1] = dev[1] + std[rounds - 1][d] / len(deviant)
        dev[2] = dev[2] + mins[rounds - 1][d] / len(deviant)
        dev[3] = dev[3] + maxes[rounds - 1][d] / len(deviant)
    for h in np.setdiff1d(np.arange(participants), deviant):
        hon[0] = hon[0] + avg[rounds - 1][h] / (participants - len(deviant))
        hon[1] = hon[1] + std[rounds - 1][h] / (participants - len(deviant))
        hon[2] = hon[2] + mins[rounds - 1][h] / (participants - len(deviant))
        hon[3] = hon[3] + maxes[rounds - 1][h] / (participants - len(deviant))

    return dev, hon


# read misbehaviour experiments
def read_OLD(participants, instance, dev_num):
    '''
        Starting:
            float
        Deviants:
            list of ints
        roundWise Contributors:
             list of list
        Improvements:
            list
    '''

    deviants = []
    contributors = []
    improvements = []
    weights = []
    aux = []

    if participants == '005':
        par = '5/5_2_'
    elif participants == '025':
        par = '25/25_5_'
    elif participants == '100':
        par = '100/100_10_'

    par = par + str(dev_num) + '/'

    with open("../save/" + par + "result" + instance + ".txt", 'r') as f:
        for line in f:
            aux.append(line)

    counter = 0  # what data we read currently
    ind = 0  # reading start after [
    round_count = -1  # where to save in contribution
    construct = ''  # create number from characters

    # which part are we reading
    for line in aux:
        if line == 'Starting:\n':
            counter = 1
            continue
        elif line == 'Deviants:\n':
            counter = 2
            continue
        elif line == 'Participants:\n':
            counter = 3
            continue
        elif line == 'Improvements:\n':
            counter = 4
            continue
        elif line == 'Weights:\n':
            counter = 5
            continue

        # read starting accuracy
        if counter == 1:
            starting = float(line.replace('\n', ''))

        # read deviants
        elif counter == 2:
            for word in line:
                if word == '[':
                    ind = 1
                    continue
                if word == ']':
                    ind = 0
                    break
                if ind == 1 and word not in {'', ' ', ',', '\n'}:
                    deviants.append(int(word))

        # read contributors
        elif counter == 3:
            for word in line:
                if word == '[':
                    ind = ind + 1
                    if ind == 2:
                        round_count = round_count + 1
                        contributors.append([])
                    continue
                if word == ']':
                    ind = ind - 1
                    continue
                if ind == 2 and word not in {'', ' ', '\n'}:
                    if word != '.':
                        construct = construct + word
                    else:
                        contributors[round_count].append(int(construct))
                        construct = ''

        # read improvements
        elif counter == 4:
            improvements.append([float(i) for i in line.replace('[', '').replace(']', '').replace('\n', '').split()])

        # read weights
        elif counter == 5:
            weights.append([float(i) for i in line.replace('[', '').replace(']', '').replace('\n', '').split()])

    improvements = [item for sublist in improvements for item in sublist]
    weights = [item for sublist in weights for item in sublist]

    # remove weights for non-weighted experiments
    return starting, deviants, contributors, improvements, weights


# calculates the average score of cheaters vs non-cheaters using OLD
def cheat_score_OLD(par, ins, dev, how, option, ignorefirst, ignorelast, treshold):
    # ingorelast - skipping the last iterations
    # treshold - ignoring changes below
    # option - count or actual

    starting, deviants, contributors, test_imp, weights = read_OLD(par, ins, dev)

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

'''
final = {'a': {'honest': np.zeros([2, 2, 9]), 'cheater': np.zeros([2, 2, 9])},
         'f': {'honest': np.zeros([2, 2, 9]), 'cheater': np.zeros([2, 2, 9])}}
finalAVG = {'a': {'honest': np.zeros([2, 2]), 'cheater': np.zeros([2, 2])},
            'f': {'honest': np.zeros([2, 2]), 'cheater': np.zeros([2, 2])}}

participants = '100'
dev_num = 5
for model in range(2):  # 0:mlp, 1:cnn
    for dataset in range(2):  # 0:mnist, 1:cifar
        for type in ['a', 'f']:  # attack/freeride
            for instance in range(9):
                score = cheat_score_OLD(participants, str(model) + str(dataset) + str(instance) + type, dev_num, ['neg', 'inc', 'help'], 'count', 0, 0, 0)
                final[type]['honest'][model][dataset][instance] = np.mean(score[dev_num:])
                final[type]['cheater'][model][dataset][instance] = np.mean(score[:dev_num])
            finalAVG[type]['honest'][model][dataset] = np.mean(final[type]['honest'][model][dataset], axis=0)
            finalAVG[type]['cheater'][model][dataset] = np.mean(final[type]['cheater'][model][dataset], axis=0)
print(finalAVG)
'''