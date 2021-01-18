import numpy as np


# OLD version of reading the result file
def readOLD(participants, instance):
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
        par = '5/5_2_1/'
    elif participants == '025':
        par = '25/25_5_1/'
    elif participants == '100':
        par = '100/100_10_1/'

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


# NEW version of reading the result file
def readNEW(instance):
    start = np.load('../save/' + instance + '/' + 'start.npy')
    #deviant = np.load('../save/' + instance + '/' + 'deviant.npy')
    weight = np.load('../save/' + instance + '/' + 'weight.npy')
    contributor = np.load('../save/' + instance + '/' + 'contributor.npy').astype(int)
    test_imp = np.load('../save/' + instance + '/' + 'test.npy')
    train_imp = np.load('../save/' + instance + '/' + 'train.npy')
    return start, weight, contributor, test_imp, train_imp


# looking for the deviants with multiple tests
def test(what, how, option, ignorefirst, ignorelast, treshold):
    # ingorelast - skipping the last iterations
    # treshold - ignoring changes below
    # option - count or actual

    if 'mnist' in what or 'cifar' in what:
        start, weight, contributors, improvements, train_imp = readNEW(what)
    else:
        start, deviants, contributors, improvements, weight = readOLD(what[:3], what[3:])

    score = np.zeros(np.amax(contributors) + 1)
    if 'neg' in how:  # if big negative, than -
        for round in range(ignorefirst, len(improvements) - ignorelast):
            if improvements[round] < -treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        score[contributors[round][contributor]] = \
                            score[contributors[round][contributor]] - 1
                    else:
                        score[contributors[round][contributor]] = \
                            score[contributors[round][contributor]] + improvements[round]

    if 'inc' in how:  # if big improvement next than -
        for round in range(ignorefirst, len(improvements) - 1 - ignorelast):
            if improvements[round] < improvements[round + 1] - treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        score[contributors[round][contributor]] = \
                            score[contributors[round][contributor]] - 1
                    else:
                        score[contributors[round][contributor]] = \
                            score[contributors[round][contributor]] + improvements[round] - improvements[round + 1]

    if 'help' in how:  # if big improvement compared to last, than +
        for round in range(ignorefirst, len(improvements) - ignorelast):
            if improvements[round] > improvements[round - 1] + treshold:
                for contributor in range(len(contributors[0])):
                    if option == 'count':
                        score[contributors[round][contributor]] = \
                            score[contributors[round][contributor]] + 1
                    else:
                        score[contributors[round][contributor]] = \
                            score[contributors[round][contributor]] + improvements[round] - improvements[round - 1]

    return score


# determine the success rate of identifying the cheater with different test combinations
def catch(tests, instances, models, Dsets, option, ignorefirst, ignorelast, treshold, number):
    # number - check worst x

    # success[NoiseType][ModelID][DatasetID][TestCase][0/1] (0: successful case, 1: all case)
    success = {'a': np.zeros([len(models), len(Dsets), len(tests), 2]),
               'f': np.zeros([len(models), len(Dsets), len(tests), 2])}
    final = {'a': np.zeros([len(models), len(Dsets), len(tests)]), 'f': np.zeros([len(models), len(Dsets), len(tests)])}

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
