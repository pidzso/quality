import numpy as np


def read(instance):
    # reading the file with the following format:
    '''
        Starting:
            float
        Deviants:
            list of ints
        Contributors:
             list of list
        Improvements:
            list
    '''
    deviants = []
    contributors = []
    improvements = []
    weights = []
    aux = []
    with open("../save/result" + instance + ".txt", 'r') as f:
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


def test(what, how, option, ignorefirst, ignorelast, treshold):
    # ingorelast - skipping the last iterations
    # treshold - ignoring changes below
    # option - count or actual
    # looking for the deviants with multiple tests
    starting, deviants, contributors, improvements = read(what)
    score = np.zeros(1 + max([max(sublist) for sublist in contributors]))
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
    return deviants, score


def aggregate(experiments, ppl, option, ignorefirst, ignorelast, treshold):
    # average the experiment results (for baseline)
    tests = ['neg', 'inc', 'help']
    scores = {'neg': np.zeros(ppl), 'inc': np.zeros(ppl), 'help': np.zeros(ppl)}
    stat = {'neg': [0, 0], 'inc': [0, 0], 'help': [0, 0]}
    for ex in experiments:
        for t in tests:
            scores[t] = test(ex, t, option, ignorefirst, ignorelast, treshold)[1]
            stat[t][0] = stat[t][0] + np.mean(scores[t]) / len(experiments)
            stat[t][1] = stat[t][1] + np.var(scores[t]) / len(experiments)
    return stat


def catch(tests, instances, models, Dsets, option, ignorefirst, ignorelast, treshold, number):
    # determine the success rate of identifying the cheater with different test combinations
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


def highest(tests, instances, models, Dsets, option, ignorefirst, ignorelast, treshold):
    # determine the highest position of a cheater

    # final[NoiseType][ModelID][DatasetID][TestCase]
    final = {'a': np.zeros([len(models), len(Dsets), len(tests)]), 'f': np.zeros([len(models), len(Dsets), len(tests)])}

    for model in range(len(models)):  # 0:mlp, 1:cnn
        for dataset in range(len(Dsets)):  # 0:mnist, 1:cifar
            for type in ['a', 'f']:  # attack/freeride
                files = [str(model) + str(dataset) + str(instance) + type for instance in range(instances)]
                for file in range(len(files)):
                    for test_iter in range(len(tests)):
                        dev, score = test(files[file], tests[test_iter], option, ignorefirst, ignorelast, treshold)
                        aux = score[0]
                        for dev_iter in dev:
                            if aux < score[dev_iter]:
                                aux = score[dev_iter]
                        score.sort()
                        final[type][model][dataset] = final[type][model][dataset] + np.max(np.where(score == aux)) / len(files)
    return final


def avg(tests, instances, models, Dsets, option, ignorefirst, ignorelast, treshold):
    # determine the average position of the cheaters

    # final[NoiseType][ModelID][DatasetID][TestCase]
    final = {'a': np.zeros([len(models), len(Dsets), len(tests)]), 'f': np.zeros([len(models), len(Dsets), len(tests)])}

    for model in range(len(models)):  # 0:mlp, 1:cnn
        for dataset in range(len(Dsets)):  # 0:mnist, 1:cifar
            for type in ['a', 'f']:  # attack/freeride
                files = [str(model) + str(dataset) + str(instance) + type for instance in range(instances)]
                for file in range(len(files)):
                    for test_iter in range(len(tests)):
                        dev, score = test(files[file], tests[test_iter], option, ignorefirst, ignorelast, treshold)
                        aux = 0
                        for dev_iter in dev:
                            aux = aux + score[dev_iter] / len(dev)
                        for s in score:
                            if s <= aux:
                                final[type][model][dataset] = final[type][model][dataset] + 1 / len(files)
    return final  # remove 1 as it counted the deviant itself


def quality_inf(tests, instances, models, Dsets, ppl, option, ignorefirst, ignorelast, treshold):
    # linear case

    # linscore[model][dataset][participant]
    linscore = np.zeros([len(models), len(Dsets), ppl])

    for model in range(len(models)):  # 0:mlp, 1:cnn
        for dataset in range(len(Dsets)):  # 0:mnist, 1:cifar
            files = [str(model) + str(dataset) + str(instance) + 'l' for instance in range(instances)]
            for file in range(len(files)):
                for test_iter in range(len(tests)):
                    linscore[model][dataset] = [linscore[model][dataset][i] +
                                                test(files[file], tests[test_iter], option, ignorefirst, ignorelast, treshold)[1][i] \
                                                for i in range(ppl)]
    return linscore


def optimize(test, experiment, models, datasets, participants, option, ignorefirst, ignorelast):
    # determine the best treshold
    badpoint = []
    tresholds = [0.0, 0.05, 0.10, 0.15, 0.2, 0.25]
    for th in range(len(tresholds)):
        badpoint.append([0, 0, 0, 0, 0])
        linscore = quality_inf(test, len(experiment), models, datasets, participants, \
                               option, ignorefirst, ignorelast, tresholds[th])
        aux0 = np.argsort(linscore[0][0])
        aux1 = np.argsort(linscore[0][1])
        aux2 = np.argsort(linscore[1][0])
        aux3 = np.argsort(linscore[1][1])
        for par in range(participants):
            badpoint[th][0] = badpoint[th][0] + abs(par - aux0[par])
            badpoint[th][1] = badpoint[th][1] + abs(par - aux1[par])
            badpoint[th][2] = badpoint[th][2] + abs(par - aux2[par])
            badpoint[th][3] = badpoint[th][3] + abs(par - aux3[par])
            # dataset/model -wise average
            badpoint[th][4] = np.mean([badpoint[th][ind] for ind in [0, 1, 2, 3]])

    return badpoint
