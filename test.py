from q_inf import aggregate, test, quality_inf, catch, optimize, test, highest, avg, read
import numpy as np

# experiment parameters
experiment = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']
tests = [['help', 'neg', 'inc']]#, ['help'], ['neg'], ['inc']]
datasets = ['mnist', 'cifar']
models = ['mlp', 'cnn']
option = 'count'
treshold = 0
participants = 5
Niteration = 250
bestof = 5
ignorelast = 0
ignorefirst = 0

'''
# checking multiple treshold
print(optimize(['neg'], experiment, models, datasets, participants, option, ignorefirst, ignorelast))
print(optimize(['inc'], experiment, models, datasets, participants, option, ignorefirst, ignorelast))
print(optimize(['help'], experiment, models, datasets, participants, option, ignorefirst, ignorelast))
print(optimize(['neg', 'inc'], experiment, models, datasets, participants, option, ignorefirst, ignorelast))
print(optimize(['neg', 'help'], experiment, models, datasets, participants, option, ignorefirst, ignorelast))
print(optimize(['inc', 'help'], experiment, models, datasets, participants, option, ignorefirst, ignorelast))
print(optimize(['neg', 'inc', 'help'], experiment, models, datasets, participants, option, ignorefirst, ignorelast))
'''

'''
# check multiple ingorefirst
for ign in [0, 2, 4, 6, 8, 10]:
    ignorefirst = ign
    print(optimize(['neg', 'inc', 'help'], experiment, models, datasets, participants, 'count', ignorefirst, ignorelast))
    print(optimize(['neg', 'inc', 'help'], experiment, models, datasets, participants, 'value', ignorefirst, ignorelast))
'''

'''
# determine the best test
success = catch(tests, len(experiment), models, datasets, option, ignorefirst, ignorelast, treshold, bestof)
att = [(success['a'][0][0][i]+success['a'][0][1][i]+success['a'][1][0][i]+success['a'][1][1][i])/4 for i in range(len(tests))]
fre = [(success['f'][0][0][i]+success['f'][0][1][i]+success['f'][1][0][i]+success['f'][1][1][i])/4 for i in range(len(tests))]
mni = [(success['a'][0][0][i]+success['f'][0][0][i]+
        success['a'][1][0][i]+success['f'][1][0][i])/4 for i in range(len(tests))]
cif = [(success['a'][0][1][i]+success['f'][0][1][i]+
        success['a'][1][1][i]+success['f'][1][1][i])/4 for i in range(len(tests))]
mlp = [(success['a'][0][0][i]+success['f'][0][0][i]+success['a'][0][1][i]+
        success['f'][0][1][i])/4 for i in range(len(tests))]
cnn = [(success['a'][1][0][i]+success['f'][1][0][i]+success['a'][1][1][i]+
        +success['f'][1][1][i])/4 for i in range(len(tests))]
mind = [(success['a'][0][0][i]+success['a'][0][1][i]+success['a'][1][0][i]+success['a'][1][1][i]+
         success['f'][0][0][i]+success['f'][0][1][i]+success['f'][1][0][i]+success['f'][1][1][i])/8 for i in range(len(tests))]
print(att.index(max(att)), fre.index(max(fre))),
      mni.index(max(mni)), cif.index(max(cif)), mlp.index(max(mlp)), cnn.index(max(cnn)), mind.index(max(mind)))
'''

#base = aggregate(experiment, participants, option, ignorefirst, ignorelast, treshold)
#success = catch(tests, len(experiment), models, datasets, option, ignorefirst, ignorelast, treshold, bestof)
#high = highest(tests, len(experiment), models, datasets, option, ignorefirst, ignorelast, treshold)
#av = avg(tests, len(experiment), models, datasets, option, ignorefirst, ignorelast, treshold)
#linscore = quality_inf(['neg', 'help', 'inc'], len(experiment),  models, datasets, participants, option, ignorefirst, ignorelast, treshold)

start = np.zeros([10])
improvement = np.zeros([10, Niteration])
weights = np.zeros([10, participants])

start[0], deviants, contributors, improvement[0], weights[0] = read('110l')
start[1], deviants, contributors, improvement[1], weights[1] = read('111l')
start[2], deviants, contributors, improvement[2], weights[2] = read('112l')
start[3], deviants, contributors, improvement[3], weights[3] = read('113l')
start[4], deviants, contributors, improvement[4], weights[4] = read('114l')
start[5], deviants, contributors, improvement[5], weights[5] = read('115l')
start[6], deviants, contributors, improvement[6], weights[6] = read('116l')
start[7], deviants, contributors, improvement[7], weights[7] = read('117l')
start[8], deviants, contributors, improvement[8], weights[8] = read('118l')
start[9], deviants, contributors, improvement[9], weights[9] = read('119l')

accuracy = [np.mean(start)]
for i in range(1, Niteration+1):
    accuracy.append(accuracy[i-1]+np.mean(improvement, axis=0)[i-1])

print(accuracy, np.mean(weights, axis=0))
