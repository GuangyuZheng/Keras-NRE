from sklearn.metrics import precision_recall_curve
import numpy as np
import os

path = os.getcwd()
print(path)

f1_dict = {}

filename = ['CNN+ATT', 'Hoffmann', 'MIMLRE', 'Mintz', 'PCNN+ATT']
color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
for i in range(len(filename)):
    precision = np.load(os.path.join(path, 'data', filename[i] + '_precision.npy'))
    recall = np.load(os.path.join(path, 'data', filename[i] + '_recall.npy'))
    max_f1 = 0
    for j in range(len(precision)):
        if precision[j] == 0 or recall[j] == 0:
            continue
        else:
            f1 = 2 * (precision[j] * recall[j]) / (precision[j] + recall[j])
        max_f1 = max(max_f1, f1)
    print(filename[i] + ' ' + str(max_f1))
    f1_dict[filename[i]] = max_f1

# ATTENTION: put the model iters you want to plot into the list
model_iter = [10900]
for one_iter in model_iter:
    y_true = np.load(os.path.join(path, 'data', 'allans.npy'))
    y_scores = np.load(os.path.join(path, 'out', 'sample_allprob_iter_' + str(one_iter) + '.npy'))

    precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    max_f1 = 0
    for j in range(len(precision)):
        if precision[j] == 0 or recall[j] == 0:
            continue
        else:
            f1 = 2 * (precision[j] * recall[j]) / (precision[j] + recall[j])
        max_f1 = max(max_f1, f1)
    print('BGRU+2ATT' + ' ' + str(max_f1))
    f1_dict['BGRU+2ATT'] = max_f1

with open('F1_result.txt', 'w') as f:
    for filename in f1_dict:
        f.write(filename + ' ' + str(f1_dict[filename]) + '\n')
