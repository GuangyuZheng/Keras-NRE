from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


path = os.getcwd()
print(path)

plt.clf()

# ATTENTION: put the model you want to plot
test_epochs = [6]
for epoch in test_epochs:
    if epoch < 10:
        epoch = '0' + str(epoch)
    else:
        epoch = str(epoch)
    filename = ['CNN+ATT', 'Hoffmann', 'MIMLRE', 'Mintz', 'PCNN+ATT']
    color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    for i in range(len(filename)):
        precision = np.load(os.path.join(path, 'data', filename[i] + '_precision.npy'))
        recall = np.load(os.path.join(path, 'data', filename[i] + '_recall.npy'))
        plt.plot(recall, precision, color=color[i], lw=2, label=filename[i])

    y_true = np.load(os.path.join(path, 'data', 'allans.npy'))
    y_scores = np.load(os.path.join(path, 'out', 'allprob-'+epoch+'.npy'))

    precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    plt.plot(recall[:], precision[:], lw=2, color='navy', label='BGRU+2ATT')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig('pr_result-epoch'+epoch)
    plt.close()
