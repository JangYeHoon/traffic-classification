import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Statistical'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.rc('font', size=26)
    plt.rc('axes', labelsize=26)   # x,y축 label 폰트 크기
    plt.rc('xtick', labelsize=26)  # x축 눈금 폰트 크기 
    plt.rc('ytick', labelsize=26)  # y축 눈금 폰트 크기
    plt.rc('legend', fontsize=28)  # 범례 폰트 크기
    plt.rc('figure', titlesize=28) # figure title 폰트 크기
    plt.figure(figsize=(11, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=-60)
        plt.tick_params(axis='x', pad=5)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

data_csv = pd.read_csv('data/statistical_data.csv')
data_csv = np.array(data_csv)
feature_data = data_csv[:, 1:13]

testdata_csv = pd.read_csv('data/test_data_70.csv')
testdata_csv = np.array(testdata_csv)
test_data = testdata_csv[:, 1:7]
test_classes = testdata_csv[:, 7:]

y_true = []
y_pred = []
class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))
for i in range(len(test_classes)):
    result = -1
    pred = 0
    for c in range(len(feature_data)):
        cnt = 0
        for f in range(0, len(feature_data[c]), 2):
            if feature_data[c][f] >= test_data[i][int(f / 2)] and test_data[i][int(f / 2)] >= feature_data[c][f + 1]:
                cnt += 1
        if result < cnt:
            result = cnt
            pred = c
    y_pred.append(pred)
    if pred == test_classes[i][0]:
        class_correct[pred] += 1
    class_total[int(test_classes[i][0])] += 1
    y_true.append(int(test_classes[i][0]))

print('------------top 1------------')
for i in range(6):
	if class_total[i] > 0:
		print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
			str(i), 100 * class_correct[i] / class_total[i],
			np.sum(class_correct[i]), np.sum(class_total[i])))
	else:
		print('Test Accuracy of %5s: N/A (no training examples)' % (i))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
	100. * np.sum(class_correct) / np.sum(class_total),
	np.sum(class_correct), np.sum(class_total)))

label=['Voip', 'Game', 'Real-time\nstreaming', 'Non real-time\nstreaming', 'Cloud\nstorage', 'Web']
conf = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(conf, target_names=label)