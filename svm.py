import sklearn
from sklearn import svm
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.rc('font', size=20)
    plt.rc('axes', labelsize=17)   # x,y축 label 폰트 크기
    plt.rc('xtick', labelsize=15)  # x축 눈금 폰트 크기 
    plt.rc('ytick', labelsize=15)  # y축 눈금 폰트 크기
    plt.rc('legend', fontsize=15)  # 범례 폰트 크기
    plt.rc('figure', titlesize=15) # figure title 폰트 크기
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

pkt_cnt = 70

Trainset = pandas.read_csv('data/train_data_' + str(pkt_cnt) + '.csv')
Testset = pandas.read_csv('data/test_data_' + str(pkt_cnt) + '.csv')

Traindata = Trainset.drop(['num', 'class'], axis=1)
Trainlabel = Trainset['class'].copy()

Testdata = Testset.drop(['num', 'class'], axis=1)
Testlabel = Testset['class'].copy()

sc = StandardScaler()
sc.fit(Traindata)

Traindata_std = sc.transform(Traindata.values.reshape(-1,6))
Testdata_std = sc.transform(Testdata.values.reshape(-1,6))

Model = svm.SVC()
Model.fit(Traindata_std, Trainlabel)
Predict = Model.predict(Testdata_std)

label=['voip', 'game', 'real-time', 'non-real-time', 'cloud', 'web']
conf = confusion_matrix(Testlabel, Predict)
plot_confusion_matrix(conf, target_names=label)

print('accuracy', metrics.accuracy_score(Testlabel,Predict) )