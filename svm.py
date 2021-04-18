import sklearn
from sklearn import svm
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics

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

print('accuracy', metrics.accuracy_score(Testlabel,Predict) )