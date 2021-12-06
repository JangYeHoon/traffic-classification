import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.autograd import Variable
import random
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from numpy import *
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import itertools

input = 6
pkt_cnt = 70
n_epochs = 20# 학습 epoch 지정
criterion = nn.MSELoss()

class SimDataset(Dataset):
    def __init__(self, feature_dir, transform=None):
        self.csv_file = pd.read_csv(feature_dir)
        self.transform = transform
        df = pd.read_csv(feature_dir)
        df = np.array(df)
        random.shuffle(df)
        df2 = df[:,1:input+1]
        self.result_array = df[:,input+1:]
        result = []
        for i in range(len(self.result_array)):
            templist = []
            for j in range(0, 6):
                if self.result_array[i] == j:
                    templist.append(1)
                else:
                    templist.append(0)
            result.append(templist)
        self.result_array = result
        standard_scaler = RobustScaler()
        x_scaled = standard_scaler.fit_transform(df2)
        self.feature_array = x_scaled
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        result = self.result_array[idx]
        result = np.array([result])
        feature = self.feature_array[idx]
        feature = np.array([feature])
        
        if self.transform:
            feature = self.transform(feature)
            result = self.transform(result)
        result = result.float()

        return feature, result
transform = transforms.ToTensor()
face_dataset = SimDataset(feature_dir='data/train_data_' + str(pkt_cnt) + '.csv',
                                   transform=transform)
test_loader = SimDataset(feature_dir='data/test_data_' + str(pkt_cnt) + '.csv',
                                   transform=transform)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,6)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        
        return x

def mlp_train(net):
	net.train()
	visualize_loss = 0.0
	optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
	for epoch in range(n_epochs):
		logs = {}
		train_loss = 0.0
		###################
		#    모델 학습    #
		###################
		for data, target in face_dataset:
			# 모든 optimizer 변수와 gradients를 초기화
			optimizer.zero_grad()
			# 정방향 학습 : 입력을 모델로 전달하여 예측된 출력 계산 
			output = net(data[0].float())
			
			# Loss 계
			loss = criterion(output[0], target[0][0])
			
			# 역전파 : 모델의 매개변수를 고려하여 loss의 gradients를 계산
			loss.backward()
			# 매개변수 업데이트
			optimizer.step()
			# 훈련 Loss 업데이트
			train_loss += loss.item()*data.size(0)
			visualize_loss = train_loss/face_dataset.__len__()
		print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1,visualize_loss))
	torch.save(net.state_dict(), 'model/model_mlp_f' + str(input) + '_c' + str(pkt_cnt) + '.pth')				# model 저장 함수

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

def mlp_test(net):
	test_loss = 0.0
	class_correct = list(0. for i in range(6))
	class_correct2 = list(0. for i in range(6))
	class_total = list(0. for i in range(6))
	y_true = []
	y_pred = []
	y_pred2 = []

	net.eval()

	for data, target in test_loader:
		output = net(data[0].float())
		loss = criterion(output[0], target[0][0])
		test_loss += loss.item()*data.size(0)
		_, pred = torch.max(output, 1)
		pred2 = pred
		if target[0][0].data[pred] == 1:
			class_correct[pred] += 1
			class_correct2[pred2] += 1
		else:
			output[0][pred2] = -1000
			_, pred2 = torch.max(output, 1)
			if target[0][0].data[pred2] == 1:
				class_correct2[pred2] += 1
		y_pred2.append(pred2)
		y_pred.append(pred.item())
		classes = -1
		for i in range(0, 6):
			if target[0][0].data[i] == 1:
				classes = i
		class_total[classes] += 1
		y_true.append(classes)

	test_loss = test_loss/test_loader.__len__()
	print('Test Loss: {:.6f}\n'.format(test_loss))
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

	print('------------top 2------------')
	for i in range(6):
		if class_total[i] > 0:
			print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
				str(i), 100 * class_correct2[i] / class_total[i],
				np.sum(class_correct2[i]), np.sum(class_total[i])))
		else:
			print('Test Accuracy of %5s: N/A (no training examples)' % (i))

	print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
		100. * np.sum(class_correct2) / np.sum(class_total),
		np.sum(class_correct2), np.sum(class_total)))
	
	label=['voip', 'game', 'real-time', 'non-real-time', 'cloud', 'web']
	conf = confusion_matrix(y_true, y_pred)
	plot_confusion_matrix(conf, target_names=label)

if __name__ == "__main__":
	net = Net()
	mlp_train(net)
	mlp_test(net)