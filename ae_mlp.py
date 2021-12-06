import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torch.autograd import Variable
import random
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
import time
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import scipy.stats as st
import torch.nn.functional as F
from scipy.spatial import distance
import itertools

z_dim = 3
input = 6
pd_colum = z_dim*2+1
epoch_number = 20
pkt_cnt = 70

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.fc1 = nn.Linear(input, 32)
		self.fc2 = nn.Linear(32, 16)
		self.fc3 = nn.Linear(16, z_dim)
	def forward(self, x):
		h1 = F.relu(self.fc1(x))
		h2 = F.relu(self.fc2(h1))
		return self.fc3(h2)
	def freeze(self):
		for param in self.parameters():
			param.requires_grad = False
	
	def unfreeze(self):
		for param in self.parameters():
			param.requires_grad = True

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.fc4 = nn.Linear(z_dim, 16)
		self.fc5 = nn.Linear(16, 32)
		self.fc6 = nn.Linear(32, input)
	def forward(self, z):
		h4 = F.relu(self.fc4(z))
		h5 = F.relu(self.fc5(h4))
		return torch.sigmoid(self.fc6(h5))

class AE(nn.Module):
	def __init__(self):
		super(AE, self).__init__()
		
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):
		z = self.encoder.forward(x.view(-1, input))
		return self.decoder.forward(z)
		
class MLP(nn.Module):
	def __init__(self, encoder):
		super(MLP, self).__init__()
		
		self.encoder = encoder
		self.fc4 = nn.Linear(z_dim, 32)
		self.fc5 = nn.Linear(32, 16)
		self.fc6 = nn.Linear(16, 6)
		
	def forward(self, x):
		z = self.encoder.forward(x)
		
		h4 = F.relu(self.fc4(z))
		h5 = F.relu(self.fc5(h4))
		return F.relu(self.fc6(h5))

class Packet_Dataset(Dataset):
	def __init__(self, feature_dir, transform=None):
		self.csv_file = pd.read_csv(feature_dir)
		self.transform = transform
		df = pd.read_csv(feature_dir)
		df = np.array(df)
		np.random.shuffle(df)
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
		standard_scaler = StandardScaler()
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
#############################
#
# AE train
#
#############################
def ae_train(ae):
	transform = transforms.ToTensor()
	flow_dataset = Packet_Dataset(feature_dir='data/train_data_' + str(pkt_cnt) + '.csv',
									   transform=transform)
	loss_func = nn.MSELoss()
	optimizer = optim.Adam(ae.parameters(), lr=0.0001)
	start_time  = time.time()
	for epoch in range(epoch_number):
		total_time = time.time()
		ae.train()
		train_loss = 0
		pd_list = []
		for inputs, classes in flow_dataset:
			optimizer.zero_grad()
			recon_batch = ae(inputs[0].float())
			# Loss 계
			loss = loss_func(recon_batch, inputs[0].float())
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			temp_list = []
		print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / flow_dataset.__len__()))
		print('epoch interval time : ',(time.time() - total_time))
	#torch.save(full_model.state_dict(), 'model/model_ae_mlp_f6_c200_z5.pth')				# model 저장 함수
	finish_time = time.time()
	print('time : ', start_time - finish_time)
	ae.encoder.freeze()

#############################
#
# ae + mlp train
#
#############################
def ae_mlp_train(ae):
	transform = transforms.ToTensor()
	train_dataset = Packet_Dataset(feature_dir='data/train_data_' + str(pkt_cnt) + '.csv',
									   transform=transform)
	classifier = MLP(encoder=ae.encoder)
	optimizer = torch.optim.SGD(classifier.parameters(), lr=0.0001)
	criterion = nn.MSELoss()
	start_time  = time.time()
	for epoch in range(epoch_number):
		total_time = time.time()
		classifier.train()
		train_loss = 0
		pd_list = []
		for inputs, classes in train_dataset:
			optimizer.zero_grad()
			output = classifier(inputs[0].float())
			# Loss 계
			loss = criterion(output[0], classes[0][0])
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / train_dataset.__len__()))
		print('epoch interval time : ',(time.time() - total_time))
	torch.save(classifier.state_dict(), 'model/model_ae_mlp_f' + str(input) + '_c' + str(pkt_cnt) + '_z'+ str(z_dim) + '.pth')				# model 저장 함수
	finish_time = time.time()
	print('time : ', start_time - finish_time)
	return classifier

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

#############################
#
# test
#
#############################
def test(classifier):
	transform = transforms.ToTensor()
	test_dataset = Packet_Dataset(feature_dir='data/test_data_' + str(pkt_cnt) + '.csv',
									   transform=transform)
	classifier.eval()
	test_loss = 0
	class_correct = list(0. for i in range(6))
	class_correct2 = list(0. for i in range(6))
	class_total = list(0. for i in range(6))
	y_true = []
	y_pred = []
	y_pred2 = []
	pd_list = []
	criterion = nn.MSELoss()
	with torch.no_grad():
		for inputs, target in test_dataset:
			output = classifier(inputs[0].float())
			loss = criterion(output[0], target[0][0])
			test_loss += loss.item()*inputs.size(0)
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
	test_loss /= test_dataset.__len__()
	print('====> Test set loss: {:.4f}'.format(test_loss))


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
	ae = AE()
	ae_train(ae)
	classifier = ae_mlp_train(ae)
	test(classifier)