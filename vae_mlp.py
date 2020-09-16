import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
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
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import scipy.stats as st
import torch.nn.functional as F
from scipy.spatial import distance
import pyro.distributions as dist

z_dim = 1
input = 6
pd_colum = z_dim*2+1
epoch_number = 20
pkt_cnt = 20

class Normal(object):
	def __init__(self, mu, sigma, log_sigma, v=None, r=None):
		self.mu = mu
		self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
		self.logsigma = log_sigma
		dim = mu.get_shape()
		if v is None:
			v = torch.FloatTensor(*dim)
		if r is None:
			r = torch.FloatTensor(*dim)
		self.v = v
		self.r = r

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.fc1 = nn.Linear(input, 32)
		self.fc2 = nn.Linear(32, 16)
		self.fc31 = nn.Linear(16, z_dim)
		self.fc32 = nn.Linear(16, z_dim)

	def forward(self, x):
		h1 = F.relu(self.fc1(x))
		h2 = F.relu(self.fc2(h1))
		return self.fc31(h2), self.fc32(h2)
	
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
class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()
		
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x):
		mu, logvar = self.encoder.forward(x.view(-1, input))
		z = self.reparameterize(mu, logvar)
		return self.decoder.forward(z), mu, logvar
		
class MLP(nn.Module):
	def __init__(self, encoder):
		super(MLP, self).__init__()
		
		self.encoder = encoder
		self.fc1 = nn.Linear(z_dim,32)
		self.fc2 = nn.Linear(32,16)
		self.fc3 = nn.Linear(16,6)
	
	def forward(self, x):
		z_loc, z_scale = self.encoder.forward(x)
		z = dist.Normal(z_loc, torch.exp(z_scale)).sample()
		
		x = self.fc1(z)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		
		return x

reconstruction_function = nn.MSELoss(reduction='sum')
def loss_function(recon_x, x, mu, logvar):
	BCE = reconstruction_function(recon_x, x)

	# https://arxiv.org/abs/1312.6114 (Appendix B)
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element).mul_(-0.5)

	return BCE + KLD

class Packet_Dataset(Dataset):
	def __init__(self, feature_dir, transform=None, shuf=True):
		self.csv_file = pd.read_csv(feature_dir)
		self.transform = transform
		df = pd.read_csv(feature_dir)
		df = np.array(df)
		if shuf == True:
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
# vae train
#
#############################
def vae_train(vae):
	transform = transforms.ToTensor()
	flow_dataset = Packet_Dataset(feature_dir='data/train_data_' + str(pkt_cnt) + '.csv',
									   transform=transform, shuf=False)

	optimizer = optim.Adam(vae.parameters(), lr=0.0001)
	# 시간 재기 코드
	start_time  = time.time()
	for epoch in range(epoch_number):
		total_time = time.time()
		vae.train()
		train_loss = 0
		pd_list = []
		for inputs, classes in flow_dataset:
			optimizer.zero_grad()
			recon_batch, mu, logvar = vae(inputs[0].float())
			# Loss 계
			loss = loss_function(recon_batch, inputs[0].float(), mu, logvar)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / flow_dataset.__len__()))
		print('epoch interval time : ',(time.time() - total_time))
	#torch.save(vae.state_dict(), 'model/model_vae_mlp_f6_c200_z1.pth')				# model 저장 함수
	finish_time = time.time()
	print('time : ', start_time - finish_time)
	vae.encoder.freeze()

	#############################
	#
	# vae + mlp train
	#
	#############################
def vae_mlp_train(vae):
	transform = transforms.ToTensor()
	train_dataset = Packet_Dataset(feature_dir='data/train_data_' + str(pkt_cnt) + '.csv',
									   transform=transform, shuf=True)
	classifier = MLP(encoder=vae.encoder)
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
	torch.save(classifier.state_dict(), 'model/model_vae_mlp_f' + str(input) + '_c' + str(pkt_cnt) + '_z'+ str(z_dim) + '.pth')				# model 저장 함수
	finish_time = time.time()
	print('time : ', start_time - finish_time)
	
	return classifier

#############################
#
# test
#
#############################
def test(classifier):
	transform = transforms.ToTensor()
	test_dataset = Packet_Dataset(feature_dir='data/test_data_' + str(pkt_cnt) + '.csv',
									   transform=transform, shuf=False)
	classifier.eval()
	criterion = nn.MSELoss()
	test_loss = 0
	class_correct = list(0. for i in range(6))
	class_correct2 = list(0. for i in range(6))
	class_total = list(0. for i in range(6))
	y_true = []
	y_pred = []
	y_pred2 = []
	pd_list = []
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
			y_pred.append(pred)
			classes = -1
			for i in range(0, 6):
				if target[0][0].data[i] == 1:
					y_true.append(i)
					classes = i
			class_total[classes] += 1
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

if __name__ == "__main__":
	vae = VAE()
	vae_train(vae)
	classifier = vae_mlp_train(vae)
	test(classifier)