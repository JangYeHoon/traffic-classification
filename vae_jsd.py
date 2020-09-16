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

z_dim = 3
input = 6
pd_colum = z_dim*2+1
epoch_number = 20
pkt_cnt = 70
app = 40

reconstruction_function = nn.MSELoss(reduction='sum')
class Packet_Dataset(Dataset):
	def __init__(self, feature_dir, transform=None):
		self.csv_file = pd.read_csv(feature_dir)
		self.transform = transform
		df = pd.read_csv(feature_dir)
		df = np.array(df)
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


class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		self.fc1 = nn.Linear(input, 32)
		self.fc2 = nn.Linear(32, 16)
		self.fc31 = nn.Linear(16, z_dim)
		self.fc32 = nn.Linear(16, z_dim)
		self.fc4 = nn.Linear(z_dim, 16)
		self.fc5 = nn.Linear(16, 32)
		self.fc6 = nn.Linear(32, input)

	def encode(self, x):
		h1 = F.relu(self.fc1(x))
		h2 = F.relu(self.fc2(h1))
		return self.fc31(h2), self.fc32(h2)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z):
		h4 = F.relu(self.fc4(z))
		h5 = F.relu(self.fc5(h4))
		return torch.sigmoid(self.fc6(h5))
	
	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, input))
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
	BCE = reconstruction_function(recon_x, x)

	# https://arxiv.org/abs/1312.6114 (Appendix B)
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element).mul_(-0.5)

	return BCE + KLD
#############################
#
# train
#
#############################
def vae_train(model, train_dataset, norm_columns):
	optimizer = optim.Adam(model.parameters(), lr=0.0001)
	# 시간 재기 코드
	start_time  = time.time()
	check = [0, 0, 0, 0, 0, 0]
	idx = 0
	train_norm_list = list([] for i in range(6))
	for epoch in range(epoch_number):
		total_time = time.time()
		model.train()
		train_loss = 0
		for inputs, classes in train_dataset:
			optimizer.zero_grad()
			recon_batch, mu, logvar = model(inputs[0].float())
			# Loss 계
			loss = loss_function(recon_batch, inputs[0].float(), mu, logvar)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			temp_list = []
			if epoch == epoch_number-1:
				for i in range(0, 6):
					if classes[0][0][i] == 1:
						idx = i
				for i in range(0, z_dim):
					mean = mu.detach().numpy()
					sigma = logvar.detach().numpy()
					esp = torch.exp(0.5*logvar)
					esp2 = esp.detach().numpy()
					temp_list.append(mean[0][i])
					if sigma[0][i] < 0:
						sigma[0][i] *= -1
					temp_list.append(esp2[0][i])
				temp_list.append(idx)
				train_norm_list[idx].append(temp_list)
				check[idx] += 1
		print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / train_dataset.__len__()))
		print('epoch interval time : ',(time.time() - total_time))
	torch.save(model.state_dict(), 'model/model_vae_jsd_app' + str(app) + '_f' + str(input) + '_c' + str(pkt_cnt) + '_z'+ str(z_dim) + '.pth')				# model 저장 함수
	finish_time = time.time()
	print('time : ', start_time - finish_time)
	norm_list = []
	norm_list2 = []
	for i in range(6):
		norm_list.append(random.sample(train_norm_list[i], 40))
	for i in range(6):
		for j in range(40):
			norm_list2.append(norm_list[i][j])
	df = pd.DataFrame(
		norm_list2,
		columns = norm_columns
	)
	df.to_csv('data/vae_train_data_app' + str(app) + '_' + str(pkt_cnt) + '_z' + str(z_dim) + '_i' + str(input) + '.csv')


#############################
#
# test data get mu,sigma
#
#############################
def vae_test(model, test_dataset, norm_columns):
	model.eval()
	test_loss = 0
	test_norm_list = []
	train_norm_list = list([] for i in range(6))
	with torch.no_grad():
		for inputs, classes in test_dataset:
			recon_batch, mu, logvar = model(inputs[0].float())
			test_loss += loss_function(recon_batch, inputs[0].float(), mu, logvar).item()
					
			temp_list = []
			for i in range(0, 6):
				if classes[0][0][i] == 1:
					idx = i
			for i in range(0, z_dim):
				mean = mu.detach().numpy()
				sigma = logvar.detach().numpy()
				esp = torch.exp(0.5*logvar)
				esp2 = esp.detach().numpy()
				temp_list.append(mean[0][i])
				if sigma[0][i] < 0:
					sigma[0][i] *= -1
				temp_list.append(esp2[0][i])
			temp_list.append(idx)
			test_norm_list.append(temp_list)
	test_loss /= test_dataset.__len__()
	print('====> Test set loss: {:.4f}'.format(test_loss))
	df = pd.DataFrame(
		test_norm_list,
		columns = norm_columns
	)
	df.to_csv('data/vae_test_data_app' + str(app) + '_' + str(pkt_cnt) + '_z' + str(z_dim) + '_i' + str(input) + '.csv')
	
#############################
#
# test
#
#############################
def jsd_test():
	train_csv = pd.read_csv('data/vae_train_data_app' + str(app) + '_' + str(pkt_cnt) + '_z' + str(z_dim) + '_i' + str(input) + '.csv')
	train_csv = np.array(train_csv)
	train_list = train_csv[:,1:pd_colum]
	train_result = train_csv[:,pd_colum:]

	test_csv = pd.read_csv('data/vae_test_data_app' + str(app) + '_' + str(pkt_cnt) + '_z' + str(z_dim) + '_i' + str(input) + '.csv')
	test_csv = np.array(test_csv)
	test_list = test_csv[:,1:pd_colum]
	result_array = test_csv[:,pd_colum:]

	y_true = []
	y_pred = []
	y_pred2 = []

	class_correct = list(0. for i in range(6))
	class_correct2 = list(0. for i in range(6))
	class_total = list(0. for i in range(6))

	matrix = [-1 for row in range(6)]

	js = [0 for row in range(z_dim)]
	time_check = []
	for k in range(len(test_list)):
		start_time = time.time()
		for i in range(len(train_list)):
			for id in range(0, pd_colum-1, 2):
				p_x = np.random.normal(train_list[i][id], train_list[i][id+1], size=200)
				q_x = np.random.normal(test_list[i][id], test_list[i][id+1], size=200)
				idx = int(id/2)
				
				p = norm.pdf(p_x, train_list[i][id], train_list[i][id+1])
				q = norm.pdf(q_x, test_list[k][id], test_list[k][id+1])
				m = (p+q)/2
				kl_pm = entropy(p,m)
				kl_qm = entropy(q,m)
				js[idx] = kl_pm/2 + kl_qm/2
			r = int(train_result[i][0])
			s = 0
			for jsd in range(z_dim):
				s += js[jsd]
			if matrix[r] == -1:
				matrix[r] = s
			#else:
			#	matrix[r] += s
			elif matrix[r] > s:
				matrix[r] = s
		#for i in range(0, 6):
		#	matrix[i] = matrix[i] / 10
		pred = 0
		pred2 = 0
		_, pred = torch.min(torch.FloatTensor(matrix), 0)
		classes = int(result_array[k][0])
		pred2 = pred
		if classes == pred:
			class_correct[pred] += 1
			class_correct2[pred2] += 1
		else:
			matrix[pred2] = 10000000.99999
			_, pred2 = torch.min(torch.FloatTensor(matrix), 0)
			if classes == pred2:
				class_correct2[pred2] += 1
		y_pred.append(pred)
		y_pred2.append(pred2)
		
		class_total[classes] += 1
		y_true.append(classes)
		
		time_check.append(time.time() - start_time)
		
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

	print('max_time : %f' % (np.max(time_check)))
	print('avg_time : %f' % (np.mean(time_check)))
	print('min_time : %f' % (np.min(time_check)))

if __name__ == "__main__":
	model = VAE()
	transform = transforms.ToTensor()
	train_dataset = Packet_Dataset(feature_dir='data/train_data_' + str(pkt_cnt) + '.csv',
									   transform=transform)
	test_dataset = Packet_Dataset(feature_dir='data/test_data_' + str(pkt_cnt) + '.csv',
									   transform=transform)
	norm_columns = ['mu', 'sig', 'mu', 'sig', 'mu', 'sig', 'class']
	vae_train(model, train_dataset, norm_columns)
	vae_test(model, test_dataset, norm_columns)
	jsd_test()