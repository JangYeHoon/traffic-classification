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
import itertools

z_dim = 3
input = 6
pd_colum = z_dim*2+1
epoch_number = 20
pkt_cnt = 70

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
        self.fc7 = nn.Linear(z_dim, 32)
        self.fc8 = nn.Linear(32, 16)
        self.fc9 = nn.Linear(16, 6)

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
	
    def clf_supervised(self, z):
        h6 = F.relu(self.fc7(z))
        h7 = F.relu(self.fc8(h6))
        return self.fc9(h7)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.clf_supervised(z)

reconstruction_function = nn.MSELoss(reduction='sum')
classification_function = nn.MSELoss()
def loss_function(recon_x, x, mu, logvar, output, target):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    clf_loss = classification_function(output[0], target[0][0])

    return BCE + KLD + clf_loss

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


#############################
#
# train
#
#############################
transform = transforms.ToTensor()
face_dataset = Packet_Dataset(feature_dir='data/train_data_' + str(pkt_cnt) + '.csv',
                                   transform=transform)

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 시간 재기 코드
start_time  = time.time()
check = [0, 0, 0, 0, 0, 0]
idx = 0
for epoch in range(epoch_number):
    total_time = time.time()
    model.train()
    train_loss = 0
    pd_list = []
    for inputs, classes in face_dataset:
        optimizer.zero_grad()
        recon_batch, mu, logvar, output = model(inputs[0].float())
        # Loss 계
        loss = loss_function(recon_batch, inputs[0].float(), mu, logvar, output, classes)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
      epoch, train_loss / face_dataset.__len__()))
    print('epoch interval time : ',(time.time() - total_time))
torch.save(model.state_dict(), 'model/model_vae_mlp_f6_c200_z1.pth')				# model 저장 함수
finish_time = time.time()
print('time : ', start_time - finish_time)

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
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
# test data get mu,sigma
#
#############################
transform = transforms.ToTensor()
face_dataset = Packet_Dataset(feature_dir='data/test_data_' + str(pkt_cnt) + '.csv',
                                   transform=transform)
model.eval()
test_loss = 0
class_correct = list(0. for i in range(6))
class_correct2 = list(0. for i in range(6))
class_total = list(0. for i in range(6))
y_true = []
y_pred = []
y_pred2 = []
pd_list = []
with torch.no_grad():
    for inputs, target in face_dataset:
        recon_batch, mu, logvar, output = model(inputs[0].float())
        test_loss += loss_function(recon_batch, inputs[0].float(), mu, logvar, output, target).item()
		
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
test_loss /= face_dataset.__len__()
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