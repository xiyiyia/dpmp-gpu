import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

import torchvision.models
import time
import matplotlib.pyplot as plt
from rbm_new import RBM
import sys
from functions import *
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import io,sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-l', type=int, default=0.1, help='learning rate')
parser.add_argument('-b', type=int, default=128, help='batchsize')
parser.add_argument('-e', type=int, default=100, help='epoch')
parser.add_argument('-d', type=str, default='mnist')
parser.add_argument('-n', type=str, default='resnet')
parser.add_argument('-ne', type=int, default=10, help='number of n_estimators')
args = parser.parse_args()

def get_Dataloader_model(d,batch_size):
    # Load data

    if d == 'kmnist':
        train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

        test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
        # transformer = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #     ]
        # )
        # train_dataset = datasets.CIFAR10(
        #         './data', train=True, download=True, transform=transformer
        #     )
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        # )
        # test_dataset = datasets.CIFAR10(
        #         './data', train=False, download=True, transform=transformer
        #     )
        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        # )
    if d == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    if d == 'fmnist':
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    if d == 'qmnist':
        train_dataset = torchvision.datasets.QMNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

        test_dataset = torchvision.datasets.QMNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
        # transformer = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #     ]
        # )

        # train_dataset = datasets.CIFAR100(
        #         './data', train=True, download=True, transform=transformer
        #     )
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        # )
        # test_dataset = datasets.CIFAR100(
        #         './data', train=False, download=True, transform=transformer
        #     )
        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        # )
    return train_dataset, test_dataset, train_loader,test_loader

########## CONFIGURATION ##########
BATCH_SIZE = args.b
if (args.d == 'cifar10' or args.d == 'cifar100'):
    VISIBLE_UNITS = 1024*3  # 28 x 28 images
else:
    VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
EPOCHS = args.e


CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device('cuda:0')


########## LOADING DATASET ##########
print('Loading dataset...')

train_dataset, test_dataset, train_loader, test_loader = get_Dataloader_model(args.d,args.b)


########## TRAINING RBM ##########
print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)
error_list = []
start = time.time()
for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS).cuda()  # flatten input data

        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch).cpu()

        epoch_error += batch_error
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error/(BATCH_SIZE*len(train_loader))))
    error_list.append(epoch_error/(BATCH_SIZE*len(train_loader)))
stop = time.time()
print('time:',stop-start)
plt.plot(error_list)
plt.savefig('./pic/'+args.d+'loss.png')

########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()



########## CLASSIFICATION ##########
print('Classifying...')

clf = LogisticRegression()
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)
print(predictions)
plt.figure(figsize=(5,5), dpi=180)
for i in range(0,4):
    for j in range(0,4):
        img = np.array(predictions[i*4+j]).reshape(28,28)
        plt.subplot(4,4,i*4+j+1)
        plt.imshow(img ,cmap = plt.cm.gray)
plt.savefig('./pic/'+args.d+'.png')

print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))

