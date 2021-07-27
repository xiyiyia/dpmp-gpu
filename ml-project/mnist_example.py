import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms
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

    if d == 'cifar10':
        train_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 28),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 28),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_loader = DataLoader(
            datasets.CIFAR10(
                './data', train=True, download=True, transform=train_transformer
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            datasets.CIFAR10('./data', train=False, transform=test_transformer),
            batch_size=batch_size,
            shuffle=True,
        )
    if d == 'mnist':
        train_transformer = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        train_loader = DataLoader(
            datasets.MNIST(
                './data', train=True, download=True, transform=train_transformer
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            datasets.MNIST('./data', train=False, transform=test_transformer),
            batch_size=batch_size,
            shuffle=True,
        )
    if d == 'fmnist':
        train_transformer = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        train_loader = DataLoader(
            datasets.FashionMNIST(
                './data', train=True, download=True, transform=train_transformer
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=test_transformer),
            batch_size=batch_size,
            shuffle=True,
        )

    if d == 'cifar100':
        train_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 28),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 28),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_loader = DataLoader(
            datasets.CIFAR100(
                './data', train=True, download=True, transform=train_transformer
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            datasets.CIFAR100('./data', train=False, transform=test_transformer),
            batch_size=batch_size,
            shuffle=True,
        )
    return train_loader,test_loader

########## CONFIGURATION ##########
BATCH_SIZE = args.b
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
EPOCHS = args.e


CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)


########## LOADING DATASET ##########
print('Loading dataset...')

train_loader, test_loader = get_Dataloader_model(args.d,args.b)


########## TRAINING RBM ##########
print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)
error_list = []
for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

    error_list.append(epoch_error)
plt.plot(error_list.cpu())
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

print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))

