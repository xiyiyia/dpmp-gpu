"""run.py:"""
#!/usr/bin/env python

from models import resnet
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import argparse
import time
import math
from datetime import datetime
import torch.nn.functional as F
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchensemble_ import BaggingClassifier  # voting is a classic ensemble strategy
from torchensemble_  import bagging as bagg
from models import resnet,vgg
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import io,sys
import time
loss_function = nn.CrossEntropyLoss()

def get_Dataloader_model(n,d,batch_size):
    # Load data

    if d == 'cifar10':
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                )
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                )
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                )
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                )
            ]
        )
        train_loader = DataLoader(
            datasets.FashionMNIST('./data', train=True, download=True, transform=train_transformer),
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
                transforms.ToTensor()
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        train_loader = DataLoader(
            datasets.CIFAR100('./data', train=True, download=True, transform=train_transformer),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            datasets.CIFAR100('./data', train=False, transform=test_transformer),
            batch_size=batch_size,
            shuffle=True,
        )
        # Define the ensemble
        if n == 'resnet':
            model = resnet.resnet18(num_classes=100)              # here is your deep learning model
        if n == 'vgg':
            model = vgg.vgg11_bn(num_class=100)               # here is your deep learning model
        return train_loader,test_loader,model.cuda()

    # Define the ensemble
    if n == 'resnet':
        model = resnet.resnet18(num_classes=10)         # here is your deep learning model
    if n == 'vgg':
        model = vgg.vgg11_bn(num_class=10)         # here is your deep learning model
    return train_loader,test_loader,model.cuda()

@torch.no_grad()
def eval_training(test_loader,model):

    model.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    sum = 0.0
    for (images, labels) in test_loader:

        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        sum += 1
    print(correct/sum)


""" Distributed Synchronous SGD Example """
def run(args, model):
    
    train_loader,test_loader,model = get_Dataloader_model(args.n,args.d,args.b)

    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    error_list = []
    start = time.time()
    for epoch in range(args.e):
        loss_ = []
        epoch_loss = 0.0
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            loss_.append(epoch_loss / args.b)
        print(', epoch ', epoch, ': ', epoch_loss / args.b)
        error_list.append(sum(loss_)/len(train_loader))
    stop = time.time()
    print('time:',stop-start)
    plt.plot(error_list)
    plt.savefig('./pic/'+args.d+'loss.png')
    eval_training(test_loader,model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=int, default=0.1, help='learning rate')
    parser.add_argument('-b', type=int, default=128, help='batchsize')
    parser.add_argument('-e', type=int, default=1, help='epoch')
    parser.add_argument('-d', type=str, default='cifar10')
    parser.add_argument('-n', type=str, default='resnet')
    parser.add_argument('-ne', type=int, default=3, help='number of n_estimators')
    args = parser.parse_args()

    run(args)

