import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import random
import copy
import pandas as pd
import numpy as np
import queue
import math
import argparse
import time
import sys

from models import inceptionv3, resnet, vgg, mobilenet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def Set_dataset(dataset, args):
    if dataset == 'CIFAR10':

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='/home/ICDCS/cifar-10-batches-py/', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.b, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='/home/ICDCS/cifar-10-batches-py/', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.b, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader, testloader
    elif dataset == 'MNIST':

        # Data
        print('==> Preparing data..')
        # normalize
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        # download dataset
        trainset = torchvision.datasets.MNIST(root = "./data/",
                        transform=transform,
                        train = True,
                        download = True)
        # load dataset with batch=64
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size = 64,
                                            shuffle = True)

        testset = torchvision.datasets.MNIST(root="./data/",
                        transform = transform,
                        train = False)

        testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size = 64,
                                            shuffle = False)
        return trainloader, testloader
    else:
        print ('Data load error!')
        return 0

def Set_model(client, args):
    print('==> Building model..')
    # Model = [None for i in range (client)]
    # Optimizer = [None for i in range (client)]
    if(args.net == 'vgg'):
        model = vgg.vgg19_bn()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        return model, optimizer
    if(args.net == 'resnet101'):
        model = resnet.resnet101()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        return model, optimizer
    if(args.net == 'resnet18'):
        model = resnet.resnet18()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        return model, optimizer
    if(args.net == 'resnet50'):
        model = resnet.resnet50()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        return model, optimizer
    if(args.net == 'inception3'):
        model = inceptionv3.inceptionv3()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        return model, optimizer
    if(args.net == 'mobilenet'):
        model = mobilenet.mobilenet()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        return model, optimizer


def Train(model, optimizer, client, trainloader):
    print('==> Training model..')
    criterion = nn.CrossEntropyLoss().to(device)
    #print(next(model[0].parameters()).is_cuda)
    # cpu ? gpu

    model = model.to(device)
    P = [None for i in range (client)]

    # share a common dataset
    train_loss = 0
    correct = 0
    total = 0
    Loss = 0
    time_start = time.time()
    Batch_time = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx < 1:
                batch_start = time.time()
                inputs, targets = inputs.to(device), targets.to(device)
                idx = (batch_idx % client)
                model.train()
                optimizer.zero_grad()
                outputs = model(inputs)
                Loss = criterion(outputs, targets)
                Loss.backward()
                optimizer.step()
                train_loss += Loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                batch_end = time.time()
                Batch_time.append(batch_end - batch_start)
            else:
                break


def run(dataset, client, net):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--b',default=128,type=int,help='batch_size')
    parser.add_argument('--e',default=1,type=int,help='epoch')
    parser.add_argument('--net', default=net,type=str,help='Net')
    args = parser.parse_args()


    X, Y, Z = [], [], []
    trainloader, testloader = Set_dataset(dataset, args)
    model, optimizer = Set_model(client, args)
    # pbar = tqdm(range(args.epoch))
    start_time = 0
    # for i in pbar:
    return model, optimizer, trainloader, client

if __name__ == '__main__':
    step = 40
    # with optimization

    Model = [None for i in range (step)]
    Trainloader = [None for i in range (step)]
    Optimizer = [None for i in range (step)]
    Client = [None for i in range (step)]
    for i in range (step):
        j = random.randint(0,6)
        if j == 1: network = 'vgg'
        elif j == 2: network = 'resnet18'
        elif j == 3: network = 'resnet50'
        elif j == 4: network = 'resnet101'
        elif j == 5: network = 'inception3'
        elif j == 6: network = 'mobilenet'
        Model[i], Optimizer[i], Trainloader[i], Client[i] = run(dataset = 'CIFAR10', client = 1, net = network)

    for i in range (step):
        Train(Model[i], Optimizer[i], Client[i], Trainloader[i])
        torch.cuda.empty_cache()


    # without optimization
    # for i in range (step):
    #     if i%4 == 0: 
    #         Model, Optimizer, Trainloader, Client = run(dataset = 'CIFAR10', client = 1, net = 'MobileNet')
    #         Train(Model, Optimizer, Client, Trainloader)
    #         torch.cuda.empty_cache()
    #     elif i%4 == 1: 
    #         Model, Optimizer, Trainloader, Client = run(dataset = 'CIFAR10', client = 1, net = 'ResNet18')
    #         Train(Model, Optimizer, Client, Trainloader)
    #         torch.cuda.empty_cache()
    #     elif i%4 == 2: 
    #         Model, Optimizer, Trainloader, Client = run(dataset = 'CIFAR10', client = 1, net = 'MobileNet')
    #         Train(Model, Optimizer, Client, Trainloader)
    #         torch.cuda.empty_cache()
    #     elif i%4 == 3: 
    #         Model, Optimizer, Trainloader, Client = run(dataset = 'CIFAR10', client = 1, net = 'ResNet18')
    #         Train(Model, Optimizer, Client, Trainloader)
    #         torch.cuda.empty_cache()
