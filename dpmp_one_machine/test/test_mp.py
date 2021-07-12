"""run.py:"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# from dpmp_one_machine.test.test_dp import make_layers
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
import timeit
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import resnet_gpu, resnet
from torch.utils.tensorboard import SummaryWriter

loss_function = nn.CrossEntropyLoss()

class VGG(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.cfg = [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M']
        self.features = make_layers(self.cfg,batch_norm=True)
        self.classifier = nn.Sequential(
          nn.Linear(512, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, num_class)
        )
    def forward(self, x):
      output = self.features(x)
      output = output.view(output.size()[0], -1)
      output = self.classifier(output)
      return output
def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
      if l == 'M':
          layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          continue
      layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
      if batch_norm:
          layers += [nn.BatchNorm2d(l)]
      layers += [nn.ReLU(inplace=True)]
      input_channel = l
    return nn.Sequential(*layers)

class ModelParallelvgg(VGG):
    def __init__(self, num_class=10,g = 1):
        super().__init__()
        self.split_size = int(128/g)
        self.g = g
        self.cfg = [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M']
        self.features = make_layers(self.cfg, batch_norm=True)
        # iter = (i for i in range(50))
        # sum(1 for _ in iter)
        # self.list = [None for i in range(2)]
        
        self.seq1 = self.features[0:int(sum(1 for _ in self.features)/self.g)]
        self.seq2 = self.features[sum(1 for _ in self.seq1):sum(1 for _ in self.features)]
        self.list = [self.seq1,self.seq2]
        # print(torch.nn.Sequential(*(list(self.seq1)+list(self.seq2))))
        self.classifier = nn.Sequential(
          nn.Linear(512, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, num_class)
        )
        # g = 1
        if(g >= 2):
          self.seq1 = self.seq1.to('cuda:0')
          self.seq2 = self.seq2.to('cuda:1')
          self.classifier = self.classifier.to('cuda:1')
    def forward(self, x):
        if(self.g >= 2):
            splits = iter(x.split(self.split_size, dim=0))
            s_next = next(splits)
            s_prev = self.seq1(s_next).to('cuda:1')
            ret = []
            for s_next in splits:
                s_prev = self.seq2(s_prev)
                ret.append(self.classifier(s_prev.view(s_prev.size()[0], -1)))
                s_prev = self.seq1(s_next).to('cuda:1')
            s_prev = self.seq2(s_prev)
            ret.append(self.classifier(s_prev.view(s_prev.size(0), -1)))
            return torch.cat(ret)
        else:
            output = self.features(x)
            output = output.view(output.size()[0], -1)
            output = self.classifier(output)
            return output
def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
      if l == 'M':
          layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          continue
      layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
      # print(layers)
      if batch_norm:
          layers += [nn.BatchNorm2d(l)]
      layers += [nn.ReLU(inplace=True)]
      input_channel = l
    return nn.Sequential(*layers)

""" Distributed Synchronous SGD Example """
def run(model):
    torch.manual_seed(1234)
    # model.cuda()
    print(model)
    # summary(model.cuda(), [(3, 255, 255)])
    dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                # transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    train_set = torch.utils.data.DataLoader(dataset,
                                              batch_size=128,
                                              shuffle=True,
                                              )
    # train_set = train_set.cuda()
    # model = vgg11_bn()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    # num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(1):
        epoch_loss = 0.0
        for data, target in train_set:
          data = data.cuda()
          target = target.cuda()
    
          optimizer.zero_grad()
    
          output = model(data)
          # print(len(output),len(target))
          target = target.to(output.device)
          # loss = loss_function(output, target).cuda()
          loss = loss_function(output, target).cuda()
          epoch_loss += loss.item()
          # print(epoch_loss, loss)
          loss.backward()
          # average_gradients(model)
          optimizer.step()
        # print('Rank ', dist.get_rank(), ', epoch ',
        #       epoch, ': ', epoch_loss / num_batches)
        print('Rank ', 0, ', epoch ',
              epoch, ': ', epoch_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, default=2, help='number of gpus')
    parser.add_argument('-b', type=int, default=128, help='batchsize')
    args = parser.parse_args()
    print(torch.cuda.device_count())
    # size = args.g
    size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")


    #model parallel compare 
    stmt = "run(model)"

    # setup = "model = ModelParallelvgg(g = 2)"
    setup = "model = resnet_gpu.resnet50(args)"
    # setup = "model = resnet.resnet50()"
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=1, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    setup = "model = VGG().cuda()"
    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=1, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


    def plot(means, stds, labels, fig_name):
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(means)), means, yerr=stds,
            align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
        ax.set_ylabel('ResNet50 Execution Time (Second)')
        ax.set_xticks(np.arange(len(means)))
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close(fig)


    plot([mp_mean, rn_mean],
        [mp_std, rn_std],
        ['Model Parallel', 'Single GPU'],
        'mp_vs_rn.png')