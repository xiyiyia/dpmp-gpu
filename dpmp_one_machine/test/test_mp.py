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
from models import resnet_gpu, resnet, resnet_gpipe
from torch.utils.tensorboard import SummaryWriter
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
from collections import OrderedDict
loss_function = nn.CrossEntropyLoss()

"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.
This code is transformed :mod:`torchvision.models.resnet`.
"""
from collections import OrderedDict
from typing import Any, List

from torch import nn

from models.bottleneck import bottleneck
from models.flatten_sequential import flatten_sequential

__all__ = ['resnet152']


def build_resnet(layers: List[int],
                 num_classes: int = 10,
                 inplace: bool = False
                 ) -> nn.Sequential:
    """Builds a ResNet as a simple sequential model.
    Note:
        The implementation is copied from :mod:`torchvision.models.resnet`.
    """
    inplanes = 64

    def make_layer(planes: int,
                   blocks: int,
                   stride: int = 1,
                   inplace: bool = False,
                   ) -> nn.Sequential:
        nonlocal inplanes

        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(bottleneck(inplanes, planes, stride, downsample, inplace))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(bottleneck(inplanes, planes, inplace=inplace))

        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', make_layer(64, layers[0], inplace=inplace)),
        ('layer2', make_layer(128, layers[1], stride=2, inplace=inplace)),
        ('layer3', make_layer(256, layers[2], stride=2, inplace=inplace)),
        ('layer4', make_layer(512, layers[3], stride=2, inplace=inplace)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', nn.Flatten()),
        ('fc', nn.Linear(512 * 4, num_classes)),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)

    # Initialize weights for Conv2d and BatchNorm2d layers.
    # Stolen from torchvision-0.4.0.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            return

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            return

    model.apply(init_weight)

    return model


def resnet152(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 8, 36, 3], **kwargs)

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
def flatten_sequential(module):
    def _flatten(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Sequential):
                for sub_name, sub_child in _flatten(child):
                    yield (f'{name}_{sub_name}', sub_child)
            else:
                yield (name, child)
    return nn.Sequential(OrderedDict(_flatten(module)))
""" Distributed Synchronous SGD Example """
def run(args, model):
    torch.manual_seed(1234)
    # model.cuda()
    # model = nn.Sequential(a, b, c, d)
    # print(model)
        
    # partitions = torch.cuda.device_count()
    partitions = args.g
    sample = torch.empty(args.b, 3, 224, 224)
    balance = balance_by_time(partitions, model, sample)
    model = GPipe(resnet152(), balance, chunks=10)
    # print(model)
    # summary(model.cuda(), [(3, 255, 255)])
    dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                # transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    train_set = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.b,
                                              shuffle=True,
                                              )
    # train_set = train_set.cuda()
    # model = vgg11_bn()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    # num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    start = time.time()
    for epoch in range(1):
        epoch_loss = 0.0
        for data, target in train_set:
            
            data = data.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            batch_start = time.time()
            output = model(data)
            stop = time.time()
            print('training_time_fw', stop - batch_start)
            # print(len(output),len(target))
            target = target.to(output.device)
            # loss = loss_function(output, target).cuda()
            # print(len(output),len(target))
            # if(len(output) == len(target)):
            loss = loss_function(output, target).cuda()
            epoch_loss += loss.item()
            # print(epoch_loss, loss)
            loss.backward()
            # average_gradients(model)
            optimizer.step()
            #back_stop = time.time()
            #print('training_time_bc', stop - start)
            batch_stop = time.time()
            print('training_time_bc', batch_stop - stop)
            # print('Rank ', 0, ', epoch ',
            #       epoch, ': ', epoch_loss)
        print('Rank ', 0, ', epoch ',
                epoch, ': ', epoch_loss/len(train_set))
    stop = time.time()
    print('training_time_dp', stop - start)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, default=2, help='number of gpus')
    parser.add_argument('-b', type=int, default=100, help='batchsize')
    args = parser.parse_args()
    # print(torch.cuda.device_count())
    size = args.g
    # size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")


    #model parallel compare 
    stmt = "run(args,model)"

    # setup = "model = ModelParallelvgg(g = 2)"
    # setup = "model = resnet_gpu.resnet152(args)"
    # setup = "model = resnet.resnet152()"
    setup = "model = resnet_gpipe.resnet152(args)"
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=1, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)
    print("time_training_mp",mp_mean,mp_std)
    # setup = "model = VGG().cuda()"
    # rn_run_times = timeit.repeat(
    #     stmt, setup, number=1, repeat=1, globals=globals())
    # rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


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

    print("time_training_mp",mp_mean,mp_std)
    plot([mp_mean],
        [mp_std],
        ['Model Parallel'],
        'pipe_mp.png')