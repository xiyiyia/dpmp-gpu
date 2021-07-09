"""run.py:"""
#!/usr/bin/env python

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
from torch.utils.tensorboard import SummaryWriter

loss_function = nn.CrossEntropyLoss()

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        # rng = random.random()
        random.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

""" Partitioning MNIST """
def partition_dataset():
    dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                # transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=int(bsz),
                                         shuffle=True)
    print(train_set, bsz)
    return train_set, bsz

# def run(rank, size):
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])

""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
          # Send send_buff
          send_req = dist.isend(send_buff, right)
          dist.recv(recv_buff, left)
          accum[:] += recv_buff[:]
       else:
          # Send recv_buff
          send_req = dist.isend(recv_buff, right)
          dist.recv(send_buff, left)
          accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features

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
      # print(x)
      output = self.features(x)
      # print(output)
      output = output.view(output.size()[0], -1)
      # print(output)
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

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

""" Distributed Synchronous SGD Example """
def run(rank, size, model):
    torch.manual_seed(1234)
    train_set_abort, bsz = partition_dataset()
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
    # print("你是什么脸")
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    # print("你是什么脸")
    num_batches = math.ceil(len(train_set.dataset) / float(bsz))

    # next(model.parameters()).is_cuda

    for epoch in range(1):
        epoch_loss = 0.0
        for data, target in train_set:
          data = data.cuda()
          target = target.cuda()
          # print("你是什么脸")
          optimizer.zero_grad()
          # print("你是什么脸")
          output = model(data)
          print(len(output),len(target))
          loss = loss_function(output, target)
          epoch_loss += loss.item()
          print(epoch_loss, loss)
          loss.backward()
          average_gradients(model)
          optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    # fn(rank, size)

    dist.init_process_group("nccl", rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    model = vgg11_bn().to(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )
    print(rank)
    # print("done", model)
    fn(rank, size, model)
    # # Rank 1 gets one more input than rank 0.
    # inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
    # with model.join():
    #     for _ in range(5):
    #         for inp in inputs:
    #             loss = model(inp).sum()
    #             loss.backward()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, default=1, help='number of gpus')
    args = parser.parse_args()
    print(torch.cuda.device_count())
    size = args.g
    processes = []
    mp.set_start_method("spawn")
    
    # torch.distributed.init_process_group(
    #     backend='nccl', world_size=N, init_method='...'
    # )
    # model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()