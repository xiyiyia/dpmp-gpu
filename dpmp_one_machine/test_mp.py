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
    # size = dist.get_world_size()
    size = 1
    bsz = 128 / float(size)
    # partition_sizes = [1.0 / size for _ in range(size)]
    # partition = DataPartitioner(dataset, partition_sizes)
    # partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(dataset,
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
    # size = float(dist.get_world_size())
    size = 1
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

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
        iter = (i for i in range(50))
        sum(1 for _ in iter)
        self.seq1 = self.features[0:int(sum(1 for _ in self.features)/self.g)]
        self.seq2 = self.features[sum(1 for _ in self.seq1):sum(1 for _ in self.features)]
        self.classifier = nn.Sequential(
          nn.Linear(512, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, num_class)
        )
        if(g >= 2):
          # self.features = self.features.to('cuda:0')
          self.seq1 = self.seq1.to('cuda:0')
          self.seq2 = self.seq2.to('cuda:1')
          self.classifier = self.classifier.to('cuda:1')
    def forward(self, x):
      # for i in self.features:
      #   print(i)
      # for i in self.seq1:
      #   print(i)
      # for j in self.seq2:
      #   print(j)
      # print(self.features)
      if(self.g >= 2):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        # print(s_next.size(0),s_next.size(1))
        s_prev = self.seq1(s_next).to('cuda:1')
        # print(len(splits),len(s_prev),len(s_next))
        ret = []

        for s_next in splits:
          # print('error?')
          s_prev = self.seq2(s_prev)
          # print(len(s_prev),len(s_next))
          # print('error?',s_next.size(0),s_next.size(1),s_prev.size(0),s_prev.size(1),s_prev.view(s_prev.size(0), -1).size(0),s_prev.view(s_prev.size(0), -1).size(1))
          # output.view(output.size()[0], -1)
          ret.append(self.classifier(s_prev.view(s_prev.size()[0], -1)))
          # print('error?')
          s_prev = self.seq1(s_next).to('cuda:1')
          # print(len(s_prev),len(s_next))
          # print('error?')
        s_prev = self.seq2(s_prev)
        # print('error?')
        ret.append(self.classifier(s_prev.view(s_prev.size(0), -1)))
        # output = self.features(x.to('cuda:0'))
        # output = output.view(output.size()[0], -1)
        # output = self.classifier(output).to('cuda:1')
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
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    num_batches = math.ceil(len(train_set.dataset) / float(bsz))

    # next(model.parameters()).is_cuda

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
          loss = loss_function(output, target)
          epoch_loss += loss.item()
          print(epoch_loss, loss)
          loss.backward()
          # average_gradients(model)
          optimizer.step()
        # print('Rank ', dist.get_rank(), ', epoch ',
        #       epoch, ': ', epoch_loss / num_batches)
        print('Rank ', 0, ', epoch ',
              epoch, ': ', epoch_loss / num_batches)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    # fn(rank, size)

    dist.init_process_group("nccl", rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    model = VGG().to(rank)
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
  # size = args.g
  size = torch.cuda.device_count()
  processes = []
  mp.set_start_method("spawn")
  
  #data parallel
  # torch.distributed.init_process_group(
  #     backend='nccl', world_size=N, init_method='...'
  # )
  # model = DistributedDataParallel(model, device_ids=[i], output_device=i)
  # for rank in range(size):
  #     p = mp.Process(target=init_process, args=(rank, size, run))
  #     p.start()
  #     processes.append(p)
  # for p in processes:
  #     p.join()

  #model parallel compare 
  stmt = "run(0,1,model)"

  setup = "model = ModelParallelvgg(g = 2)"
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