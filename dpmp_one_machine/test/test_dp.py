"""run.py:"""
#!/usr/bin/env python

import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
# import pandas as pd
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
import click
from torch.utils.data import DataLoader, dataset
# from torch.utils.tensorboard import SummaryWriter
from models import inceptionv3, resnet, vgg
# import resnet
from typing import cast

loss_function = nn.CrossEntropyLoss()


def hr() -> None:
    """Prints a horizontal line."""
    width, _ = click.get_terminal_size()
    click.echo('-' * width)


def log(msg: str, clear: bool = False, nl: bool = True) -> None:
    """Prints a message with elapsed time."""
    if clear:
        # Clear the output line to overwrite.
        width, _ = click.get_terminal_size()
        click.echo('\b\r', nl=False)
        click.echo(' ' * width, nl=False)
        click.echo('\b\r', nl=False)

    t = time.time() - 0 #BASE_TIME
    h = t // 3600
    t %= 3600
    m = t // 60
    t %= 60
    s = t

    click.echo('%02d:%02d:%02d | ' % (h, m, s), nl=False)
    click.echo(msg, nl=nl)

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
def partition_dataset(args):
    dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                # transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = args.b
    partition_sizes = [1.0 / size for _ in range(size)]
    print(partition_sizes)
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=int(bsz/size),
                                         shuffle=True)
    return train_set, bsz


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


""" Distributed Synchronous SGD Example """
def run(rank, size, model, epochs, args, data):
    # torch.manual_seed(1234)
    # train_set, bsz = partition_dataset(args)
    # data, bsz = partition_dataset(args)

    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    len_data = len(data)//50
    base_time = time.time()
    # print(model)
    data_trained = 0
    for epoch in range(epochs):
        throughputs = []
        elapsed_times = []
        communications = []
        trainings = []
        # training_time_list = []
        # communication_time_list = []
        # name = [i for i in range(len(train_set))]
        for i, (input, target) in enumerate(data):
            if(rank ==0):
                tick = time.time()
            data_trained += input.size(0)
            if(rank == 0):
                tts = time.time()
            output = model(input)
            loss = loss_function(output, target)
            loss.backward()
            if(rank == 0):
                tte = time.time()
                trainings.append(tte-tts)
            if(i <= 50):
                average_gradients(model)
            optimizer.step()
            optimizer.zero_grad()
            if(rank == 0):
                elapsed_time = tock - tick
                elapsed_times.append(elapsed_time)
                print("print")
                percent = (i+1) / len(data) * 100
                throughput = data_trained / sum(elapsed_times)
                log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                    '' % (epoch+1, epochs, percent, throughput), clear=True, nl=False)
                tock = time.time()

        # training_time_list = np.array(training_time_list).reshape(1,len(train_set))
        # communication_time_list = np.array(communication_time_list).reshape(1,len(train_set))
        # training_time = pd.DataFrame(columns=name,data=training_time_list)
        # communication_time = pd.DataFrame(columns=name,data=communication_time_list)
        # training_time.to_csv('./training_time'+str(dist.get_rank())+'.csv',encoding='gbk')
        # communication_time.to_csv('./communication_time'+str(dist.get_rank())+'.csv',encoding='gbk')

        # torch.cuda.synchronize(in_device)
        if(rank == 0):
            throughput = 50000 / (sum(elapsed_times)/(epoch+1))
            log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
                '' % (epoch+1, epochs, throughput, sum(elapsed_times)/(epoch+1)), clear=True)
            throughputs.append(throughput)
    print(data_trained)
    if(rank == 0):
        n = len(throughputs)
        throughput = sum(throughputs) / n
        elapsed_time = sum(elapsed_times) / n
        communication = sum(communications) / n
        training = sum(trainings) / n
        click.echo('%.3f samples/sec, total: %.3f sec/epoch, communication: %.3f sec/epoch, training: %.3f sec/epoch (average)'
                '' % (throughput, elapsed_time, communication,training))

def init_process(args,rank, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'

    # dataset_size = 50000//args.g
    dist.init_process_group(args.ben, rank=rank, world_size=args.g)
    # dist.init_process_group("gloo", rank=rank, world_size=args.g)
    torch.cuda.set_device(rank)
    model = vgg.vgg11_bn().to(rank)
    # model = resnet.resnet101(num_classes=10)
    # model = cast(nn.Sequential, model)
    # model = resnet.resnet18().to(rank)
    # print(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )
    dataset_size = 50000//args.g
    input = torch.rand(args.b, 3, 224, 224, device='cuda:'+str(rank))  ## remove args.g
    target = torch.randint(1000, (args.b,), device='cuda:'+str(rank))  ## remove args.g
    data = [(input, target)] * (dataset_size//args.b)
    # print(dataset_size)
    # print(data[0])

    if dataset_size % args.b != 0:       ## remove args.g
        last_input = input[:dataset_size % args.b]  ## remove args.g
        last_target = target[:dataset_size % args.b]   ## remove args.g
        data.append((last_input, last_target))

    fn(rank, args.g, model, args.e, args, data)

if __name__ == "__main__":
    # torchvision.datasets.CIFAR10('./data', train=True, download=True,
    #                          transform=transforms.Compose([
    #                             # transforms.Resize([32, 32]),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.1307,), (0.3081,))
    #                          ]))
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, default=1, help='number of gpus')
    parser.add_argument('-b', type=int, default=128, help='batchsize')
    parser.add_argument('-e', type=int, default=1, help='epoch')
    parser.add_argument('-ben', type=str, default='nccl')
    args_1 = parser.parse_args()


    
    processes = []
    mp.set_start_method("spawn")

    for rank in range(args_1.g):
        p = mp.Process(target=init_process, args=(args_1, rank, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()