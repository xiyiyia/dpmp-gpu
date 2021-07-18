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
import pandas as pd
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
Processing = [] # processing time for all tasks on GPU0
Training = [] # training time for all tasks on GPU0
Communication = [] # communication time for all tasks on GPU0


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
    # size = dist.get_world_size()
    size = args.g
    bsz = args.b
    partition_sizes = [1.0 / size for _ in range(size)]
    print(partition_sizes)
    partition = DataPartitioner(dataset, partition_sizes)
    # partition = partition.use(dist.get_rank())
    partition = partition.use(0)
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
    # len_data = len(data)//50
    # base_time = time.time()
    # print(model)
    data_trained = 0
    # communications = []
    # trainings = []
    len_ = 0
    for epoch in range(epochs):
        throughputs = []
        elapsed_times = []
        # training_time_list = []
        # communication_time_list = []
        for i, (input, target) in enumerate(data):
            if i <= 1:
                # if(rank == 0):
                #     load_data_ts = time.time()
                input = input.cuda()
                target = target.cuda()
                # if(rank == 0):
                #     load_data_te = time.time()
                    # print('data_time', load_data_te-load_data_ts)
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
                    Training.append(tte - tts)
                # if(i <= 50):
                #     average_gradients(model)

                if(rank == 0):
                    cts = time.time()
                average_gradients(model)
                if(rank == 0):
                    cte = time.time()
                    Communication.append(cte-cts)
                optimizer.step()
                optimizer.zero_grad()
                if(rank == 0):
                    tock = time.time()
                    elapsed_time = tock - tick
                    elapsed_times.append(elapsed_time)
                    print("print")
                    percent = (i+1) / len(data) * 100
                    throughput = data_trained / sum(elapsed_times)
                    log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                        '' % (epoch+1, epochs, percent, throughput), clear=True, nl=False)
            else: break
            len_ = i+1
        if(rank == 0):
            throughput = 50000 / (sum(elapsed_times)/(epoch+1))
            log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
                '' % (epoch+1, epochs, throughput, sum(elapsed_times)/(epoch+1)), clear=True)
            throughputs.append(throughput)

    # print(data_trained)
    # if(rank == 0):
    #     n = len(throughputs)
    #     throughput = sum(throughputs) / n
    #     elapsed_time = sum(elapsed_times) / n
    #     communication = sum(communications) / n
    #     training = sum(trainings) / n
    #     click.echo('%.3f samples/sec, total: %.3f sec/epoch, communication: %.3f sec/epoch, training: %.3f sec/epoch (average)'
    #             '' % (throughput, elapsed_time, communication,training))

        # # print(len(trainings),len(communications))
        # name_ = [i for i in range(len_*epochs)]
        # # print(len(name_))
        # training_time = pd.DataFrame(columns=name_,data=np.array(trainings).reshape(1,len_*epochs))
        # communication_time = pd.DataFrame(columns=name_,data=np.array(communications).reshape(1,len_*epochs))
        # training_time.to_csv('./training_time'+args.n+'.csv',encoding='gbk')
        # communication_time.to_csv('./communication_time'+args.n+'.csv',encoding='gbk')

def init_model(args):
    if(args.n == 'vgg'):
        model = vgg.vgg19_bn()
    if(args.n == 'resnet101'):
        model = resnet.resnet101()
    if(args.n == 'resnet18'):
        model = resnet.resnet18()
    if(args.n == 'resnet50'):
        model = resnet.resnet50()
    return model

def init_process(args,rank, fn, model, data, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # dataset_size = 50000//args.g
    dist.init_process_group(args.ben, rank=rank, world_size=args.g)
    # dist.init_process_group("gloo", rank=rank, world_size=args.g)
    torch.cuda.set_device(rank)
    # if(rank == 0):
    #     load_model_ts = time.time()
    model = model.to(rank)
    # if(args.n == 'vgg'):
    #     model = model.to(rank)
    # if(args.n == 'resnet101'):
    #     model = model.to(rank)
    # if(args.n == 'resnet18'):
    #     model = model.to(rank)
    # if(args.n == 'resnet50'):
    #     model = model.to(rank)
        # model = resnet.resnet18().to(rank)
    # print(model)

    # if(rank == 0):
    #     load_model_te = time.time()
    #     print('model_time', load_model_te-load_model_ts)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )

    # dataset_size = 50000//args.g
    # input = torch.rand(args.b, 3, 32, 32)#, device='cuda:'+str(rank))  ## remove args.g
    # target = torch.randint(10, (args.b,))#, device='cuda:'+str(rank))  ## remove args.g
    # data = [(input, target)] * (dataset_size//args.b)

    # if dataset_size % args.b != 0:       ## remove args.g
    #     last_input = input[:dataset_size % args.b]  ## remove args.g
    #     last_target = target[:dataset_size % args.b]   ## remove args.g
    #     data.append((last_input, last_target))                         # random make data

    fn(rank, args.g, model, args.e, args, data)

    # fn(rank, args.g, model, args.e, args)

def store():
    # print(len(name_))
    dataframe = pd.DataFrame(Processing, columns=['X'])
    dataframe = pd.concat([dataframe, pd.DataFrame(Training,columns=['Y'])],axis=1)
    dataframe = pd.concat([dataframe, pd.DataFrame(Communication,columns=['Z'])],axis=1)
    dataframe.to_csv("./Time.csv",header = False,index=False,sep=',')

if __name__ == "__main__":

    scale = 4 # num of tasks
    GPUs = 4

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-g', type=int, default=1, help='number of gpus')
    # parser.add_argument('-b', type=int, default=128, help='batchsize')
    # parser.add_argument('-e', type=int, default=1, help='epoch')
    # parser.add_argument('-ben', type=str, default='nccl')
    # args_1 = parser.parse_args()
    mp.set_start_method("spawn")

    # data, bsz = partition_dataset(args_1)

    Args = [[None for i in range (GPUs)] for j in range (scale)]
    Model = [[None for i in range (GPUs)] for j in range (scale)]
    Data = [[None for i in range (GPUs)] for j in range (scale)]
    BSZ = [[None for i in range (GPUs)] for j in range (scale)]
    for i in range (scale):
        if i % 4 == 0: network = 'resnet101'
        elif i %4 == 1: network = 'resnet18'
        elif i %4 == 2: network = 'resnet50'
        elif i% 4 == 3: network = 'vgg'
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', type=int, default=GPUs, help='number of gpus')
        parser.add_argument('-b', type=int, default=128, help='batchsize')
        parser.add_argument('-e', type=int, default=1, help='epoch')
        parser.add_argument('-ben', type=str, default='nccl')
        parser.add_argument('-n', type=str, default=network)
        for j in range (GPUs):
            Args[i][j] = parser.parse_args()
            Model[i][j] = init_model(Args[i][j])
            Data[i][j], BSZ[i][j] = partition_dataset(Args[i][j])

    
    for i in range (scale):
        processes = []
        for rank in range(GPUs):
            if rank == 0:
                process_time_start = time.time()
            p = mp.Process(target=init_process, args=(Args[i][rank], rank, run, Model[i][rank], Data[i][rank]))
            p.start()
            if rank == 0:
                process_time_end = time.time()
                # save the processing time
                Processing.append(process_time_end - process_time_start)
            processes.append(p)
        for p in processes:
            p.join()
    print(Training)
    print(Communication)
    store()