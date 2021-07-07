#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""Blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 1:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '163.143.0.101'
    os.environ['MASTER_PORT'] = '6003'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    os.environ['MASTER_ADDR'] = '163.143.0.101'
    os.environ['MASTER_PORT'] = '6003'
    dist.init_process_group('gloo', rank=2, world_size=size)
    # run(2, size)
    # for rank in range(size):
    #     p = mp.Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()