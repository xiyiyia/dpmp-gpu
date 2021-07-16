"""ResNet-101 Speed Benchmark"""
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
import resnet
from resnet import resnet101,vgg11
import torchgpipe
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import shutil

Stuffs = Tuple[nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]


class Experiments:

    @staticmethod
    def baseline(model: nn.Module, devices: List[int]) -> Stuffs:

        # partitions = args.g
        # sample = torch.empty(args.b, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, resnet152(), sample, device=torch.device('cuda'))
        # model = GPipe(resnet152(), balance, chunks=args.c)

        batch_size = 118
        device = devices[0]
        model.to(device)
        return model, batch_size, [torch.device(device)]

    @staticmethod
    def pipeline1(model: nn.Module, devices: List[int]) -> Stuffs:

        # batch_size = 256
        # chunks = 2

        batch_size = 220
        chunks = 2


        # partitions = 1
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)
        balance = [37] # vgg
        # balance = [370] # 101
        # balance = [183] # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline2(model: nn.Module, devices: List[int]) -> Stuffs:
        # batch_size = 512
        # chunks = 8

        batch_size = 4096
        chunks = 40

        # partitions = 2
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)

        balance = [16, 21]  # vgg
        # balance = [135, 235]  # 101
        # # balance = [67, 116]  # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)
    @staticmethod
    def pipeline3(model: nn.Module, devices: List[int]) -> Stuffs:
        # batch_size = 512
        # chunks = 16
        batch_size = 2048
        chunks = 100
        # batch_size = 1048
        # chunks = 100
        
        # partitions = 8
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)

        balance = [10, 12, 15] #vgg
        # balance = [26, 22, 33, 44, 44, 66, 66, 69] #101
        # balance = [13, 12, 14, 22, 22, 33, 33, 34]  # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)
    @staticmethod
    def pipeline4(model: nn.Module, devices: List[int]) -> Stuffs:
        # batch_size = 1048
        # chunks = 100
        batch_size = 4096
        chunks = 400

        # partitions = 4
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)

        # # batch_size = 560
        # # chunks = 4
        balance = [9, 9, 9, 10] # vgg
        # balance = [22, 46, 61, 54]  # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)
    @staticmethod
    def pipeline5(model: nn.Module, devices: List[int]) -> Stuffs:
        # batch_size = 512
        # chunks = 16
        batch_size = 2048
        chunks = 200
        # batch_size = 1048
        # chunks = 100
        
        # partitions = 8
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)

        balance = [7, 7, 7, 7, 9] #vgg
        # balance = [26, 22, 33, 44, 44, 66, 66, 69] #101
        # balance = [13, 12, 14, 22, 22, 33, 33, 34]  # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)
    @staticmethod
    def pipeline6(model: nn.Module, devices: List[int]) -> Stuffs:
        # batch_size = 512
        # chunks = 16
        batch_size = 4096
        chunks = 100
        # batch_size = 1024
        # chunks = 100
        
        # partitions = 8
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)

        balance = [4, 4, 5, 8, 8, 8] #vgg
        # balance = [26, 22, 33, 44, 44, 66, 66, 69] #101
        # balance = [13, 12, 14, 22, 22, 33, 33, 34]  # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)
    @staticmethod
    def pipeline7(model: nn.Module, devices: List[int]) -> Stuffs:
        # batch_size = 512
        # chunks = 16
        batch_size = 1024
        chunks = 24
        # batch_size = 2048  #best
        # chunks = 100
        # batch_size = 1024
        # chunks = 100
        # partitions = 8
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)

        balance = [4, 4, 4, 4, 6, 7, 8] #vgg
        # balance = [26, 22, 33, 44, 44, 66, 66, 69] #101
        # balance = [13, 12, 14, 22, 22, 33, 33, 34]  # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)
    @staticmethod
    def pipeline8(model: nn.Module, devices: List[int]) -> Stuffs:
        # batch_size = 512
        # chunks = 16
        batch_size = 2048
        chunks = 48
        # batch_size = 2048
        # chunks = 200
        
        # partitions = 8
        # sample = torch.empty(batch_size, 3, 224, 224).cuda()
        # balance = balance_by_time(partitions, model, sample, device=torch.device('cuda'))
        # model = GPipe(model, balance, chunks=chunks)

        balance = [3, 4, 4, 4, 4, 5, 5, 8] #vgg
        # balance = [26, 22, 33, 44, 44, 66, 66, 69] #101
        # balance = [13, 12, 14, 22, 22, 33, 33, 34]  # 50
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)


EXPERIMENTS: Dict[str, Experiment] = {
    'baseline': Experiments.baseline,
    'pipeline-1': Experiments.pipeline1,
    'pipeline-2': Experiments.pipeline2,
    'pipeline-3': Experiments.pipeline3,
    'pipeline-4': Experiments.pipeline4,
    'pipeline-5': Experiments.pipeline5,
    'pipeline-6': Experiments.pipeline6,
    'pipeline-7': Experiments.pipeline7,
    'pipeline-8': Experiments.pipeline8,
}


BASE_TIME: float = 0


def hr() -> None:
    """Prints a horizontal line."""
    width, _ = shutil.get_terminal_size()
    click.echo('-' * width)


def log(msg: str, clear: bool = False, nl: bool = True) -> None:
    """Prints a message with elapsed time."""
    if clear:
        # Clear the output line to overwrite.
        width, _ = shutil.get_terminal_size()
        click.echo('\b\r', nl=False)
        click.echo(' ' * width, nl=False)
        click.echo('\b\r', nl=False)

    t = time.time() - BASE_TIME
    h = t // 3600
    t %= 3600
    m = t // 60
    t %= 60
    s = t

    click.echo('%02d:%02d:%02d | ' % (h, m, s), nl=False)
    click.echo(msg, nl=nl)


def parse_devices(ctx: Any, param: Any, value: Optional[str]) -> List[int]:
    if value is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in value.split(',')]


@click.command()
@click.pass_context
@click.argument(
    'experiment',
    type=click.Choice(sorted(EXPERIMENTS.keys())),
)
@click.option(
    '--epochs', '-e',
    type=int,
    default=10,
    help='Number of epochs (default: 10)',
)
@click.option(
    '--skip-epochs', '-k',
    type=int,
    default=1,
    help='Number of epochs to skip in result (default: 1)',
)
@click.option(
    '--devices', '-d',
    metavar='0,1,2,3',
    callback=parse_devices,
    help='Device IDs to use (default: all CUDA devices)',
)

def cli(ctx: click.Context,
        experiment: str,
        epochs: int,
        skip_epochs: int,
        devices: List[int],
        ) -> None:
    """ResNet-101 Speed Benchmark"""
    if skip_epochs >= epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    model: nn.Module = vgg11()

    f = EXPERIMENTS[experiment]
    try:
        model, batch_size, _devices = f(model, devices)
    except ValueError as exc:
        # Examples:
        #   ValueError: too few devices to hold given partitions (devices: 1, paritions: 2)
        ctx.fail(str(exc))

    optimizer = SGD(model.parameters(), lr=0.1)

    in_device = _devices[0]
    out_device = _devices[-1]
    # torch.cuda.set_device(in_device)

    # This experiment cares about only training speed, rather than accuracy.
    # To eliminate any overhead due to data loading, we use fake random 224x224
    # images over 1000 labels.
    dataset_size = 50000

    input = torch.rand(batch_size, 3, 224, 224, device=in_device)
    target = torch.randint(1000, (batch_size,), device=out_device)
    data = [(input, target)] * (dataset_size//batch_size)

    # dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True,
    #                          transform=transforms.Compose([
    #                             # transforms.Resize([32, 32]),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.1307,), (0.3081,))
    #                          ]))
    # size = dist.get_world_size()
    # bsz = args.b
    # partition_sizes = [1.0 / size for _ in range(size)]
    # partition = DataPartitioner(dataset, partition_sizes)
    # partition = partition.use(dist.get_rank())
    # data = torch.utils.data.DataLoader(dataset,
    #                                      batch_size=batch_size,
    #                                      shuffle=True)

    if dataset_size % batch_size != 0:
        last_input = input[:dataset_size % batch_size]
        last_target = target[:dataset_size % batch_size]
        data.append((last_input, last_target))

    # HEADER ======================================================================================

    title = f'{experiment}, {skip_epochs+1}-{epochs} epochs'
    click.echo(title)

    if isinstance(model, GPipe):
        click.echo(f'batch size: {batch_size}, chunks: {model.chunks}, balance: {model.balance}')
    else:
        click.echo(f'batch size: {batch_size}')

    click.echo('torchgpipe: %s, python: %s, torch: %s, cudnn: %s, cuda: %s' % (
        torchgpipe.__version__,
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda
        )
        )

    # TRAIN =======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()
    # print(model)
    def run_epoch(epoch: int) -> Tuple[float, float]:
        # torch.cuda.synchronize(in_device)
        tick = time.time()

        data_trained = 0
        for i, (input, target) in enumerate(data):
            # input = input.to(in_device)
            # target = target.to(out_device)
            data_trained += input.size(0)
            # print(len(data),len(input))
            output = model(input)
            #if(len(output) == len(target)):
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # 00:01:02 | 1/20 epoch (42%) | 200.000 samples/sec (estimated)
            percent = (i+1) / len(data) * 100
            throughput = data_trained / (time.time()-tick)
            print('?')
            log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                '' % (epoch+1, epochs, percent, throughput), clear=True, nl=False)

        # torch.cuda.synchronize(in_device)
        tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = tock - tick
        throughput = dataset_size / elapsed_time
        log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
            '' % (epoch+1, epochs, throughput, elapsed_time), clear=True)
        print(data_trained)
        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []

    hr()
    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(epoch)

        if epoch < skip_epochs:
            continue

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
    hr()

    # RESULT ======================================================================================
    
    # pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n
    elapsed_time = sum(elapsed_times) / n
    click.echo('%s | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (title, throughput, elapsed_time))


if __name__ == '__main__':
    cli()
