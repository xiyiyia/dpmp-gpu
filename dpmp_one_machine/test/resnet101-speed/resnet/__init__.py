"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.

This code is transformed :mod:`torchvision.models.resnet`.

"""
from collections import OrderedDict
from typing import Any, List

from torch import nn

from resnet.bottleneck import bottleneck
from resnet.flatten_sequential import flatten_sequential

__all__ = ['resnet101']

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


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

def build_vgg() -> nn.Sequential:
    # cfg = cfg['E']
    def make_layers(cfg, batch_norm=True):
        layers = []

        input_channel = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            # layers += [nn.ReLU(inplace=True)]
            layers += [nn.ReLU()]
            input_channel = l
        return nn.Sequential(*layers)
        
    
    model = nn.Sequential(*(list(make_layers(cfg['A']))), 
            nn.Flatten(),            
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000))
    # print(model)
    model = flatten_sequential(model)
    # print(model)

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


def resnet101(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    # ResNet(BasicBlock, [2, 2, 2, 2])
    return build_resnet([2, 2, 2, 2], **kwargs) # resnet18
    # return build_resnet([3, 4, 23, 3], **kwargs) # 101
    # return build_resnet([3, 8, 36, 3], **kwargs) 
    # return ResNet(BottleNeck, [3, 4, 6, 3])

def vgg11(**kwargs: Any) -> nn.Sequential:
    """Constructs a vgg11 model."""
    return build_vgg(**kwargs)