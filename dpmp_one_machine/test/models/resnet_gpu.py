"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, args, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.split_size = int(args.b/args.g)
        self.g = args.g
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.Sequential( nn.AdaptiveAvgPool2d((1, 1)) )
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature_list = [None for i in range(args.g)]
        if(args.g == 2):
            # a = torch.nn.Sequential(*(list(self.conv1)+list(self.conv2_x)+list(self.conv3_x)+list(self.conv4_x))).to('cuda:0')
            self.layer_1 = torch.nn.Sequential(*(list(self.conv1)+list(self.conv2_x)+list(self.conv3_x)+list(self.conv4_x))).to('cuda:0')
            self.feature_list[0] = self.layer_1
            self.layer_2 = torch.nn.Sequential(*(list(self.conv5_x)+list(self.avg_pool))).to('cuda:1')
            # a = torch.nn.Sequential(*(list(self.conv5_x)+list(self.avg_pool))).to('cuda:1')
            self.feature_list[1] = self.layer_2
            self.fc = self.fc.to('cuda:1')
        if(args.g == 3):
            self.layer_1 = torch.nn.Sequential(*(list(self.conv1)+list(self.conv2_x)+list(self.conv3_x))).to('cuda:0')
            self.feature_list[0] =  self.layer_1 
            self.layer_2 = torch.nn.Sequential(*(list(self.conv4_x)+list(self.conv5_x))).to('cuda:1')
            self.feature_list[1] = self.layer_2
            self.layer_3 = torch.nn.Sequential(*(list(self.avg_pool))).to('cuda:2')
            self.feature_list[2] =  self.layer_3
            self.fc = self.fc.to('cuda:2')
        if(args.g == 4):
            # self.layer_1 = torch.nn.Sequential(*(list(self.conv1)+list(self.conv2_x))).to('cuda:0')
            # self.feature_list[0] =  self.layer_1
            self.feature_list[0] = torch.nn.Sequential(*(list(self.conv1)+list(self.conv2_x))).to('cuda:0')
            # self.layer_2 = torch.nn.Sequential(*(list(self.conv3_x)+list(self.conv4_x))).to('cuda:1')
            # self.feature_list[1] = self.layer_2 
            self.feature_list[1] = torch.nn.Sequential(*(list(self.conv3_x)+list(self.conv4_x))).to('cuda:1')
            # self.layer_3 = torch.nn.Sequential(*(list(self.conv5_x))).to('cuda:2')
            # self.feature_list[2] = self.layer_3
            self.feature_list[2] = torch.nn.Sequential(*(list(self.conv5_x))).to('cuda:2')
            # self.layer_4 = torch.nn.Sequential(*(list(self.avg_pool))).to('cuda:3')
            # self.feature_list[3] = self.layer_4
            self.feature_list[3] = torch.nn.Sequential(*(list(self.avg_pool))).to('cuda:3')
            self.fc = self.fc.to('cuda:3')
        if(args.g == 5):
            # self.layer_1 = torch.nn.Sequential(*(list(self.conv1)+list(self.conv2_x))).to('cuda:0')
            # self.feature_list[0] = self.layer_1
            self.feature_list[0] = torch.nn.Sequential(*(list(self.conv1)+list(self.conv2_x))).to('cuda:0')
            # self.layer_2 = torch.nn.Sequential(*(list(self.conv3_x))).to('cuda:1')
            # self.feature_list[1] = self.layer_2
            self.feature_list[1] = torch.nn.Sequential(*(list(self.conv3_x))).to('cuda:1')
            # self.layer_3 = torch.nn.Sequential(*(list(self.conv4_x))).to('cuda:2')
            # self.feature_list[2] = self.layer_3
            self.feature_list[2] = torch.nn.Sequential(*(list(self.conv4_x))).to('cuda:2')
            # self.layer_4 = torch.nn.Sequential(*(list(self.conv5_x))).to('cuda:3')
            # self.feature_list[3] = self.layer_4
            self.feature_list[3] = torch.nn.Sequential(*(list(self.conv5_x))).to('cuda:3')
            # self.layer_5 = torch.nn.Sequential(*(list(self.avg_pool))).to('cuda:4')
            # self.feature_list[4] = self.layer_5
            self.feature_list[4] = torch.nn.Sequential(*(list(self.avg_pool))).to('cuda:4')
            self.fc = self.fc.to('cuda:4')
        if(args.g == 6):
            self.feature_list[0] = torch.nn.Sequential(*(list(self.conv1))).to('cuda:0')
            self.feature_list[0] = torch.nn.Sequential(*(list(self.conv2_x))).to('cuda:1')
            self.feature_list[1] = torch.nn.Sequential(*(list(self.conv3_x))).to('cuda:2')
            self.feature_list[2] = torch.nn.Sequential(*(list(self.conv4_x))).to('cuda:3')
            self.feature_list[3] = torch.nn.Sequential(*(list(self.conv5_x))).to('cuda:4')
            self.feature_list[4] = torch.nn.Sequential(*(list(self.avg_pool))).to('cuda:5')
            self.fc = self.fc.to('cuda:5')
        # if(args.g == 7):
        #     self.feature_list[0].append( torch.nn.Sequential(*(list(self.conv1))).to('cuda:0'))
        #     self.feature_list[1].append( torch.nn.Sequential(*(list(self.conv2_x))).to('cuda:1'))
        #     self.feature_list[2].append( torch.nn.Sequential(*(list(self.conv3_x))).to('cuda:2'))
        #     self.feature_list[3].append( torch.nn.Sequential(*(list(self.conv4_x))).to('cuda:3'))
        #     self.feature_list[4].append( torch.nn.Sequential(*(list(self.conv5_x))).to('cuda:4'))
        #     self.feature_list[5].append( torch.nn.Sequential(*(list(self.avg_pool))).to('cuda:5'))
        #     self.fc = self.fc.to('cuda:5')
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # output = self.conv1(x)
        # output = self.conv2_x(output)
        # output = self.conv3_x(output)
        # output = self.conv4_x(output)
        # output = self.conv5_x(output)
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)

        output_list = [None for i in range(self.g)]

        splits = iter(x.split(self.split_size, dim=0))
        ret = []
        for s_next in splits:
            # print(s_next, output_list)
            s_prev = self.feature_list[0](s_next)
            output_list[0] = s_prev.to('cuda:1')
            # print(output_list)
            for j in range(len(output_list) - 1):
                # print(output_list)
                # print(j,output_list[len(output_list) - j - 2],'no None')
                if(output_list[len(output_list) - j - 2] != None):
                    # print(j,output_list[len(output_list) - j - 2],'no None')
                    if(j == 0):
                        print('once')
                        ret.append(self.fc(output_list[len(output_list) - j - 2].view(s_prev.size(0), -1)))
                        output_list[len(output_list) - j - 2] = None
                    else:
                        output_list[len(output_list) - j - 1] = self.feature_list[len(output_list) - j - 1](output_list[len(output_list) - j - 2]).to('cuda:' + str(len(output_list) - j ))
                        output_list[len(output_list) - j - 2] = None
        output_list[0] = None
        # print(ret)
        a = True
        while( a == True):
            a = False
            for i in range(len(output_list)):
                # a = False
                if(output_list[i] != None):
                    a = True
            if(a == False):
                break
            for j in range(len(output_list) - 1):
                if(output_list[len(output_list) - j - 2] != None):
                    if(j == 0):
                        print('twice')
                        ret.append(self.fc(output_list[len(output_list) - j - 2].view(s_prev.size(0), -1)))
                        output_list[len(output_list) - j - 2] = None
                    else:
                        output_list[len(output_list) - j - 1] = self.feature_list[len(output_list) - j - 1](output_list[len(output_list) - j - 2]).to('cuda:' + str(len(output_list) - j))
                        output_list[len(output_list) - j - 2] = None
        print(len(torch.cat(ret)))
        # print(ret)

        return torch.cat(ret)

def resnet18(args):
    """ return a ResNet 18 object
    """
    return ResNet(args,BasicBlock, [2, 2, 2, 2])

def resnet34(args):
    """ return a ResNet 34 object
    """
    return ResNet(args,BasicBlock, [3, 4, 6, 3])

def resnet50(args):
    """ return a ResNet 50 object
    """
    return ResNet(args,BottleNeck, [3, 4, 6, 3])

def resnet101(args):
    """ return a ResNet 101 object
    """
    return ResNet(args,BottleNeck, [3, 4, 23, 3])

def resnet152(args):
    """ return a ResNet 152 object
    """
    return ResNet(args,BottleNeck, [3, 8, 36, 3])



