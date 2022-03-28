import torch
from torch import nn
from torch.nn import functional as F
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, outp, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp, outp, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outp, outp, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(outp)
        self.down_sample = nn.Sequential()
        if stride != 1 or inp != outp * self.expansion:
            self.down_sample = nn.Sequential(
                nn.Conv2d(inp, outp * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outp * self.expansion),
            )

    def forward(self, x):
        res = self.down_sample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = F.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp, outp, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inp, outp, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outp)
        self.conv2 = nn.Conv2d(outp, outp, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outp)
        self.conv3 = nn.Conv2d(outp, outp * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outp * self.expansion)
        self.down_sample = nn.Sequential()
        if stride != 1 or inp != outp * self.expansion:
            self.down_sample = nn.Sequential(
                nn.Conv2d(inp, outp * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outp * self.expansion),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.down_sample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = F.relu(x)
        return x


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, num_classes=4):
        super(Resnet, self).__init__()
        self.in_plane = 64
        self.up_sample = nn.Upsample(scale_factor=8, mode='bilinear');
        self.conv1 = nn.Conv2d(in_channel, self.in_plane,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_plane, planes, stride))
            self.in_plane = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        m = nn.Softmax(dim=1)
        return m(x)


def Resnet18(**kwargs):
    model = Resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def Resnet50(**kwargs):
    model = Resnet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model
