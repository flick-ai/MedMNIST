import torch
from torch import nn
from torch.nn import functional as F
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, inp, outp, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(inp, outp, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(outp)
        self.conv2 = nn.Conv2d(outp, outp, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outp)
        self.conv3 = nn.Conv2d(outp, self.expansion*outp, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * outp)
        self.down_sample = nn.Sequential()
        if stride!=1 or inp!=self.expansion * outp :
            self.down_sample = nn.Sequential(
                nn.Conv2d(inp, outp * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outp * self.expansion),
            )

    def forward(self, x):
        res = self.down_sample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += res
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_plane, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_plane = in_plane
        self.conv1 = nn.Conv2d(in_channels=self.in_plane,
                     kernel_size=5, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)


def resnet(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2], **kwargs)
    return model