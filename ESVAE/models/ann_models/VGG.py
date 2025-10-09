# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/22 10:20
"""

from unicodedata import numeric
import torch.nn as nn
from spikingjelly.clock_driven import neuron

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ],
    'TETVGG': [
        [(128, 2), 'A'],
        [(256, 1), (256, 2), 'A'],
        [(512, 1), (512, 2), 'A'],
        [(512, 1), (512, 2), 'A']
    ],
    'TETVGGwoAP': [
        [(128, 2)],
        [(256, 1), (256, 2)],
        [(512, 1), (512, 2)],
        [(512, 1), (512, 2)]
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG, self).__init__()
        self.rgb_layer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.dvs_layer1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.in_channels = 64

        self.layer2 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][3], dropout)

        W = int(48 / 2 / 2 / 2 / 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * W * W, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif x == 'A':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.in_channels, x[0], kernel_size=3, padding=1, stride=x[1]))
                layers.append(nn.BatchNorm2d(x[0]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.in_channels = x[0]
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] == 3:
            # print('rgb')
            out = self.rgb_layer1(x)
        elif x.shape[1] == 2:
            # print('dvs')
            out = self.dvs_layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        return out


class VGGDVS(VGG):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGGDVS, self).__init__(vgg_name, num_classes, dropout)
        self.avgpool2d = nn.AvgPool2d(2)
        self.neuron = neuron.IFNode(v_reset=None)

    def forward(self, x):  # x.shape = [batch, 3, 128, 128]
        x = self.avgpool2d(self.avgpool2d(x))  # x.shape = [batch, 3, 32, 32]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        # out = self.neuron(out)
        return out

    def reset(self):
        self.neuron.reset()


class VGGDVS2(VGG):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGGDVS2, self).__init__(vgg_name, num_classes, dropout)
        self.init_channels = 2
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

        self.avgpool2d = nn.AvgPool2d(2)

    def forward(self, x):  # x.shape = [batch, 3, 128, 128]
        x = self.avgpool2d(self.avgpool2d(x))  # x.shape = [batch, 3, 32, 32]
        # print('encoder: ', x[0] - x[1])
        out = self.layer1(x)
        # print('layer1: ', x[0] - x[1])
        out = self.layer2(out)
        # print('layer2: ', x[0] - x[1])
        out = self.layer3(out)
        # print('layer3: ', x[0] - x[1])
        out = self.layer4(out)
        # print('layer4: ', out[0], '\n', out[1])
        out = self.layer5(out)
        # print('layer5: ', out[0], '\n', out[1])
        out = self.fc1(out)
        # print('fc1: ', out[0], '\n', out[1])
        out = self.fc2(out)
        # out = self.classifier(out)
        # out = self.neuron(out)
        return out


class VGG_normed(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_normed, self).__init__()
        self.num_classes = num_classes
        self.module_list = self._make_layers(cfg[vgg_name], dropout)

    def _make_layers(self, cfg, dropout):
        layers = []
        for i in range(5):
            for x in cfg[i]:
                if x == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.Conv2d(3, x, kernel_size=3, padding=1))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout))
                    self.init_channels = x
        layers.append(nn.Flatten())
        if self.num_classes == 1000:
            layers.append(nn.Linear(512*7*7, 4096))
        else:
            layers.append(nn.Linear(512, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module_list(x)


def vgg11(num_classes=10, dropout=0, **kargs):
    return VGG('VGG11', num_classes, dropout)


def vgg13(num_classes=10, dropout=0, **kargs):
    return VGG('VGG13', num_classes, dropout)


def vgg16(num_classes=10, dropout=0, **kargs):
    return VGG('VGG16', num_classes, dropout)


def vgg19(num_classes=10, dropout=0, **kargs):
    return VGG('VGG19', num_classes, dropout)


def tet_vgg(num_classes=10, dropout=0, **kargs):
    return VGG('TETVGG', num_classes, dropout)


def tet_vgg_woap(num_classes=10, dropout=0, **kargs):
    return VGG('TETVGGwoAP', num_classes, dropout)


def vgg16_normed(num_classes=10, dropout=0, **kargs):
    return VGG_normed('VGG16', num_classes, dropout)


def vgg16_dvs(num_classes=10, dropout=0, **kargs):
    return VGGDVS('VGG16', num_classes, dropout)


def vgg16_dvs_2(num_classes=10, dropout=0, **kargs):
    return VGGDVS2('VGG16', num_classes, dropout)
