# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training
"""

from models.TET__layer import *


class VGGSNN(nn.Module):
    def __init__(self, in_channel=2, cls_num=10, img_shape=48):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(in_channel, 64, 3, 1, 1),
            Layer(64, 128, 3, 1, 1),
            pool,
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            pool,
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGSNNwoAP(nn.Module):
    def __init__(self, in_channel=2, cls_num=10, img_shape=48):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(in_channel, 64, 3, 1, 1),
            Layer(64, 128, 3, 2, 1),
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 2, 1),
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class AttentionVGGSNNwoAP(nn.Module):
    def __init__(self, T):
        super(AttentionVGGSNNwoAP, self).__init__()

        self.rgb_input_layer = Layer(3, 64, 3, 1, 1)
        self.dvs_input_layer = Layer(2, 64, 3, 1, 1)

        self.features = nn.Sequential(
            LayerWithAttention(64, 128, 3, 2, 1, T, 'T'),
            LayerWithAttention(128, 256, 3, 1, 1, T, 'T'),
            LayerWithAttention(256, 256, 3, 2, 1, T, 'T'),
            LayerWithAttention(256, 512, 3, 1, 1, T, 'T'),
            LayerWithAttention(512, 512, 3, 2, 1, T, 'T'),
            LayerWithAttention(512, 512, 3, 1, 1, T, 'T'),
            LayerWithAttention(512, 512, 3, 2, 1, T, 'T'),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, source, target):
        if source.shape[2] == 3:
            source_x = self.rgb_input_layer(source)
        elif source.shape[2] == 2:
            source_x = self.dvs_input_layer(source)
        target_x = self.dvs_input_layer(target)

        source_x, target_x = self.features((source_x, target_x))
        source_x = torch.flatten(source_x, 2)
        target_x = torch.flatten(target_x, 2)

        source_x = self.classifier(source_x)
        target_x = self.classifier(target_x)
        return source_x, target_x

