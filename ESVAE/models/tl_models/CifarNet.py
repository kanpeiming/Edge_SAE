# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training
"""


from models.TET__layer import *
from utils import linear_CKA, temporal_linear_CKA


class CifarNetSNN(nn.Module):
    def __init__(self):
        super(CifarNetSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(3, 2))
        self.source_input = Layer(3, 64, 11, 4, 2)
        self.target_input = Layer(3, 64, 11, 4, 2)
        self.features = nn.Sequential(
            # Layer(2, 64, 3, 1, 1),
            Layer(64, 192, 5, 1, 2),
            pool,
            Layer(192, 256, 3, 1, 1),
            pool,
            Layer(256, 256, 3, 1, 1),
            pool,
        )
        # W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 4
        self.fc1 = SeqToANNContainer(
            nn.Linear(9216, 2048),
            LIFSpike()
        )
        self.bottleneck = SeqToANNContainer(
            nn.Linear(2048, 256),
        )
        self.bottleneck_lif_node = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, source, target, CKA_type='spike'):
        if self.training:
            source = self.source_input(source)
            source = self.features(source)
            source = torch.flatten(source, 2)
            source = self.fc1(source)
            source = self.bottleneck(source)  # (N, T, 256)
            source_lif = self.bottleneck_lif_node(source)
            source_clf = self.classifier(source_lif)

            target = self.target_input(target)
            target = self.features(target)
            target = torch.flatten(target, 2)
            target = self.fc1(target)
            target = self.bottleneck(target)  # (N, T, 256)
            target_lif = self.bottleneck_lif_node(target)
            target_clf = self.classifier(target_lif)

            if CKA_type == 'mem':
                tl_loss = 1 - linear_CKA(source, target, "FLATTEN")
            elif CKA_type == 'spike':
                tl_loss = 1 - linear_CKA(source_lif, target_lif, "SUM")
            return source_clf, target_clf, tl_loss
        else:
            target = self.target_input(target)
            target = self.features(target)
            target = torch.flatten(target, 2)
            target = self.fc1(target)
            target = self.bottleneck(target)
            target_lif = self.bottleneck_lif_node(target)
            target_clf = self.classifier(target_lif)
            return target_clf


class CifarNetSNNwoAP(CifarNetSNN):
    def __init__(self):
        super(CifarNetSNNwoAP, self).__init__()
        self.source_input = Layer(3, 64, 11, 4, 2)
        self.target_input = Layer(3, 64, 11, 4, 2)
        self.features = nn.Sequential(
            Layer(64, 192, 5, 1, 2),
            Layer(192, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
        )

        self.fc1 = SeqToANNContainer(
            nn.Linear(9216, 2048),
            LIFSpike()
        )
        self.bottleneck = SeqToANNContainer(
            nn.Linear(2048, 256),
        )
        self.bottleneck_lif_node = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
