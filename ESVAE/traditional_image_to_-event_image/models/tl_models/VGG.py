# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training
"""

from models.TET__layer import *
from utils import linear_CKA, temporal_linear_CKA, MSE, temporal_MSE, MMD_loss


class VGGSNN(nn.Module):
    def __init__(self, cls_num=10, img_shape=48):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)
        self.features = nn.Sequential(
            # Layer(2, 64, 3, 1, 1),
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
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        self.bottleneck_lif_node = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, source, target, encoder_tl_loss_type='TCKA', feature_tl_loss_type='MSE'):
        """
        Args:
            source: 源域输入，(N, T, 3, H, W) 为 RGB图像，(N, T, 2, H, W)为DVS图像
            target: 目标域输入，(N, T, 3, H, W) 为 RGB图像，(N, T, 2, H, W)为DVS图像
            encoder_tl_loss_type: 包括 'TCKA'(分别计算各时间步mem的CKA，最后求平均), 'CKA'(将各时间步的spike求频率后计算CKA)
            feature_tl_loss_type: 包括 'TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'.
        Returns:
            if self.training:  训练阶段
                Returns:
                    source_clf: 源域分类神经元输出，0/1脉冲序列, (N, T, class_num)
                    target_clf: 目标域分类神经元输出，0/1脉冲序列, (N, T, class_num)
                    encoder_tl_loss: 编码器计算的迁移损失，实数
                    feature_tl_loss: 提取特征计算的迁移损失，实数
            else:  测试阶段
                Returns:
                    target_clf: 目标域分类神经元输出，0/1脉冲序列, (N, T, class_num)
        """
        if self.training:
            batch_size, T = source.shape[0:2]
            # source input encoder
            if source.shape[2] == 3:  # (N, T, 3, H, W)
                source, source_mem = self.rgb_input(source)
            else:
                source, source_mem = self.dvs_input(source)

            # target input encoder
            if target.shape[2] == 3:
                target, target_mem = self.rgb_input(target)
            else:
                target, target_mem = self.dvs_input(target)

            if encoder_tl_loss_type == 'TCKA':
                # tl_loss = 1 - linear_CKA(source, target, "FLATTEN")
                encoder_tl_loss = 1 - temporal_linear_CKA(source_mem.view((batch_size, T, -1)),
                                                          target_mem.view((batch_size, T, -1)))
            elif encoder_tl_loss_type == 'CKA':
                encoder_tl_loss = 1 - linear_CKA(source.view((batch_size, T, -1)),
                                                 target.view((batch_size, T, -1)), "SUM")
            else:
                raise Exception(f"The value of encoder_tl_loss_type should in ['TCKA', 'CKA'], "
                                f"and your input is {encoder_tl_loss_type}")

            source = self.features(source)
            source = torch.flatten(source, 2)
            source = self.bottleneck(source)  # (N, T, 256)
            source, source_mem = self.bottleneck_lif_node(source, return_mem=True)
            source_clf = self.classifier(source)

            target = self.features(target)
            target = torch.flatten(target, 2)
            target = self.bottleneck(target)  # (N, T, 256)
            target, target_mem = self.bottleneck_lif_node(target, return_mem=True)
            target_clf = self.classifier(target)

            if feature_tl_loss_type == 'TMSE':
                feature_tl_loss = temporal_MSE(source_mem, target_mem)
            elif feature_tl_loss_type == 'MSE':
                feature_tl_loss = MSE(source, target, "SUM")
            elif feature_tl_loss_type == 'CKA':
                feature_tl_loss = 1 - linear_CKA(source, target, "SUM")
            elif feature_tl_loss_type == 'TCKA':
                feature_tl_loss = 1 - temporal_linear_CKA(source_mem, target_mem)
            elif feature_tl_loss_type == 'MMD':
                feature_tl_loss = MMD_loss(source, target, "SUM")
            else:
                raise Exception(f"The value of feature_tl_loss_type should in ['TMSE', 'MSE', 'TCKA', 'CKA', 'MMD'], "
                                f"and your input is {feature_tl_loss_type}")

            return source_clf, target_clf, encoder_tl_loss, feature_tl_loss
        else:
            if target.shape[2] == 3:
                target, _ = self.rgb_input(target)
            else:
                target, _ = self.dvs_input(target)
            target = self.features(target)
            target = torch.flatten(target, 2)
            target = self.bottleneck(target)
            target = self.bottleneck_lif_node(target)
            target_clf = self.classifier(target)
            return target_clf


class VGGSNNwoAP(VGGSNN):
    def __init__(self, cls_num=10, img_shape=48):
        super(VGGSNNwoAP, self).__init__(cls_num, img_shape)
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)
        self.features = nn.Sequential(
            # Layer(2, 64, 3, 1, 1),
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
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256),
                                            LIFSpike())
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
