# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training
"""
from cv2.dnn import Layer
import models.TET__layer
from tl_utils.loss_function import *
from pretrain.Edge import *
from models.TET__layer import *




class VGGSNN(nn.Module):
    def __init__(self, cls_num=10, img_shape=32, device='cuda'):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # DVS 2通道输入

        self.features = nn.Sequential(
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
        # 传统瓶颈层设计，基于动态计算的特征图尺寸
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        self.bottleneck_lif_node = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, source, target, encoder_tl_loss_type='TCKA', feature_tl_loss_type='MSE'):
        """
        Args:
            source: 源域输入，(N, T, 3, H, W) 为RGB图像
            target: 目标域输入，(N, T, 2, H, W) 为DVS数据
            encoder_tl_loss_type: 包括 'TCKA'(分别计算各时间步mem的CKA，最后求平均), 'CKA'(将各时间步的spike求频率后计算CKA)
            feature_tl_loss_type: 包括 'TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'.
        Returns:
            Returns loss values during training:
                encoder_tl_loss: 编码器计算的迁移损失
                feature_tl_loss: 提取特征计算的迁移损失
        """
        # print("Source shape:", source.shape)
        # print("Target shape:", target.shape)

        if self.training:
            batch_size, T = source.shape[0:2]

            # 处理 RGB 输入 (3通道)
            source, source_mem = self.rgb_input(source)
            
            # 处理 DVS 输入 (2通道)
            target, target_mem = self.dvs_input(target)

            # 计算编码器迁移损失
            if encoder_tl_loss_type == 'TCKA':
                encoder_tl_loss = 1 - temporal_linear_CKA(source_mem.view((batch_size, T, -1)),
                                                          target_mem.view((batch_size, T, -1)))
            elif encoder_tl_loss_type == 'CKA':
                encoder_tl_loss = 1 - linear_CKA(source.view((batch_size, T, -1)),
                                                 target.view((batch_size, T, -1)), "SUM")
            else:
                raise Exception(f"Invalid encoder_tl_loss_type: {encoder_tl_loss_type}")

            # 提取高层特征
            source = self.features(source)
            source = torch.flatten(source, 2)
            source = self.bottleneck(source)
            source, source_mem = self.bottleneck_lif_node(source, return_mem=True)
            source_clf = self.classifier(source)

            target = self.features(target)
            target = torch.flatten(target, 2)
            target = self.bottleneck(target)
            target, target_mem = self.bottleneck_lif_node(target, return_mem=True)
            target_clf = self.classifier(target)

            # 计算特征迁移损失
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
                raise Exception(f"Invalid feature_tl_loss_type: {feature_tl_loss_type}")

            return source_clf, target_clf, encoder_tl_loss, feature_tl_loss  # 返回迁移损失,预训练阶段前两个返回值不需要

        else:
            # 测试模式：只处理目标域（DVS）数据
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
    def __init__(self, cls_num=10, img_shape=32):
        """
        VGGSNNwoAP: 使用stride=2卷积替代平均池化
        
        Args:
            cls_num: 分类数量，默认10（CIFAR10）
            img_shape: 输入图像大小，默认32（与tl.py保持一致使用32x32）
        
        注意：与tl.py保持一致的bottleneck设计（LIFSpike独立）
        """
        super(VGGSNNwoAP, self).__init__(cls_num=cls_num, img_shape=img_shape, device='cuda')
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)  # 保持True以记录mem
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # 保持True以记录mem
        self.features = nn.Sequential(
            Layer(64, 128, 3, 2, 1),
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 2, 1),
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # 修复：与tl.py保持一致，LIFSpike作为独立层
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        self.bottleneck_lif_node = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
