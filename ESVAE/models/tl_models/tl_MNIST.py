import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Spiking Jelly
from spikingjelly.activation_based.layer import Linear

# Custom Libraries
import ESVAE.global_v as glv
from ESVAE.tl_utils.loss_function import *
from ESVAE.models import TET__layer

# OpenCV DNN Layer (Only if used later)
from cv2.dnn import Layer  # Only include this if you're using the Layer class

from svae_models.snn_layers import *


class Classifier(nn.Module):
    def __init__(self, latent_dim_class=256, num_classes=10):
        super(Classifier, self).__init__()

        # 第一层：线性变换 -> 批归一化 -> 激活函数 -> 丢弃法
        self.fc1 = nn.Linear(latent_dim_class, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.LeakyReLU(0.01)
        self.dropout1 = nn.Dropout(0.5)

        # 输出层：线性变换
        self.fc2 = nn.Linear(256, num_classes)

        # 可选：添加一个残差连接
        self.residual = nn.Linear(latent_dim_class, num_classes)

    def forward(self, latent):
        x = self.fc1(latent)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        logits = self.fc2(x)

        # 可选：添加残差连接
        residual = self.residual(latent)
        logits += residual

        return logits


class tl_MNIST(nn.Module):
    def __init__(self, device, num_classes=10, latent_dim_class=256):
        super(tl_MNIST, self).__init__()

        self.input_channels = 3  # 设置输入通道数（1: MNIST，3: CIFAR-10）
        self.latent_dim_class = latent_dim_class
        self.num_classes = num_classes

        self.dvs_input = nn.Conv3d(in_channels=2, out_channels=3, kernel_size=1)

        # 初始化分类器
        self.classifier = Classifier(latent_dim_class, num_classes)
        self.device = device

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # 添加DVS通道转换器
        self.dvs_to_rgb_transform = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(3)
        )

        # Build Encoder
        modules = []
        in_channels = self.input_channels
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike(),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        # 添加调整维度的线性层

        self.before_latent_class = tdLinear(1024,
                                            latent_dim_class,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim_class),
                                            spike=LIFSpike())
        # self.before_latent_class_lif_node = LIFSpike()    # 记录电位，用于后续计算

    def encode(self, x):
        x = self.encoder(x)  # (N,C,H,W,T)
        spike_feature = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        latent_class = self.before_latent_class(spike_feature)
        return latent_class  # （N ， latent_dim_class , T）

    # def forward(self, x, labels, scheduled=False):
    #     latent_class = self.encode(x)  # [N, latent_dim_recon, T]
    #
    #
    #     latent_freq_class = torch.sum(latent_class, dim=2) / latent_class.shape[2]  # [N, latent_dim_class]
    #     logits = self.classifier(latent_freq_class)  # [N, num_classes]
    #     return logits

    def forward(self, source, target, encoder_tl_loss_type='CKA', feature_tl_loss_type='CKA'):
        """
        Args:
            source: 源域输入，形状为 (N, T, C, H, W)
            target: 目标域输入，形状为 (N, T, C, H, W)
            encoder_tl_loss_type: 迁移损失类型，包括 'TCKA' 和 'CKA'
            feature_tl_loss_type: 特征迁移损失类型，包括 'TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'
        Returns:
            如果处于训练阶段:
                source_clf: 源域分类输出，形状为 (N, T, class_num)
                target_clf: 目标域分类输出，形状为 (N, T, class_num)
                encoder_tl_loss: 编码器迁移损失，实数
                feature_tl_loss: 特征迁移损失，实数
            如果处于测试阶段:
                target_clf: 目标域分类输出，形状为 (N, T, class_num)
        """

        # 修改DVS数据的通道，与RGB相匹配
        def convert_channels(data):
            if data.shape[2] == 1:  # MNIST 只有 1 个通道
                data = data.expand(-1, -1, 3, -1, -1).contiguous().to(torch.float32)  # 复制 3 份通道 (N, T, 3, H, W)
            elif data.shape[2] == 2:  # DVS 事件数据
                data = data.permute(0, 2, 1, 3, 4).contiguous()  # (N, 2, T, H, W)
                data = self.dvs_input(data)  # 1x1 卷积调整通道
                data = data.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, 3, H, W)
            return data

        # 修改DVS数据的通道，与RGB相匹配
        source = convert_channels(source)
        target = convert_channels(target)

        if self.training:
            batch_size, T, C, H, W = source.shape

            # Transform DVS data if it has 2 channels
            # if target.shape[2] == 3:
            #     target = target.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H, W)
            #     target = target.permute(0, 2, 1, 3, 4).contiguous()  # Back to (N, T, C, H, W)

            # 调整数据维度从 (N, T, C, H, W) 到 (N, C, H, W, T)
            source = source.permute(0, 2, 3, 4, 1).contiguous()  # (N, C, H, W, T)
            target = target.permute(0, 2, 3, 4, 1).contiguous()  # (N, C, H, W, T)

            # 编码源域和目标域输入
            source_latent = self.encode(source)  # (N, latent_dim_class, T)
            target_latent = self.encode(target)  # (N, latent_dim_class, T)

            # 对时间维度进行处理，例如求和或平均
            source_latent_freq = torch.sum(source_latent, dim=2) / source_latent.shape[2]  # (N, latent_dim_class)
            target_latent_freq = torch.sum(target_latent, dim=2) / target_latent.shape[2]  # (N, latent_dim_class)

            # 分类输出
            source_clf = self.classifier(source_latent_freq)  # (N, num_classes)
            target_clf = self.classifier(target_latent_freq)  # (N, num_classes)

            # 扩展分类输出的维度以匹配 (N, T, num_classes)
            source_clf = source_clf.unsqueeze(1).expand(-1, T, -1)  # (N, T, num_classes)
            target_clf = target_clf.unsqueeze(1).expand(-1, T, -1)  # (N, T, num_classes)

            # 计算编码器迁移损失
            # 由于之前注释掉了记录膜电位的代码，这里假设暂时不使用膜电位计算损失
            if encoder_tl_loss_type == 'TCKA':
                # 分别计算各时间步的 CKA，并取平均
                encoder_tl_loss = 1 - temporal_linear_CKA(
                    source_latent.view(batch_size, T, -1),
                    target_latent.view(batch_size, T, -1)
                )
            elif encoder_tl_loss_type == 'CKA':
                # 将各时间步的 spike 频率求和后计算 CKA
                encoder_tl_loss = 1 - linear_CKA(
                    source_latent.view(batch_size, -1),
                    target_latent.view(batch_size, -1),
                    "SUM"
                )
            else:
                raise ValueError(f"Unsupported encoder_tl_loss_type: {encoder_tl_loss_type}")
            # 计算特征迁移损失
            if feature_tl_loss_type == 'TMSE':
                feature_tl_loss = temporal_MSE(source_latent, target_latent)
            elif feature_tl_loss_type == 'MSE':
                feature_tl_loss = MSE(source_latent, target_latent, "SUM")
            elif feature_tl_loss_type == 'CKA':
                feature_tl_loss = 1 - linear_CKA(source_latent, target_latent, "SUM")
            elif feature_tl_loss_type == 'TCKA':
                feature_tl_loss = 1 - temporal_linear_CKA(source_latent, target_latent)
            elif feature_tl_loss_type == 'MMD':
                feature_tl_loss = MMD_loss(source_latent, target_latent, "SUM")
            else:
                raise ValueError(f"Unsupported feature_tl_loss_type: {feature_tl_loss_type}")

            return source_clf, target_clf, encoder_tl_loss, feature_tl_loss
        else:
            # 测试阶段只处理目标域输入
            batch_size, T, C, H, W = target.shape

            # 调整数据维度从 (N, T, C, H, W) 到 (N, C, H, W, T)
            target = target.permute(0, 2, 3, 4, 1).contiguous()  # (N, C, H, W, T)

            # 编码目标域输入
            target_latent = self.encode(target)  # (N, latent_dim_class, T)
            # target_latent = self.before_latent_class_lif_node(target_latent)

            # 对时间维度进行处理，例如求和或平均
            target_latent_freq = torch.sum(target_latent, dim=2) / target_latent.shape[2]  # (N, latent_dim_class)

            target_clf = self.classifier(target_latent_freq)  # (N, num_classes)

            # 扩展分类输出的维度以匹配 (N, T, num_classes)
            target_clf = target_clf.unsqueeze(1).expand(-1, T, -1)  # (N, T, num_classes)   # 用于目标函数计算损失

            return target_clf
