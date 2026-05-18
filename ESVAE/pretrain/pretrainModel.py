# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training
"""
# 移除错误的cv2.dnn.Layer导入，使用正确的SNN Layer
import models.TET__layer
from tl_utils.loss_function import *
from pretrain.Edge import *
from models.TET__layer import *




class VGGSNN(nn.Module):
    def __init__(self, cls_num=10, img_shape=48, device='cuda', use_event_attention=False, event_attention_reduction=8):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)  # RGB 3通道输入
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # DVS/Edge 2通道输入
        
        # 事件注意力开关
        self.use_event_attention = use_event_attention
        
        # 如果启用事件注意力，在前两层添加注意力模块
        if self.use_event_attention:
            from models.TET__layer import EventMidFrameAttention
            self.event_attn_1 = EventMidFrameAttention(64, reduction=event_attention_reduction)  # dvs_input后
            self.event_attn_2 = EventMidFrameAttention(128, reduction=event_attention_reduction)  # features[0]后

        self.features = nn.Sequential(
            Layer(64, 128, 3, 1, 1, False),
            pool,
            Layer(128, 256, 3, 1, 1, False),
            Layer(256, 256, 3, 1, 1, False),
            pool,
            Layer(256, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 1, 1, False),
            pool,
            Layer(512, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 1, 1, False),
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
            source: 源域输入，(N, T, C, H, W) C可以是3（RGB）或2（Edge）
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

            # 根据通道数选择输入层
            # 处理源域输入 (3通道RGB 或 2通道Edge)
            if source.shape[2] == 3:
                source, source_mem = self.rgb_input(source)
            elif source.shape[2] == 2:
                source, source_mem = self.dvs_input(source)
            else:
                raise ValueError(f"Unexpected source channel number: {source.shape[2]}, expected 2 or 3")
            
            # 处理目标域输入 (2通道DVS)
            if target.shape[2] == 2:
                target, target_mem = self.dvs_input(target)
            else:
                raise ValueError(f"Unexpected target channel number: {target.shape[2]}, expected 2")
            
            # 【新增】第1处事件注意力：dvs_input 之后
            if self.use_event_attention:
                target = self.event_attn_1(target)

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
            # Source分支正常走
            source = self.features(source)
            source = torch.flatten(source, 2)
            source = self.bottleneck(source)
            source, source_mem = self.bottleneck_lif_node(source, return_mem=True)
            source_clf = self.classifier(source)

            # Target(DVS)分支：在features第一层后添加第2处注意力
            target = self.features[0](target)  # Layer(64, 128, ...)
            
            # 【新增】第2处事件注意力：features[0] 之后
            if self.use_event_attention:
                target = self.event_attn_2(target)
            
            # 继续后续层
            for layer in self.features[1:]:
                target = layer(target)
            
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
            
            # 【新增】测试时也要走相同的事件注意力路径
            if self.use_event_attention:
                target = self.event_attn_1(target)
            
            # features第一层
            target = self.features[0](target)
            
            # 【新增】第2处注意力
            if self.use_event_attention:
                target = self.event_attn_2(target)
            
            # 继续后续层
            for layer in self.features[1:]:
                target = layer(target)
            
            target = torch.flatten(target, 2)
            target = self.bottleneck(target)
            target = self.bottleneck_lif_node(target)
            target_clf = self.classifier(target)
            return target_clf


class VGGSNNwoAP(VGGSNN):
    def __init__(self, cls_num=10, img_shape=32, use_event_attention=False, event_attention_reduction=8):
        """
        VGGSNNwoAP: 使用stride=2卷积替代平均池化
        
        Args:
            cls_num: 分类数量，默认10（CIFAR10）
            img_shape: 输入图像大小，默认32（与tl.py保持一致，CIFAR10使用32×32）
            use_event_attention: 是否启用事件注意力
            event_attention_reduction: 事件注意力通道压缩比
        
        注意：完全复制tl.py的bottleneck设计
        - bottleneck包含Linear+LIFSpike
        - bottleneck_lif_node继承自父类（会导致两次LIFSpike激活）
        - 这与tl.py的实现完全一致
        """
        super(VGGSNNwoAP, self).__init__(cls_num=cls_num, img_shape=img_shape, device='cuda',
                                         use_event_attention=use_event_attention,
                                         event_attention_reduction=event_attention_reduction)
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)  # RGB 3通道输入
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # DVS/Edge 2通道输入
        self.features = nn.Sequential(
            Layer(64, 128, 3, 2, 1, False),
            Layer(128, 256, 3, 1, 1, False),
            Layer(256, 256, 3, 2, 1, False),
            Layer(256, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 2, 1, False),
            Layer(512, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 2, 1, False),
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # 关键修复：与tl.py完全一致，LIFSpike放在SeqToANNContainer内部
        # 注意：父类的bottleneck_lif_node仍然存在，会导致两次LIFSpike激活
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256),
                                            LIFSpike())
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
