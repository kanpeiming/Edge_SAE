# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: enhanced_model.py
@time: 2024/10/27
@description: 集成多层级区域关注的增强版VGGSNN模型
"""

import sys
import os
# 添加父目录到路径，以便导入ESVAE模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TET__layer import *
from tl_utils.loss_function import *
from regional_focus_module import EnhancedVGGSNNWithRegionalFocus, RegionalFocusModule


class VGGSNN_RegionalFocus(EnhancedVGGSNNWithRegionalFocus):
    """
    增强版VGGSNN：集成多层级区域聚焦机制
    
    关键改进：
    1. DVS输入从2通道改为3通道（两通道各取0.5堆叠到第三通道）
    2. 在主干网每一层都施加区域关注约束来引导对齐
    3. 使用相同的目标函数进行各层级对齐
    4. 兼容原有的训练接口
    """
    
    def __init__(self, 
                 cls_num=10, 
                 img_shape=32, 
                 device='cuda',
                 use_regional_focus=True,
                 regional_focus_config=None):
        
        # 处理配置参数
        if regional_focus_config is None:
            regional_focus_config = {
                'similarity_type': 'cosine',
                'weight_constraint': 'softmax',
                'alpha': 1.0,
                'beta': 0.1,
            }
        
        # 调用父类构造函数，传入相应参数
        super(VGGSNN_RegionalFocus, self).__init__(
            cls_num=cls_num,
            img_shape=img_shape,
            enable_hierarchical_focus=use_regional_focus,
            focus_similarity_type=regional_focus_config.get('similarity_type', 'cosine'),
            focus_weight_constraint=regional_focus_config.get('weight_constraint', 'softmax'),
            regional_loss_weight=regional_focus_config.get('alpha', 0.5)
        )
        
        self.device = device
        self.regional_focus_config = regional_focus_config


class EnhancedVGGSNNWithRegionalFocuswoAP(nn.Module):
    """
    增强的VGGSNN without Average Pooling：集成分层区域聚焦机制
    
    关键特点：
    1. 使用stride=2的卷积层替代AvgPool2d进行下采样
    2. 瓶颈层直接集成LIFSpike
    3. 在主干网每一层都施加区域关注约束来引导对齐
    4. 默认支持48x48输入图像尺寸
    """
    
    def __init__(self, 
                 cls_num=10, 
                 img_shape=48,
                 enable_hierarchical_focus=True,
                 focus_similarity_type='cosine',
                 focus_weight_constraint='softmax',
                 regional_loss_weight=0.5):
        super(EnhancedVGGSNNWithRegionalFocuswoAP, self).__init__()
        
        # 导入必要的层
        from TET__layer import Layer, SeqToANNContainer, LIFSpike
        
        # 输入层：RGB是3通道，DVS是2通道，都映射到64通道
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # DVS是2通道输入
        
        # 分层特征提取（使用stride=2替代池化）
        self.layer1 = Layer(64, 128, 3, 2, 1)    # stride=2替代pool
        self.layer2 = Layer(128, 256, 3, 1, 1)
        self.layer3 = Layer(256, 256, 3, 2, 1)   # stride=2替代pool
        self.layer4 = Layer(256, 512, 3, 1, 1)
        self.layer5 = Layer(512, 512, 3, 2, 1)   # stride=2替代pool
        self.layer6 = Layer(512, 512, 3, 1, 1)
        self.layer7 = Layer(512, 512, 3, 2, 1)   # stride=2替代pool
        
        W = int(img_shape / 2 / 2 / 2 / 2)
        # woAP版本的瓶颈层：直接集成LIFSpike
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256), LIFSpike())
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))
        
        # 分层区域聚焦模块
        self.enable_hierarchical_focus = enable_hierarchical_focus
        self.regional_loss_weight = regional_loss_weight
        
        if enable_hierarchical_focus:
            # 为每一层创建区域关注模块（适配woAP的特征图尺寸）
            self.regional_focus_modules = nn.ModuleList([
                RegionalFocusModule(64, (img_shape, img_shape), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),      # 输入层后
                RegionalFocusModule(128, (img_shape//2, img_shape//2), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),  # layer1后
                RegionalFocusModule(256, (img_shape//4, img_shape//4), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),  # layer3后
                RegionalFocusModule(512, (img_shape//8, img_shape//8), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),  # layer5后
                RegionalFocusModule(512, (img_shape//16, img_shape//16), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint), # layer7后
                RegionalFocusModule(256, (1, 1), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint)                          # 瓶颈层
            ])
            
            # 各层权重（可学习）
            self.layer_weights = nn.Parameter(torch.tensor([0.8, 1.0, 1.2, 1.5, 1.8, 2.0]))
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def extract_hierarchical_features(self, x, domain_type='rgb'):
        """
        提取分层特征用于区域关注（woAP版本）
        
        Args:
            x: 输入数据 (N, T, C, H, W) - RGB是3通道，DVS是2通道
            domain_type: 'rgb' 或 'dvs'
        
        Returns:
            features: 各层特征列表
            x_final: 最终特征
            x_mem: 瓶颈层记忆（woAP版本中由bottleneck直接返回）
        """
        features = []
        
        # 输入编码 - RGB是3通道，DVS是2通道
        if domain_type == 'rgb':
            x, _ = self.rgb_input(x)
        else:
            x, _ = self.dvs_input(x)
        features.append(x)  # 64维特征
        
        # 逐层提取特征（woAP版本使用stride=2替代池化）
        x = self.layer1(x)  # stride=2
        features.append(x)  # 128维特征
        
        x = self.layer2(x)
        x = self.layer3(x)  # stride=2
        features.append(x)  # 256维特征
        
        x = self.layer4(x)
        x = self.layer5(x)  # stride=2
        features.append(x)  # 512维特征
        
        x = self.layer6(x)
        x = self.layer7(x)  # stride=2
        features.append(x)  # 512维特征
        
        # 瓶颈层 - woAP版本直接flatten
        x_before_flatten = x  # 保存flatten前的特征用于区域关注
        
        # 直接flatten
        x = torch.flatten(x, 2)
        
        # woAP版本的瓶颈层已经集成了LIFSpike，直接调用
        x = self.bottleneck(x)
        # 注意：woAP版本的bottleneck已经包含LIFSpike，无法直接获取mem
        # 这里我们假设x就是最终输出，x_mem设为None或x的副本
        x_mem = x  # woAP版本中无法分离mem，使用输出作为替代
        
        return features, x, x_mem
    
    def compute_regional_alignment_loss(self, source_features, target_features):
        """
        计算各层的区域对齐损失（woAP版本）
        使用相同的目标函数（MSE + 区域权重）
        
        Args:
            source_features: 源域各层特征列表
            target_features: 目标域各层特征列表
        
        Returns:
            total_regional_loss: 总的区域对齐损失
            layer_losses: 各层损失详情
        """
        if not self.enable_hierarchical_focus:
            return torch.tensor(0.0), []
        
        layer_losses = []
        layer_weights_norm = F.softmax(self.layer_weights, dim=0)
        
        for i, (source_feat, target_feat) in enumerate(zip(source_features, target_features)):
            # 计算区域权重
            regional_weights = self.regional_focus_modules[i](source_feat, target_feat)
            
            # 计算加权MSE损失（统一的目标函数）
            mse_loss_map = ((source_feat - target_feat) ** 2).mean(dim=1, keepdim=True)  # (N, 1, H, W)
            weighted_loss = (mse_loss_map * regional_weights).mean()
            
            # 应用层级权重
            layer_loss = layer_weights_norm[i] * weighted_loss
            layer_losses.append(layer_loss)
        
        # 总损失
        total_regional_loss = sum(layer_losses)
        
        return total_regional_loss, layer_losses
    
    def forward(self, source, target, encoder_tl_loss_type='TCKA', feature_tl_loss_type='MSE'):
        """
        增强的前向传播，集成分层区域关注（woAP版本）
        
        Args:
            source: 源域输入 (N, T, C, H, W)
            target: 目标域输入 (N, T, C, H, W)
            encoder_tl_loss_type: 编码器损失类型
            feature_tl_loss_type: 特征损失类型
        
        Returns:
            训练时: (source_clf, target_clf, encoder_tl_loss, feature_tl_loss)
            测试时: target_clf
        """
        if self.training:
            batch_size, T = source.shape[0:2]
            
            # 提取分层特征
            source_features, source_final, source_mem = self.extract_hierarchical_features(
                source, 'rgb' if source.shape[2] == 3 else 'dvs'
            )
            target_features, target_final, target_mem = self.extract_hierarchical_features(
                target, 'dvs' if target.shape[2] == 2 else ('rgb' if target.shape[2] == 3 else 'dvs')
            )
            
            # 计算分层区域对齐损失（主要改进）
            regional_loss, layer_losses = self.compute_regional_alignment_loss(
                source_features, target_features
            )
            
            # 原有的编码器损失（在输入层）
            if encoder_tl_loss_type == 'TCKA':
                from loss_function import temporal_linear_CKA
                # 使用输入层的记忆状态 - RGB是3通道，DVS是2通道
                if source.shape[2] == 3:
                    _, source_encoded_mem = self.rgb_input(source)
                else:
                    _, source_encoded_mem = self.dvs_input(source)
                
                if target.shape[2] == 3:
                    _, target_encoded_mem = self.rgb_input(target)
                else:
                    _, target_encoded_mem = self.dvs_input(target)
                
                encoder_tl_loss = 1 - temporal_linear_CKA(
                    source_encoded_mem.view((batch_size, T, -1)),
                    target_encoded_mem.view((batch_size, T, -1))
                )
            elif encoder_tl_loss_type == 'CKA':
                from loss_function import linear_CKA
                encoder_tl_loss = 1 - linear_CKA(
                    source_features[0].view((batch_size, T, -1)),
                    target_features[0].view((batch_size, T, -1)), "SUM"
                )
            else:
                encoder_tl_loss = torch.tensor(0.0)
            
            # 原有的特征损失（在瓶颈层）
            if feature_tl_loss_type == 'TMSE':
                from loss_function import temporal_MSE
                feature_tl_loss = temporal_MSE(source_mem, target_mem)
            elif feature_tl_loss_type == 'MSE':
                from loss_function import MSE
                feature_tl_loss = MSE(source_final, target_final, "SUM")
            elif feature_tl_loss_type == 'CKA':
                from loss_function import linear_CKA
                feature_tl_loss = 1 - linear_CKA(source_final, target_final, "SUM")
            elif feature_tl_loss_type == 'TCKA':
                from loss_function import temporal_linear_CKA
                feature_tl_loss = 1 - temporal_linear_CKA(source_mem, target_mem)
            elif feature_tl_loss_type == 'MMD':
                from loss_function import MMD_loss
                feature_tl_loss = MMD_loss(source_final, target_final, "SUM")
            else:
                feature_tl_loss = torch.tensor(0.0)
            
            # 分类输出
            source_clf = self.classifier(source_final)
            target_clf = self.classifier(target_final)
            
            # 将区域损失合并到特征损失中
            weighted_regional_loss = regional_loss * self.regional_loss_weight
            total_feature_loss = feature_tl_loss + weighted_regional_loss
            
            return source_clf, target_clf, encoder_tl_loss, total_feature_loss
        
        else:
            # 测试阶段：直接处理DVS 2通道或RGB 3通道
            target_features, target_final, _ = self.extract_hierarchical_features(
                target, 'dvs' if target.shape[2] == 2 else 'rgb'
            )
            target_clf = self.classifier(target_final)
            return target_clf


class VGGSNN_RegionalFocuswoAP(EnhancedVGGSNNWithRegionalFocuswoAP):
    """
    增强版VGGSNN without Average Pooling：集成多层级区域聚焦机制
    
    关键改进：
    1. 使用stride=2的卷积层替代AvgPool2d进行下采样
    2. 瓶颈层直接集成LIFSpike
    3. 在主干网每一层都施加区域关注约束来引导对齐
    4. 兼容原有的训练接口
    """
    
    def __init__(self, 
                 cls_num=10, 
                 img_shape=48, 
                 device='cuda',
                 use_regional_focus=True,
                 regional_focus_config=None):
        
        # 处理配置参数
        if regional_focus_config is None:
            regional_focus_config = {
                'similarity_type': 'cosine',
                'weight_constraint': 'softmax',
                'alpha': 1.0,
                'beta': 0.1,
            }
        
        # 调用父类构造函数，传入相应参数
        super(VGGSNN_RegionalFocuswoAP, self).__init__(
            cls_num=cls_num,
            img_shape=img_shape,
            enable_hierarchical_focus=use_regional_focus,
            focus_similarity_type=regional_focus_config.get('similarity_type', 'cosine'),
            focus_weight_constraint=regional_focus_config.get('weight_constraint', 'softmax'),
            regional_loss_weight=regional_focus_config.get('alpha', 0.5)
        )
        
        self.device = device
        self.regional_focus_config = regional_focus_config
    
    def forward(self, source, target, encoder_tl_loss_type='TCKA', feature_tl_loss_type='MSE'):
        """
        前向传播，兼容原有接口
        
        Args:
            source: 源域输入 (N, T, C, H, W)
            target: 目标域输入 (N, T, C, H, W)
            encoder_tl_loss_type: 编码器损失类型
            feature_tl_loss_type: 特征损失类型
        
        Returns:
            训练时: (source_clf, target_clf, encoder_tl_loss, feature_tl_loss)
            测试时: target_clf
        """
        # 调用父类的前向传播（已经包含了区域损失）
        result = super().forward(source, target, encoder_tl_loss_type, feature_tl_loss_type)
        
        # 直接返回结果，区域损失已经在父类中处理
        return result

    
# 为了向后兼容，保留原有的接口
def create_regional_focus_model(cls_num=10, img_shape=32, device='cuda', 
                               use_regional_focus=True, regional_focus_config=None,
                               use_woap=False):
    """
    创建区域关注模型的工厂函数
    
    Args:
        cls_num: 分类数
        img_shape: 输入图像尺寸
        device: 设备
        use_regional_focus: 是否使用区域关注
        regional_focus_config: 区域关注配置
        use_woap: 是否使用without Average Pooling版本
    
    Returns:
        model: 增强的VGGSNN模型
    """
    if use_woap:
        return VGGSNN_RegionalFocuswoAP(
            cls_num=cls_num,
            img_shape=img_shape,
            device=device,
            use_regional_focus=use_regional_focus,
            regional_focus_config=regional_focus_config
        )
    else:
        return VGGSNN_RegionalFocus(
            cls_num=cls_num,
            img_shape=img_shape,
            device=device,
            use_regional_focus=use_regional_focus,
            regional_focus_config=regional_focus_config
        )


# 导出
__all__ = ['VGGSNN_RegionalFocus', 'VGGSNN_RegionalFocuswoAP', 'EnhancedVGGSNNWithRegionalFocuswoAP', 'create_regional_focus_model']