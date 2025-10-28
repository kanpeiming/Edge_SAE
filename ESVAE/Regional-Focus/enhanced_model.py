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
from regional_focus_module import EnhancedVGGSNNWithRegionalFocus


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
                               use_regional_focus=True, regional_focus_config=None):
    """
    创建区域关注模型的工厂函数
    
    Args:
        cls_num: 分类数
        img_shape: 输入图像尺寸
        device: 设备
        use_regional_focus: 是否使用区域关注
        regional_focus_config: 区域关注配置
    
    Returns:
        model: 增强的VGGSNN模型
    """
    return VGGSNN_RegionalFocus(
        cls_num=cls_num,
        img_shape=img_shape,
        device=device,
        use_regional_focus=use_regional_focus,
        regional_focus_config=regional_focus_config
    )


# 导出
__all__ = ['VGGSNN_RegionalFocus', 'create_regional_focus_model']