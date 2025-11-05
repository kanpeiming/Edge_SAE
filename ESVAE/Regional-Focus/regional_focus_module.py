# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: regional_focus_module.py
@time: 2024/10/24
@description: 可学习的区域聚焦模块，用于RGB到DVS的自适应域对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

def setup_paths():
    """设置必要的模块路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 添加models路径
    models_path = os.path.join(parent_dir, 'models')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)
    
    # 添加tl_utils路径
    tl_utils_path = os.path.join(parent_dir, 'tl_utils')
    if tl_utils_path not in sys.path:
        sys.path.insert(0, tl_utils_path)

# 在模块加载时设置路径
setup_paths()


class RegionalFocusModule(nn.Module):
    """
    区域聚焦模块：自动学习RGB和DVS特征之间的相似度权重
    
    核心思想：
    1. 计算RGB和DVS特征的逐区域相似度
    2. 使用轻量级网络学习相似度到权重的映射
    3. 对相似度高的区域赋予更大的对齐权重
    4. 约束权重分布，避免退化
    
    Args:
        feature_dim: 特征维度
        spatial_size: 空间尺寸 (H, W)
        reduction: 降维比例，用于减少参数量
        similarity_type: 相似度计算方式 ['cosine', 'l2', 'dot']
        weight_constraint: 权重约束方式 ['softmax', 'sigmoid', 'none']
    """
    
    def __init__(self, 
                 feature_dim,
                 spatial_size=(8, 8),
                 reduction=4,
                 similarity_type='cosine',
                 weight_constraint='softmax',
                 temperature=1.0):
        super(RegionalFocusModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.similarity_type = similarity_type
        self.weight_constraint = weight_constraint
        self.temperature = temperature
        
        # 相似度到权重的映射网络（轻量级MLP）
        hidden_dim = max(16, feature_dim // reduction)  # 确保至少16维
        self.weight_generator = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 输入是相似度标量
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, hidden_dim // 2), 1),  # 输出权重标量
        )
        
        # 可选：使用卷积来捕捉空间上下文
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def compute_similarity(self, feat_rgb, feat_dvs):
        """
        计算RGB和DVS特征的逐像素/逐区域相似度
        
        Args:
            feat_rgb: RGB特征 (N, C, H, W) 或 (N, T, C, H, W)
            feat_dvs: DVS特征 (N, C, H, W) 或 (N, T, C, H, W)
        
        Returns:
            similarity: 相似度图 (N, 1, H, W)
        """
        # 处理时间维度
        if len(feat_rgb.shape) == 5:  # (N, T, C, H, W)
            # 对时间维度求平均，保持维度
            feat_rgb = feat_rgb.mean(dim=1, keepdim=False)  # (N, C, H, W)
            feat_dvs = feat_dvs.mean(dim=1, keepdim=False)  # (N, C, H, W)
        
        # 确保特征尺寸一致
        if feat_rgb.shape != feat_dvs.shape:
            feat_dvs = F.interpolate(feat_dvs, size=feat_rgb.shape[2:], 
                                     mode='bilinear', align_corners=False)
        
        if self.similarity_type == 'cosine':
            # 余弦相似度 (逐像素)
            feat_rgb_norm = F.normalize(feat_rgb, p=2, dim=1)
            feat_dvs_norm = F.normalize(feat_dvs, p=2, dim=1)
            similarity = (feat_rgb_norm * feat_dvs_norm).sum(dim=1, keepdim=True)  # (N, 1, H, W)
            # 从[-1, 1]映射到[0, 1]
            similarity = (similarity + 1) / 2
            
        elif self.similarity_type == 'l2':
            # L2距离（转换为相似度）
            distance = torch.sqrt(((feat_rgb - feat_dvs) ** 2).sum(dim=1, keepdim=True) + 1e-8)
            # 距离越小，相似度越高
            similarity = 1 / (1 + distance)
            
        elif self.similarity_type == 'dot':
            # 点积相似度
            similarity = (feat_rgb * feat_dvs).sum(dim=1, keepdim=True)
            # 归一化到[0, 1]
            similarity = torch.sigmoid(similarity)
        
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        
        return similarity
    
    def generate_weights(self, similarity):
        """
        根据相似度生成对齐权重
        
        Args:
            similarity: 相似度图 (N, 1, H, W) 或 (N, H, W)
        
        Returns:
            weights: 对齐权重图 (N, 1, H, W)
        """
        # 处理维度不匹配的情况
        if len(similarity.shape) == 3:
            # 如果是 (N, H, W)，添加通道维度
            similarity = similarity.unsqueeze(1)  # (N, 1, H, W)
        elif len(similarity.shape) == 4:
            # 如果是 (N, C, H, W)，确保C=1
            if similarity.shape[1] != 1:
                similarity = similarity.mean(dim=1, keepdim=True)  # (N, 1, H, W)
        else:
            raise ValueError(f"Unexpected similarity shape: {similarity.shape}")
            
        N, _, H, W = similarity.shape
        
        # 方法1: 使用MLP逐像素生成权重
        # 将相似度展平 (N*H*W, 1)
        sim_flat = similarity.view(-1, 1)
        
        # 通过MLP生成权重
        weight_flat = self.weight_generator(sim_flat)  # (N*H*W, 1)
        
        # 恢复空间维度
        weights = weight_flat.view(N, 1, H, W)
        
        # 方法2: 使用空间卷积进行精炼（考虑空间上下文）
        weights = self.spatial_refine(weights)
        
        # 应用权重约束
        if self.weight_constraint == 'softmax':
            # 在空间维度上做softmax，确保权重和为1
            weights_flat = weights.view(N, -1)
            weights_flat = F.softmax(weights_flat / self.temperature, dim=1)
            weights = weights_flat.view(N, 1, H, W)
            # 轻微放大，避免权重过小
            weights = weights * (H * W) * 0.1 + 0.9
            
        elif self.weight_constraint == 'sigmoid':
            # Sigmoid约束到[0, 1]，但不要求和为1
            weights = torch.sigmoid(weights)
            # 避免过小的权重
            weights = torch.clamp(weights, min=0.1, max=2.0)
            
        elif self.weight_constraint == 'none':
            # 使用ReLU确保非负，并限制范围
            weights = F.relu(weights)
            weights = torch.clamp(weights, min=0.1, max=2.0)
        
        return weights
    
    def forward(self, feat_rgb, feat_dvs, return_similarity=False):
        """
        前向传播
        
        Args:
            feat_rgb: RGB特征
            feat_dvs: DVS特征
            return_similarity: 是否返回相似度图（用于可视化）
        
        Returns:
            weights: 区域对齐权重 (N, 1, H, W)
            similarity: (可选) 相似度图
        """
        # 1. 计算相似度
        similarity = self.compute_similarity(feat_rgb, feat_dvs)
        
        # 2. 生成权重
        weights = self.generate_weights(similarity)
        
        if return_similarity:
            return weights, similarity
        else:
            return weights


class AdaptiveWeightedAlignmentLoss(nn.Module):
    """
    自适应加权的域对齐损失
    
    结合RegionalFocusModule，对不同区域使用不同的对齐权重
    """
    
    def __init__(self, 
                 feature_dim,
                 spatial_size=(8, 8),
                 base_loss_type='CKA',
                 similarity_type='cosine',
                 weight_constraint='softmax',
                 alpha=1.0,
                 beta=0.1):
        """
        Args:
            feature_dim: 特征维度
            spatial_size: 特征图空间大小
            base_loss_type: 基础对齐损失类型 ['CKA', 'MSE', 'MMD']
            similarity_type: 相似度计算方式
            weight_constraint: 权重约束方式
            alpha: 加权对齐损失的权重
            beta: 权重正则化损失的权重
        """
        super(AdaptiveWeightedAlignmentLoss, self).__init__()
        
        self.base_loss_type = base_loss_type
        self.alpha = alpha
        self.beta = beta
        
        # 区域聚焦模块
        self.regional_focus = RegionalFocusModule(
            feature_dim=feature_dim,
            spatial_size=spatial_size,
            similarity_type=similarity_type,
            weight_constraint=weight_constraint
        )
    
    def compute_base_loss(self, feat_rgb, feat_dvs, weights=None):
        """
        计算基础对齐损失（可选加权）
        
        Args:
            feat_rgb: RGB特征 (N, C, H, W)
            feat_dvs: DVS特征 (N, C, H, W)
            weights: 区域权重 (N, 1, H, W)
        
        Returns:
            loss: 标量损失
        """
        if self.base_loss_type == 'MSE':
            # 逐像素MSE损失
            loss_map = ((feat_rgb - feat_dvs) ** 2).mean(dim=1, keepdim=True)  # (N, 1, H, W)
            
            if weights is not None:
                # 加权MSE
                loss = (loss_map * weights).mean()
            else:
                loss = loss_map.mean()
        
        elif self.base_loss_type == 'CKA':
            # 简化的CKA损失（全局）
            # 这里我们使用特征的全局统计
            feat_rgb_flat = feat_rgb.view(feat_rgb.size(0), feat_rgb.size(1), -1).mean(dim=2)
            feat_dvs_flat = feat_dvs.view(feat_dvs.size(0), feat_dvs.size(1), -1).mean(dim=2)
            
            # 计算余弦相似度
            feat_rgb_norm = F.normalize(feat_rgb_flat, p=2, dim=1)
            feat_dvs_norm = F.normalize(feat_dvs_flat, p=2, dim=1)
            similarity = (feat_rgb_norm * feat_dvs_norm).sum(dim=1).mean()
            
            # CKA损失：最大化相似度 = 最小化负相似度
            loss = 1 - similarity
            
            # 如果有权重，可以考虑加权（这里暂时不加权，因为已经是全局特征）
        
        elif self.base_loss_type == 'MMD':
            # 最大均值差异
            feat_rgb_mean = feat_rgb.mean(dim=[0, 2, 3])
            feat_dvs_mean = feat_dvs.mean(dim=[0, 2, 3])
            loss = ((feat_rgb_mean - feat_dvs_mean) ** 2).sum()
        
        else:
            raise ValueError(f"Unknown base loss type: {self.base_loss_type}")
        
        return loss
    
    def compute_weight_regularization(self, weights):
        """
        权重正则化：防止权重退化（如全为0或集中在少数位置）
        
        Args:
            weights: 区域权重 (N, 1, H, W)
        
        Returns:
            reg_loss: 正则化损失
        """
        # 1. 熵正则化：鼓励权重分布均匀
        weights_norm = weights / (weights.sum(dim=[2, 3], keepdim=True) + 1e-8)
        entropy = -(weights_norm * torch.log(weights_norm + 1e-8)).sum(dim=[2, 3]).mean()
        entropy_loss = -entropy  # 最大化熵 = 最小化负熵
        
        # 2. 稀疏性约束：避免权重过于分散
        sparsity_loss = (weights ** 2).mean()
        
        # 组合正则化
        reg_loss = 0.5 * entropy_loss + 0.5 * sparsity_loss
        
        return reg_loss
    
    def forward(self, feat_rgb, feat_dvs, return_weights=False):
        """
        前向传播
        
        Args:
            feat_rgb: RGB特征
            feat_dvs: DVS特征
            return_weights: 是否返回权重（用于可视化）
        
        Returns:
            total_loss: 总损失
            weights: (可选) 区域权重
        """
        # 1. 生成区域权重
        weights, similarity = self.regional_focus(feat_rgb, feat_dvs, return_similarity=True)
        
        # 2. 计算加权对齐损失
        alignment_loss = self.compute_base_loss(feat_rgb, feat_dvs, weights)
        
        # 3. 计算权重正则化
        reg_loss = self.compute_weight_regularization(weights)
        
        # 4. 总损失
        total_loss = self.alpha * alignment_loss + self.beta * reg_loss
        
        if return_weights:
            return total_loss, weights, similarity
        else:
            return total_loss


class MultiScaleRegionalFocus(nn.Module):
    """
    多尺度区域聚焦模块
    
    在不同特征层次上计算相似度和权重，更全面地捕捉跨域关系
    """
    
    def __init__(self, 
                 feature_dims=[128, 256, 512],
                 spatial_sizes=[(16, 16), (8, 8), (4, 4)],
                 similarity_type='cosine',
                 weight_constraint='softmax'):
        super(MultiScaleRegionalFocus, self).__init__()
        
        self.num_scales = len(feature_dims)
        
        # 为每个尺度创建一个RegionalFocusModule
        self.regional_focus_modules = nn.ModuleList([
            RegionalFocusModule(
                feature_dim=feature_dims[i],
                spatial_size=spatial_sizes[i],
                similarity_type=similarity_type,
                weight_constraint=weight_constraint
            )
            for i in range(self.num_scales)
        ])
        
        # 多尺度融合权重（可学习）
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
    
    def forward(self, feats_rgb_list, feats_dvs_list):
        """
        Args:
            feats_rgb_list: RGB特征列表 [feat1, feat2, feat3]
            feats_dvs_list: DVS特征列表 [feat1, feat2, feat3]
        
        Returns:
            weights_list: 每个尺度的权重列表
            fused_weights: 融合后的权重（上采样到最大尺寸）
        """
        weights_list = []
        
        # 计算每个尺度的权重
        for i, (feat_rgb, feat_dvs) in enumerate(zip(feats_rgb_list, feats_dvs_list)):
            weights = self.regional_focus_modules[i](feat_rgb, feat_dvs)
            weights_list.append(weights)
        
        # 融合多尺度权重（上采样到最大尺寸）
        max_size = feats_rgb_list[0].shape[2:]
        fused_weights = 0
        
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        
        for i, weights in enumerate(weights_list):
            # 上采样到最大尺寸
            weights_upsampled = F.interpolate(weights, size=max_size, 
                                             mode='bilinear', align_corners=False)
            fused_weights += scale_weights_norm[i] * weights_upsampled
        
        return weights_list, fused_weights




class EnhancedVGGSNNWithRegionalFocus(nn.Module):
    """
    增强的VGGSNN：集成分层区域聚焦机制
    
    关键改进：
    1. RGB是3通道输入，DVS是2通道输入，都映射到64维特征
    2. 在主干网每一层都施加区域关注约束来引导对齐
    3. 使用相同的目标函数进行各层级对齐
    """
    
    def __init__(self, 
                 cls_num=10, 
                 img_shape=32,
                 enable_hierarchical_focus=True,
                 focus_similarity_type='cosine',
                 focus_weight_constraint='softmax',
                 regional_loss_weight=0.5):
        super(EnhancedVGGSNNWithRegionalFocus, self).__init__()
        
        # 导入必要的层
        from TET__layer import Layer, SeqToANNContainer, LIFSpike
        
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        
        # 输入层：RGB是3通道，DVS是2通道，都映射到64通道
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # DVS是2通道输入
        
        # 分层特征提取（便于插入区域关注）
        self.layer1 = Layer(64, 128, 3, 1, 1)
        self.pool1 = pool
        self.layer2 = Layer(128, 256, 3, 1, 1)
        self.layer3 = Layer(256, 256, 3, 1, 1)
        self.pool2 = pool
        self.layer4 = Layer(256, 512, 3, 1, 1)
        self.layer5 = Layer(512, 512, 3, 1, 1)
        self.pool3 = pool
        self.layer6 = Layer(512, 512, 3, 1, 1)
        self.layer7 = Layer(512, 512, 3, 1, 1)
        self.pool4 = pool
        
        W = int(img_shape / 2 / 2 / 2 / 2)
        # 与baseline模型保持完全一致，使用动态计算的维度
        # 不使用自适应池化层，直接使用计算出的W维度
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        self.bottleneck_lif_node = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))
        
        # 分层区域聚焦模块
        self.enable_hierarchical_focus = enable_hierarchical_focus
        self.regional_loss_weight = regional_loss_weight
        
        if enable_hierarchical_focus:
            # 为每一层创建区域关注模块
            self.regional_focus_modules = nn.ModuleList([
                RegionalFocusModule(64, (img_shape, img_shape), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),      # 输入层后
                RegionalFocusModule(128, (img_shape//2, img_shape//2), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),  # pool1后
                RegionalFocusModule(256, (img_shape//4, img_shape//4), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),  # pool2后
                RegionalFocusModule(512, (img_shape//8, img_shape//8), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint),  # pool3后
                RegionalFocusModule(512, (img_shape//16, img_shape//16), similarity_type=focus_similarity_type, weight_constraint=focus_weight_constraint), # pool4后
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
        提取分层特征用于区域关注
        
        Args:
            x: 输入数据 (N, T, C, H, W) - RGB是3通道，DVS是2通道
            domain_type: 'rgb' 或 'dvs'
        
        Returns:
            features: 各层特征列表
            x_final: 最终特征
            x_mem: 瓶颈层记忆
        """
        features = []
        
        # 输入编码 - RGB是3通道，DVS是2通道
        if domain_type == 'rgb':
            x, _ = self.rgb_input(x)
        else:
            x, _ = self.dvs_input(x)
        features.append(x)  # 64维特征
        
        # 逐层提取特征
        x = self.layer1(x)
        x = self.pool1(x)
        features.append(x)  # 128维特征
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        features.append(x)  # 256维特征
        
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool3(x)
        features.append(x)  # 512维特征
        
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.pool4(x)
        features.append(x)  # 512维特征
        
        # 瓶颈层 - 与baseline模型保持一致，直接flatten
        x_before_flatten = x  # 保存flatten前的特征用于区域关注
        
        # 直接flatten，不使用自适应池化
        x = torch.flatten(x, 2)
        
        # 使用动态维度的瓶颈层
        x = self.bottleneck(x)
        x, x_mem = self.bottleneck_lif_node(x, return_mem=True)
        # 注意：不将flatten后的特征添加到features中，因为它不是4D张量
        # features.append(x)  # 这会导致维度问题
        
        return features, x, x_mem
    
    def compute_regional_alignment_loss(self, source_features, target_features):
        """
        计算各层的区域对齐损失
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
        增强的前向传播，集成分层区域关注
        
        Args:
            source: 源域输入 (N, T, C, H, W)
            target: 目标域输入 (N, T, C, H, W)
            encoder_tl_loss_type: 编码器损失类型
            feature_tl_loss_type: 特征损失类型
        
        Returns:
            训练时: (source_clf, target_clf, encoder_tl_loss, feature_tl_loss, regional_loss)
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


# 导出主要类
__all__ = [
    'RegionalFocusModule',
    'AdaptiveWeightedAlignmentLoss', 
    'MultiScaleRegionalFocus',
    'EnhancedVGGSNNWithRegionalFocus'
]

