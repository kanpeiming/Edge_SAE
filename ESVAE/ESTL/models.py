# -*- coding: utf-8 -*-
"""
Structure-Guided DVS Classification with Cross-Attention Fusion
DVS动态查询RGB结构知识，通过对比学习缓解样本量不匹配问题

@author: Based on ESEG framework
@description: Cross-modal attention fusion for DVS classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import sys
import os

# Add parent directory to path for imports to avoid module not found errors
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.TET__layer import *
from pretrain.Edge import MultiThresholdCannyEdgeModule


# ============================================================================
# D2CAF Fusion Module adapted for SNN (Temporal features)
# ============================================================================

class SNND2CAFFusion(nn.Module):
    """
    D2CAF Fusion module adapted for Spiking Neural Networks.
    Handles temporal dimension (T) in addition to spatial dimensions (H, W).
    
    Parameters
    ----------
    C_edge : int
        Channel dimension of edge features
    C_dvs : int
        Channel dimension of DVS features
    C_out : int
        Output channel dimension after fusion
    H, W : int
        Spatial resolution of feature maps
    mask_size : int
        Dynamic window size for attention
    fusion_type : str
        'concat' or 'cross_attention'
    """
    
    def __init__(self,
                 C_edge: int,
                 C_dvs: int,
                 C_out: int,
                 H: int,
                 W: int,
                 mask_size: int = 3,
                 fusion_type: str = 'cross_attention'):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.C_out = C_out
        
        # Channel alignment
        self.conv_edge_align = SeqToANNContainer(nn.Conv2d(C_edge, C_out, 1))
        self.conv_dvs_align = SeqToANNContainer(nn.Conv2d(C_dvs, C_out, 1))
        
        if fusion_type == 'cross_attention':
            # Q/K/V projections for cross-attention
            self.conv_q = nn.Conv1d(C_out, C_out, 1)  # Query from edge
            self.conv_k = nn.Conv1d(C_out, C_out, 1)  # Key from DVS
            self.conv_v = nn.Conv1d(C_out, C_out, 1)  # Value from DVS
            
            # Dynamic window mask
            self.attention_mask = self._create_attention_masks(H, W, mask_size)
            
        # Final fusion convolution
        fusion_in_channels = 2 * C_out if fusion_type == 'concat' else C_out
        self.conv_fusion = SeqToANNContainer(nn.Conv2d(fusion_in_channels, C_out, 1))
        self.lif = LIFSpike()
    
    @staticmethod
    def _create_attention_masks(H: int, W: int, mask_size: int) -> torch.Tensor:
        """Create dynamic window attention masks (same as ESEG)"""
        masks = -10000 * torch.ones(H * W, H * W)
        half = mask_size // 2
        
        for q_x in range(H):
            for q_y in range(W):
                x0, x1 = max(0, q_x - half), min(H, q_x + half + 1)
                y0, y1 = max(0, q_y - half), min(W, q_y + half + 1)
                
                local = -10000 * torch.ones(H, W)
                local[x0:x1, y0:y1] = 0
                masks[q_x * W + q_y, :] = local.flatten()
        
        return masks.cuda()
    
    def forward(self, edge_features: torch.Tensor, dvs_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse edge-guided features with DVS features.
        
        Args:
            edge_features: (B, T, C_edge, H, W) - Edge branch features
            dvs_features: (B, T, C_dvs, H, W) - DVS branch features
            
        Returns:
            fused: (B, T, C_out, H, W) - Fused features
        """
        B, T, _, H, W = dvs_features.shape
        
        # Align channels
        edge_aligned = self.conv_edge_align(edge_features)  # (B, T, C_out, H, W)
        dvs_aligned = self.conv_dvs_align(dvs_features)
        
        if self.fusion_type == 'concat':
            # Simple concatenation + fusion conv
            fused = torch.cat([edge_aligned, dvs_aligned], dim=2)
            fused = self.conv_fusion(fused)
            fused = self.lif(fused)
            
        elif self.fusion_type == 'cross_attention':
            # Cross-attention fusion (D2CAF style)
            # Reshape to process temporal dimension
            edge_flat = edge_aligned.view(B * T, self.C_out, H, W)
            dvs_flat = dvs_aligned.view(B * T, self.C_out, H, W)
            
            # Compute similarity and density (DIM - Density Importance Modulation)
            l2_edge = torch.norm(edge_flat, p=2, dim=1, keepdim=True)
            l2_dvs = torch.norm(dvs_flat, p=2, dim=1, keepdim=True)
            
            edge_norm = edge_flat / (l2_edge + 1e-6)
            dvs_norm = dvs_flat / (l2_dvs + 1e-6)
            
            # Similarity score
            similarity = (edge_norm * dvs_norm).sum(dim=1, keepdim=True)
            similarity = (similarity + 1) / 2  # Normalize to [0, 1]
            
            # Density score (edge features are typically sparse)
            density = torch.norm(edge_flat, p=2, dim=1, keepdim=True)
            density_flat = density.view(B * T, -1)
            density_flat = F.softmax(density_flat, dim=1) * density_flat.shape[-1]
            density = density_flat.view(B * T, 1, H, W)
            
            # Weighted features
            edge_weighted = (edge_flat * similarity).reshape(B * T, self.C_out, -1)
            dvs_weighted = (dvs_flat * density).reshape(B * T, self.C_out, -1)
            
            # Cross-attention
            q = self.conv_q(edge_weighted)  # Query from edge
            k = self.conv_k(dvs_weighted)   # Key from DVS
            v = self.conv_v(dvs_weighted)   # Value from DVS
            
            # Scaled dot-product attention with dynamic mask
            attn = torch.bmm(q.transpose(1, 2), k) / (self.C_out ** 0.5)
            attn = attn + self.attention_mask  # Apply dynamic window mask
            attn = F.softmax(attn, dim=-1)
            
            # Attention output
            out = torch.bmm(attn, v.transpose(1, 2))  # (BT, HW, C)
            out = out.view(B * T, H, W, self.C_out).permute(0, 3, 1, 2)
            
            # Fusion
            fused = out.view(B, T, self.C_out, H, W)
            fused = self.conv_fusion(fused)
            fused = self.lif(fused)
        
        return fused


# ============================================================================
# Cross-Modal Attention Fusion (DVS动态查询RGB结构知识)
# ============================================================================

class CrossModalAttentionFusion(nn.Module):
    """
    跨模态注意力融合模块
    DVS特征作为Query，动态查询RGB结构特征(Key/Value)
    
    核心思想：
    - DVS主动学习：Query = DVS ("我需要什么结构信息？")
    - RGB知识库：Key/Value = RGB ("我有什么结构知识")
    - 动态选择：Attention自动学习哪些RGB特征对DVS有用
    """
    
    def __init__(self, dvs_channels, rgb_channels, fusion_channels=512, num_heads=8):
        super().__init__()
        
        self.fusion_channels = fusion_channels
        self.num_heads = num_heads
        
        # 特征对齐投影
        self.dvs_proj = SeqToANNContainer(nn.Conv2d(dvs_channels, fusion_channels, 1))
        self.rgb_proj = nn.Conv2d(rgb_channels, fusion_channels, 1)  # RGB无时间维度
        
        # Multi-head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(fusion_channels)
        self.norm2 = nn.LayerNorm(fusion_channels)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(fusion_channels, fusion_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_channels * 2, fusion_channels),
            nn.Dropout(0.1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, dvs_feat, rgb_feat):
        """
        DVS动态查询RGB结构知识
        
        Args:
            dvs_feat: (B, T, C_dvs, H, W) - DVS时序特征
            rgb_feat: (B, C_rgb, H, W) - RGB结构特征（无时间维度）
        
        Returns:
            enhanced_dvs: (B, T, C_fusion, H, W) - 增强的DVS特征
        """
        B, T, C_d, H, W = dvs_feat.shape
        
        # 特征对齐
        dvs_aligned = self.dvs_proj(dvs_feat)  # (B, T, C, H, W)
        rgb_aligned = self.rgb_proj(rgb_feat).unsqueeze(1)  # (B, 1, C, H, W)
        
        # Flatten spatial dimensions for attention
        dvs_tokens = dvs_aligned.flatten(3).permute(0, 1, 3, 2)  # (B, T, HW, C)
        rgb_tokens = rgb_aligned.flatten(3).permute(0, 1, 3, 2)  # (B, 1, HW, C)
        rgb_tokens = rgb_tokens.repeat(1, T, 1, 1)  # (B, T, HW, C) 重复到每个时间步
        
        # Reshape for batch processing: (B*T, HW, C)
        dvs_flat = dvs_tokens.reshape(B * T, H * W, self.fusion_channels)
        rgb_flat = rgb_tokens.reshape(B * T, H * W, self.fusion_channels)
        
        # Cross-Attention: DVS查询RGB
        attn_out, attn_weights = self.cross_attn(
            query=dvs_flat,      # DVS问：我需要什么？
            key=rgb_flat,        # RGB答：我有什么结构
            value=rgb_flat       # RGB给：对应的结构特征
        )
        
        # Residual + Norm
        dvs_flat = self.norm1(dvs_flat + attn_out)
        
        # Feed-Forward Network
        dvs_flat = self.norm2(dvs_flat + self.ffn(dvs_flat))
        
        # Reshape back to feature map
        enhanced_dvs = dvs_flat.reshape(B, T, H, W, self.fusion_channels)
        enhanced_dvs = enhanced_dvs.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        
        return enhanced_dvs


# ============================================================================
# VGG-based Structure Branch (RGB结构表示学习)
# ============================================================================

class VGGEdgeBranch(nn.Module):
    """
    VGG SNN branch for RGB structure extraction.
    
    两种模式统一使用RGB输入（3通道）：
    1. Pretraining: RGB -> edge prediction (学习RGB→结构的映射)
    2. Fusion: RGB -> structure features (复用预训练的结构提取能力)
    
    关键：预训练后，模型已学会从RGB提取结构，无需手工边缘提取！
    """
    
    def __init__(self, img_shape=32, pretrain_mode=False):
        super().__init__()
        
        self.pretrain_mode = pretrain_mode
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        
        # Input layer - 统一使用RGB 3通道输入
        # 预训练：学习RGB→结构
        # 融合：直接用RGB提取结构（不需要Canny了！）
        self.input_layer = Layer(3, 64, 3, 1, 1, True)
        
        # Multi-scale feature extraction
        # Stage 1: 64 -> 128
        self.stage1 = nn.Sequential(
            Layer(64, 128, 3, 1, 1),
            pool,
        )
        
        # Stage 2: 128 -> 256
        self.stage2 = nn.Sequential(
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            pool,
        )
        
        # Stage 3: 256 -> 512
        self.stage3 = nn.Sequential(
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
        )
        
        # Stage 4: 512 -> 512
        self.stage4 = nn.Sequential(
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
        )
        
        # 预训练模式：边缘预测头
        if pretrain_mode:
            self.edge_head = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 3, 1)  # 输出3通道 [弱边缘, 中等边缘, 强边缘]
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - 两种模式统一使用RGB输入
        
        Pretrain mode:
            Input: (B, 3, H, W) RGB image
            Output: (B, 3, H, W) edge prediction (学习任务)
        
        Fusion mode:
            Input: (B, 3, H, W) RGB image（复用预训练能力！）
            Output: List of multi-scale structure features
        """
        if self.pretrain_mode:
            # 预训练模式：RGB -> 边缘预测（训练）
            # 添加时间维度用于SNN
            if x.dim() == 4:  # (B, 3, H, W)
                x = x.unsqueeze(1)  # (B, 1, 3, H, W)
            
            feat = self.input_layer(x)
            if isinstance(feat, tuple):
                feat, _ = feat
            
            f1 = self.stage1(feat)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)  # (B, T, 512, 2, 2)
            
            # 上采样到原始尺寸
            f4_up = F.interpolate(f4.squeeze(1), size=(32, 32), mode='bilinear', align_corners=False)
            
            # 边缘预测 - 3通道输出
            edge_pred = self.edge_head(f4_up)  # (B, 3, 32, 32)
            
            return edge_pred
        
        else:
            # 融合模式：RGB -> 结构特征（利用预训练知识）
            # 添加时间维度用于SNN
            if x.dim() == 4:  # (B, 3, H, W)
                x = x.unsqueeze(1)  # (B, 1, 3, H, W)
            
            feat = self.input_layer(x)
            if isinstance(feat, tuple):
                feat, _ = feat
            
            f1 = self.stage1(feat)      # Scale 1/2
            f2 = self.stage2(f1)         # Scale 1/4
            f3 = self.stage3(f2)         # Scale 1/8
            f4 = self.stage4(f3)         # Scale 1/16
            
            return [f1, f2, f3, f4]


# ============================================================================
# VGG-based DVS Branch (baseline classifier)
# ============================================================================

class VGGDVSBranch(nn.Module):
    """
    VGG SNN branch for processing DVS data (baseline).
    Identical structure to Edge branch but processes 2-channel DVS data.
    """
    
    def __init__(self, img_shape=32):
        super().__init__()
        
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        
        # Input layer for DVS data
        self.input_layer = Layer(2, 64, 3, 1, 1, True)
        
        # Multi-scale feature extraction (same as edge branch)
        self.stage1 = nn.Sequential(
            Layer(64, 128, 3, 1, 1),
            pool,
        )
        
        self.stage2 = nn.Sequential(
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            pool,
        )
        
        self.stage3 = nn.Sequential(
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
        )
        
        self.stage4 = nn.Sequential(
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, dvs_data: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale DVS features.
        
        Args:
            dvs_data: (B, T, 2, H, W) - DVS event data
            
        Returns:
            List of features at different scales
        """
        x = self.input_layer(dvs_data)
        # If return_mem=True, unpack the tuple
        if isinstance(x, tuple):
            x, _ = x
        
        # Extract multi-scale features
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        
        return [f1, f2, f3, f4]


# ============================================================================
# Main Model: Edge-Guided DVS Classifier with D2CAF Fusion
# ============================================================================

class EdgeGuidedVGGSNN(nn.Module):
    """
    Structure-Guided DVS Classification with Cross-Attention.
    
    Architecture:
    1. RGB Structure Branch: 直接从RGB提取结构特征（预训练+冻结）
       - 预训练时学会：RGB → 边缘预测（结构学习）
       - 融合时复用：RGB → 结构特征（无需手工边缘提取！）
    2. DVS Branch: 处理DVS时序特征（可训练）
    3. Cross-Attention Fusion: DVS动态查询RGB结构知识（可训练）
    4. Classifier: 分类头（可训练）
    
    核心优势:
    - ✅ 端到端：预训练后，RGB直接输入，无需Canny等手工特征
    - ✅ Cross-Attention：DVS主动查询有用的RGB结构信息
    - ✅ 对比学习：缓解样本量不匹配（10k DVS vs 50k RGB）
    """
    
    def __init__(self,
                 cls_num: int = 10,
                 img_shape: int = 32,
                 device: str = 'cuda',
                 fusion_stages: List[int] = [4],  # 只在最后一层融合
                 fusion_type: str = 'cross_attention'):
        super().__init__()
        
        self.cls_num = cls_num
        self.img_shape = img_shape
        self.fusion_stages = fusion_stages
        self.fusion_type = fusion_type
        
        # RGB Structure branch and DVS branches
        # 注意：不再需要手工边缘提取！VGGEdgeBranch预训练时已学会从RGB提取结构
        self.edge_branch = VGGEdgeBranch(img_shape)  # RGB→结构特征（预训练能力）
        self.dvs_branch = VGGDVSBranch(img_shape)
        
        # Feature dimensions: [128, 256, 512, 512]
        # Spatial resolutions: [16, 8, 4, 2] for img_shape=32
        feature_dims = [128, 256, 512, 512]
        spatial_sizes = [img_shape // (2 ** (i+1)) for i in range(4)]
        
        # Fusion modules (可选D2CAF或Cross-Attention)
        self.fusion_modules = nn.ModuleList()
        for stage_idx in range(4):
            if (stage_idx + 1) in fusion_stages:
                spatial_size = spatial_sizes[stage_idx]
                
                if fusion_type == 'cross_attention':
                    # 新的Cross-Attention融合
                    fusion = CrossModalAttentionFusion(
                        dvs_channels=feature_dims[stage_idx],
                        rgb_channels=feature_dims[stage_idx],
                        fusion_channels=feature_dims[stage_idx],
                        num_heads=8
                    )
                else:
                    # 旧的D2CAF融合
                    fusion = SNND2CAFFusion(
                        C_edge=feature_dims[stage_idx],
                        C_dvs=feature_dims[stage_idx],
                        C_out=feature_dims[stage_idx],
                        H=spatial_size,
                        W=spatial_size,
                        mask_size=3,
                        fusion_type='concat'
                    )
                
                self.fusion_modules.append(fusion)
            else:
                self.fusion_modules.append(None)
        
        # Classification head
        W = int(img_shape / 16)  # After 4 pooling layers
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        self.bottleneck_lif = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self,
                rgb_data: torch.Tensor = None,
                dvs_data: torch.Tensor = None,
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass with RGB-based structure guidance and DVS classification.
        
        关键改进：RGB直接输入edge_branch，无需手工边缘提取！
        预训练已让模型学会从RGB提取结构信息。
        
        Args:
            rgb_data: (B, 3, H, W) - RGB image（直接输入，不需要边缘提取！）
            dvs_data: (B, T, 2, H, W) - DVS event data (temporal)
            return_features: If True, return intermediate features (用于对比学习)
            
        Returns:
            output: (B, T, cls_num) - Classification logits
            如果return_features=True，额外返回中间特征字典
        """
        # Handle backward compatibility - if only one input is provided
        if dvs_data is None:
            # Assume rgb_data is actually dvs_data (old API)
            dvs_data = rgb_data
            rgb_data = None
        
        B, T, C, H, W = dvs_data.shape
        
        # 提取RGB结构特征（无需边缘提取，直接用预训练能力！）
        rgb_structure_features = None
        if rgb_data is not None:
            # RGB直接输入edge_branch，利用预训练的结构提取能力
            # edge_branch在预训练时已学会：RGB → 结构特征
            rgb_structure_list = self.edge_branch(rgb_data)  # List of 4 feature maps
            rgb_structure_features = rgb_structure_list  # 保存所有尺度特征
        else:
            # Fallback: 从DVS生成伪RGB
            dvs_intensity = dvs_data.mean(dim=1)  # (B, 2, H, W)
            # 简单复制到3通道
            pseudo_rgb = dvs_intensity.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # (B, 3, H, W)
            rgb_structure_list = self.edge_branch(pseudo_rgb)
            rgb_structure_features = rgb_structure_list
        
        # 提取DVS特征
        dvs_features = self.dvs_branch(dvs_data)  # List of 4 feature maps (dense DVS)
        
        # Multi-scale fusion
        fused_features = []
        for stage_idx in range(4):
            if self.fusion_modules[stage_idx] is not None:
                # Fuse at this stage
                if self.fusion_type == 'cross_attention':
                    # Cross-Attention: DVS查询RGB (RGB特征无时间维度)
                    rgb_feat_static = rgb_structure_features[stage_idx].mean(dim=1)  # (B, C, H, W)
                    fused = self.fusion_modules[stage_idx](
                        dvs_feat=dvs_features[stage_idx],
                        rgb_feat=rgb_feat_static
                    )
                else:
                    # D2CAF fusion
                    fused = self.fusion_modules[stage_idx](
                        rgb_structure_features[stage_idx],
                        dvs_features[stage_idx]
                    )
                fused_features.append(fused)
            else:
                # No fusion, use DVS features directly
                fused_features.append(dvs_features[stage_idx])
        
        # Use the last fused feature for classification
        final_features = fused_features[-1]  # (B, T, 512, 2, 2)
        
        # Flatten and classify
        final_features_flat = torch.flatten(final_features, 2)  # (B, T, 512*2*2)
        bottleneck_out = self.bottleneck(final_features_flat)
        bottleneck_out = self.bottleneck_lif(bottleneck_out)
        output = self.classifier(bottleneck_out)  # (B, T, cls_num)
        
        if return_features:
            # 返回用于对比学习的特征
            # DVS特征：时间平均后作为DVS representation
            dvs_repr = dvs_features[-1].mean(dim=1)  # (B, C, H, W)
            dvs_repr_pooled = F.adaptive_avg_pool2d(dvs_repr, (1, 1)).flatten(1)  # (B, C)
            
            # RGB特征：作为RGB representation
            rgb_repr = rgb_structure_features[-1].mean(dim=1)  # (B, C, H, W)
            rgb_repr_pooled = F.adaptive_avg_pool2d(rgb_repr, (1, 1)).flatten(1)  # (B, C)
            
            return output, {
                'rgb_features': rgb_structure_features,
                'dvs_features': dvs_features,
                'fused_features': fused_features,
                'rgb_repr': rgb_repr_pooled,  # 用于对比学习
                'dvs_repr': dvs_repr_pooled   # 用于对比学习
            }
        
        return output


# ============================================================================
# Baseline Model (DVS only, no edge guidance)
# ============================================================================

class BaselineVGGSNN(nn.Module):
    """
    Baseline DVS classifier without edge guidance (for comparison).
    Identical to original VGGSNN from baseline.py for fair comparison.
    """
    
    def __init__(self, cls_num=10, img_shape=32):
        super().__init__()
        
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        
        # 与原始VGGSNN保持一致：不使用return_mem
        self.dvs_input = Layer(2, 64, 3, 1, 1)
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
        
        W = int(img_shape / 16)
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        # 与原始VGGSNN保持一致：不使用额外的LIF层
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, dvs_data):
        x = self.dvs_input(dvs_data)
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test the model
    print("Testing DVS-based Edge-Guided VGG SNN...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = EdgeGuidedVGGSNN(
        cls_num=10,
        img_shape=32,
        device=device,
        fusion_stages=[2, 3, 4],
        fusion_type='cross_attention'
    ).to(device)
    
    # Test input - only DVS data
    B, T, H, W = 4, 10, 32, 32
    dvs_data = torch.randn(B, T, 2, H, W).to(device)  # DVS with temporal dimension
    
    # Forward pass
    print(f"Input shape: DVS={dvs_data.shape}")
    output, features = model(dvs_data, return_features=True)
    
    print(f"Output shape: {output.shape}")  # Should be (4, 10, 10)
    print(f"Number of fused stages: {len(features['fused_features'])}")
    print("✓ Model test passed!")

