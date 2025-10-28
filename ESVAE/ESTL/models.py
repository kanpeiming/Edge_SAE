# -*- coding: utf-8 -*-
"""
N-Caltech101 DVS Classification Models
包含CNN-SNN和ViT-SNN（Spikformer）基线模型

@description: Baseline SNN models for DVS classification
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


# ============================================================================
# Baseline Model 1: VGG-SNN (CNN-based, DVS only)
# ============================================================================

class BaselineVGGSNN(nn.Module):
    """
    Baseline DVS classifier without edge guidance (for comparison).
    CNN-based architecture using VGG structure.
    """
    
    def __init__(self, cls_num=10, img_shape=32):
        super().__init__()
        
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        
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
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, dvs_data):
        x = self.dvs_input(dvs_data)
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x


# ============================================================================
# Baseline Model 2: Spikformer (ViT-SNN, DVS only)
# ============================================================================

class SpikingSelfAttention(nn.Module):
    """
    Spiking Self-Attention Module
    基于脉冲的自注意力机制
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = SeqToANNContainer(nn.Linear(dim, dim * 3, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = SeqToANNContainer(nn.Linear(dim, dim))
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_lif = LIFSpike()

    def forward(self, x):
        """
        Args:
            x: (B, T, N, C) where N is number of tokens
        Returns:
            out: (B, T, N, C)
        """
        B, T, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (B, T, N, 3*C)
        qkv = qkv.reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # (3, B, T, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, T, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum
        x = (attn @ v).transpose(2, 3).reshape(B, T, N, C)  # (B, T, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_lif(x)
        x = self.proj_drop(x)
        
        return x


class SpikingMLPBlock(nn.Module):
    """
    Spiking MLP Block
    基于脉冲的MLP模块
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = SeqToANNContainer(nn.Linear(in_features, hidden_features))
        self.lif1 = LIFSpike()
        self.fc2 = SeqToANNContainer(nn.Linear(hidden_features, out_features))
        self.lif2 = LIFSpike()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.drop(x)
        return x


class SpikingTransformerBlock(nn.Module):
    """
    Spiking Transformer Block
    包含Self-Attention和MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLPBlock(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, N, C)
        """
        # Self-Attention with residual
        x = x + self.attn(self.apply_norm(self.norm1, x))
        
        # MLP with residual
        x = x + self.mlp(self.apply_norm(self.norm2, x))
        
        return x
    
    def apply_norm(self, norm_layer, x):
        """Apply LayerNorm across spatial and channel dimensions"""
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C)
        x = norm_layer(x)
        x = x.reshape(B, T, N, C)
        return x


class SpikingPatchEmbedding(nn.Module):
    """
    Spiking Patch Embedding
    将DVS事件数据转换为patch tokens
    """
    def __init__(self, img_size_h=48, img_size_w=48, patch_size=4, in_channels=2, embed_dim=256):
        super().__init__()
        self.img_size = (img_size_h, img_size_w)
        self.patch_size = patch_size
        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding using convolution
        self.proj = SeqToANNContainer(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        self.proj_lif = LIFSpike()

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) DVS event data
        Returns:
            out: (B, T, N, embed_dim) where N = num_patches
        """
        B, T, C, H, W = x.shape
        
        # Project to patches
        x = self.proj(x)  # (B, T, embed_dim, H', W')
        x = self.proj_lif(x)
        
        # Flatten spatial dimensions
        # x shape: (B, T, embed_dim, H', W')
        x = x.flatten(3)  # (B, T, embed_dim, H'*W')
        x = x.transpose(2, 3)  # (B, T, H'*W', embed_dim) = (B, T, N, embed_dim)
        
        return x


class BaselineSpikformer(nn.Module):
    """
    Baseline Spikformer for DVS Classification
    基于Spiking Transformer的DVS分类器（基线版本）
    
    架构与CLIP ViT对齐，便于后续知识蒸馏
    
    参考:
    - Spikformer (ICLR 2023)
    - Spike-driven Transformer (NeurIPS 2023)
    """
    
    def __init__(self,
                 img_size_h=48,
                 img_size_w=48,
                 patch_size=4,
                 in_channels=2,
                 num_classes=100,
                 embed_dim=256,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.):
        """
        Args:
            img_size_h, img_size_w: 输入图像大小
            patch_size: Patch大小（类似CLIP的16）
            in_channels: 输入通道数（DVS=2: ON/OFF）
            num_classes: 分类类别数
            embed_dim: Token嵌入维度
            depth: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度比例
            qkv_bias: QKV是否使用bias
            drop_rate: Dropout比例
            attn_drop_rate: Attention dropout比例
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        
        # Patch embedding
        self.patch_embed = SpikingPatchEmbedding(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = SeqToANNContainer(nn.Linear(embed_dim, num_classes))
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: (B, T, C, H, W) DVS event data
            return_features: 是否返回中间特征（用于蒸馏）
        
        Returns:
            logits: (B, T, num_classes) if not return_features
            (logits, features_dict) if return_features
        """
        B, T, C, H, W = x.shape
        
        # 1. Patch embedding
        x = self.patch_embed(x)  # (B, T, N, embed_dim)
        
        # 2. Add CLS token
        cls_tokens = self.cls_token.expand(B, T, -1, -1)  # (B, T, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=2)  # (B, T, N+1, embed_dim)
        
        # 3. Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # 5. Extract features
        x = self.apply_norm(self.norm, x)
        
        # CLS token for classification
        cls_token_out = x[:, :, 0]  # (B, T, embed_dim)
        
        # Patch tokens for feature alignment
        patch_tokens_out = x[:, :, 1:]  # (B, T, N, embed_dim)
        
        # 6. Classification head
        logits = self.head(cls_token_out)  # (B, T, num_classes)
        
        if return_features:
            features_dict = {
                'cls_token': cls_token_out,      # (B, T, embed_dim)
                'patch_tokens': patch_tokens_out, # (B, T, N, embed_dim)
                'logits': logits                  # (B, T, num_classes)
            }
            return logits, features_dict
        
        return logits
    
    def apply_norm(self, norm_layer, x):
        """Apply LayerNorm"""
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C)
        x = norm_layer(x)
        x = x.reshape(B, T, N, C)
        return x
