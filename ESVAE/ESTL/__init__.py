# -*- coding: utf-8 -*-
"""
ESTL: N-Caltech101 DVS Classification - Baseline Module
仅保留DVS分类基线训练

核心功能:
1. BaselineVGGSNN: DVS-only分类模型
2. Caltech101数据加载器: 支持DVS-only和RGB+DVS配对
3. 基线训练脚本: train_caltech101_baseline.py

使用方法:
=== DVS-only基线训练 ===
python train_caltech101_baseline.py --epochs 150 --batch_size 16 --lr 0.001
"""

from .models import BaselineVGGSNN, BaselineSpikformer
from .caltech101_dataloader import (
    get_caltech101_dvs_only_dataloaders,
    get_caltech101_dataloaders,
    Caltech101DVSOnlyDataset,
    Caltech101Dataset
)

__all__ = [
    'BaselineVGGSNN',
    'BaselineSpikformer',
    'get_caltech101_dvs_only_dataloaders',
    'get_caltech101_dataloaders',
    'Caltech101DVSOnlyDataset',
    'Caltech101Dataset',
]

__version__ = '1.0.0'
__author__ = 'Based on ESVAE framework'
