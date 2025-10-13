# -*- coding: utf-8 -*-
"""
ESTL: Structure-Guided DVS Classification with Cross-Attention and Contrastive Learning
DVS动态查询RGB结构知识，通过对比学习缓解样本量不匹配

核心特性:
1. Cross-Attention融合: DVS主动查询RGB结构信息
2. 对比学习: 拉近同类RGB-DVS特征，缓解样本量不匹配
3. 支持更多epoch: DVS样本少(10k)，可训练200+ epoch

训练流程（三阶段）:
=== 阶段1: 生成边缘数据集 ===
python preprocess_edges.py

=== 阶段2: RGB结构分支预训练 ===
python pretrain_edge.py --epochs 50 --batch_size 64 --lr 0.001
保存: /home/user/kpm/kpm/results/ESTL/edge-branch/checkpoints/best_edge_branch.pth

=== 阶段3: 融合训练（Cross-Attention + 对比学习）===
python train.py --epochs 200 --use_contrastive --contrastive_weight 0.3 \\
    --pretrained_edge /path/to/best_edge_branch.pth --freeze_edge
保存: /home/user/kpm/kpm/results/ESTL/fusion/checkpoints/
"""

from .models import (
    EdgeGuidedVGGSNN,
    BaselineVGGSNN,
    SNND2CAFFusion,
    CrossModalAttentionFusion,
    VGGEdgeBranch,
    VGGDVSBranch
)

from .trainer import EdgeGuidedTrainer, ContrastiveLoss
from .edge_pretrainer import EdgePretrainer

__all__ = [
    'EdgeGuidedVGGSNN',
    'BaselineVGGSNN',
    'SNND2CAFFusion',
    'CrossModalAttentionFusion',
    'VGGEdgeBranch',
    'VGGDVSBranch',
    'EdgeGuidedTrainer',
    'ContrastiveLoss',
    'EdgePretrainer',
]

__version__ = '3.0.0'
__author__ = 'Based on ESEG framework'
