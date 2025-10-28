# -*- coding: utf-8 -*-
"""
Preprocessing Module
预处理模块

提供VLM特征提取和语义边缘生成的批处理脚本
"""

from .extract_vlm_features import extract_caltech101_vlm_features
from .generate_semantic_edges import generate_caltech101_semantic_edges

__all__ = [
    'extract_caltech101_vlm_features',
    'generate_caltech101_semantic_edges',
]

