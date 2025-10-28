# -*- coding: utf-8 -*-
"""
VLM-Guided Edge Detection Module
VLM引导的边缘检测模块

提供VLM特征提取、语义边缘生成等功能
"""

from .feature_extractor import VLMFeatureExtractor
from .semantic_edge import SemanticEdgeGenerator, MultiScaleEdgeDetector
from .vlm_guided_trainer import VLMGuidedDVSTrainer

__all__ = [
    'VLMFeatureExtractor',
    'SemanticEdgeGenerator',
    'MultiScaleEdgeDetector',
    'VLMGuidedDVSTrainer',
]

