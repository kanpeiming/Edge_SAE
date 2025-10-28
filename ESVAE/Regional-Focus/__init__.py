# -*- coding: utf-8 -*-
"""
Regional Focus Module for RGB-to-DVS Transfer Learning

多层级区域关注模块，用于改进RGB到DVS的迁移学习。

核心改进：
1. DVS输入从2通道改为3通道（两通道各取0.5堆叠到第三通道）
2. 在主干网每一层都施加区域关注约束来引导对齐
3. 使用相同的目标函数进行各层级对齐
4. 自适应学习各层的重要性权重

主要组件：
- RegionalFocusModule: 基础区域聚焦模块
- EnhancedVGGSNNWithRegionalFocus: 集成多层级区域关注的VGGSNN
- VGGSNN_RegionalFocus: 兼容原有接口的增强版模型

使用示例：
```python
from Regional_Focus import VGGSNN_RegionalFocus

model = VGGSNN_RegionalFocus(
    cls_num=10,
    img_shape=32,
    use_regional_focus=True,
    regional_focus_config={
        'similarity_type': 'cosine',
        'alpha': 0.5
    }
)
```
"""

from .regional_focus_module import (
    RegionalFocusModule,
    AdaptiveWeightedAlignmentLoss,
    MultiScaleRegionalFocus,
    EnhancedVGGSNNWithRegionalFocus
)

from .enhanced_model import VGGSNN_RegionalFocus

__all__ = [
    'RegionalFocusModule',
    'AdaptiveWeightedAlignmentLoss',
    'MultiScaleRegionalFocus',
    'EnhancedVGGSNNWithRegionalFocus',
    'VGGSNN_RegionalFocus'
]

__version__ = '2.0.0'
__author__ = 'QgZhan'
__email__ = 'zhanqg@foxmail.com'

