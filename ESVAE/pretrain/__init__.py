# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: __init__.py.py
@time: 2022/4/19 11:10
"""

from .rgb_only_trainer import RGBOnlyTrainer
from .edge2dvs_trainer import AlignmentTLTrainer_Edge2DVS

__all__ = ['RGBOnlyTrainer', 'AlignmentTLTrainer_Edge2DVS']
