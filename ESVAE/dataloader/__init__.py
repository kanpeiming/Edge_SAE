# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: __init__.py.py
@time: 2022/4/19 11:09
"""

from .cifar import *
from .mnist import *
from .caltech101 import *
from .dataloader_utils import (
    DVSAugment,
    DVSAugmentCaltech101,
    DVSAugmentCIFAR10,
    Cutout,
    cutmix_data,
    mixup_data,
    mixup_criterion,
    CIFAR10Policy,
    ImageNetPolicy,
    RGBToGrayscale3Channel,
    DataLoaderX
)

# from .cinic import get_tl_cinic10_wo_cifar10
# from .imagenet2caltech import *
# from .imagenet import *
# from .Office31 import get_small_office31
