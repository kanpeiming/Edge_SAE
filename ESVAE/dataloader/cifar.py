# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: cifar.py
@time: 2022/4/19 11:19
"""

import os
import bisect
import random
import torch
import numpy as np
from collections import Counter
from PIL import Image
# 13行的导入有一些问题
# _utils 856行,可能是torch2.4.1没有这种方法，因此在本文档中定义
# from torch._utils import _accumulate
from .dataloader_utils import (
    DataLoaderX, Cutout, CIFAR10Policy, RGBToGrayscale3Channel,
    DVSResize, DVSAugment, DVSAugmentCIFAR10, split_to_train_test_set
)
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from spikingjelly.datasets import cifar10_dvs
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

# your own data dir
USER_NAME = 'kpm'
DIR = {'CIFAR10': f'/home/user/{USER_NAME}/kpm/Dataset/CIFAR10/cifar10',
       'CIFAR10DVS': f'/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat',
       'CIFAR10DVS_CATCH': f'/home/user/{USER_NAME}/kpm/Dataset/CIFAR10/CIFAR10DVS_dst_cache',
       }


def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def get_tl_cifar10(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0, val_ratio=0.0, use_cutout=False, cutout_length=16, use_grayscale=False):
    """
    获取RGB到DVS迁移学习的CIFAR10数据加载器
    
    Args:
        batch_size: 批次大小
        train_set_ratio: RGB训练集使用比例
        dvs_train_set_ratio: DVS训练集使用比例
        val_ratio: 验证集比例（从DVS训练集中划分），默认0.0与tl.py保持一致
        use_cutout: 是否使用Cutout数据增强
        cutout_length: Cutout的长度
        use_grayscale: 是否将RGB转换为灰度图（保持三通道）
    
    Returns:
        train_dataloader: 训练数据加载器（RGB+DVS配对数据，用于迁移学习）
        val_dataloader: 验证数据加载器（仅DVS数据，用于验证DVS分类性能）
        test_dataloader: 测试数据加载器（仅DVS数据）
    """
    # 构建RGB训练变换序列 - 与tl.py保持一致使用32×32
    rgb_transforms = [
        transforms.Resize(48),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
    ]
    
    # 可选添加灰度转换（在数据增强之前）
    if use_grayscale:
        rgb_transforms.append(RGBToGrayscale3Channel())
    
    rgb_transforms.extend([
        # CIFAR10Policy(),  # AutoAugment策略
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
    ])
    
    # 可选添加Cutout数据增强
    if use_cutout:
        rgb_transforms.append(Cutout(n_holes=1, length=cutout_length))
        
    rgb_trans_train = transforms.Compose(rgb_transforms)
    # DVS数据transform - 与tl.py保持一致使用32×32
    dvs_trans = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        # DVS数据通常是二值的，使用简单的归一化
        # transforms.Normalize((0.5, 0.5), (0.5, 0.5)),  # 将[0,1]映射到[-1,1]
    ])

    # 计算实际用于配对训练的DVS比例
    # 修复：只有在实际使用验证集时才减少DVS训练数据
    if val_ratio > 0.0:
        actual_dvs_train_ratio = dvs_train_set_ratio * (1 - val_ratio)
    else:
        # 与tl.py保持一致：不使用验证集时，使用全部DVS训练数据
        actual_dvs_train_ratio = dvs_train_set_ratio
    
    # 创建迁移学习训练数据（RGB+DVS配对）
    tl_train_data = TLCIFAR10(DIR['CIFAR10'], DIR['CIFAR10DVS'], train=True, dvs_train_set_ratio=actual_dvs_train_ratio,
                              transform=rgb_trans_train, dvs_transform=dvs_trans, download=True)
    
    # 按train_set_ratio划分RGB训练集
    if train_set_ratio < 1.0:
        n_train = len(tl_train_data)
        split = int(n_train * train_set_ratio)
        print(n_train, split)
        tl_train_data = my_random_split(tl_train_data, [split, n_train - split],
                                        generator=torch.Generator().manual_seed(1000))

    # 创建纯DVS数据集用于验证和测试
    dvs_test_data = DVSCifar10v1(os.path.join(DIR['CIFAR10DVS'], 'test'), train=False, transform=False)
    
    # 只在需要验证集时才划分DVS训练数据
    if val_ratio > 0.0:
        dvs_train_full = DVSCifar10v1(os.path.join(DIR['CIFAR10DVS'], 'train'), train=True, transform=True)
        
        # 按dvs_train_set_ratio划分DVS训练集
        if dvs_train_set_ratio < 1.0:
            n_dvs_train = len(dvs_train_full)
            dvs_split = int(n_dvs_train * dvs_train_set_ratio)
            dvs_train_full = my_random_split(dvs_train_full, [dvs_split, n_dvs_train - dvs_split],
                                            generator=torch.Generator().manual_seed(1000))
        
        # 从DVS训练集中划分验证集
        n_dvs_train_final = len(dvs_train_full)
        n_dvs_val = int(n_dvs_train_final * val_ratio)
        n_dvs_train_actual = n_dvs_train_final - n_dvs_val
        
        print(f"DVS数据划分详情:")
        print(f"  原始DVS训练集总数: {n_dvs_train_final}")
        print(f"  验证集比例: {val_ratio}")
        print(f"  用于RGB-DVS配对训练: {n_dvs_train_actual} (避免数据泄露)")
        print(f"  DVS验证集: {n_dvs_val} (独立验证)")
        print(f"  实际配对训练比例: {actual_dvs_train_ratio:.3f}")
        
        dvs_train_data, dvs_val_data = my_random_split(
            dvs_train_full,
            [n_dvs_train_actual, n_dvs_val],
            generator=torch.Generator().manual_seed(1000)
        )
        
        # 验证：纯DVS数据，用于验证DVS分类性能
        val_dataloader = DataLoaderX(dvs_val_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False,
                                     pin_memory=True)
    else:
        # 与tl.py保持一致：不使用验证集时，返回None
        val_dataloader = None
        print(f"DVS数据划分详情 (与tl.py一致):")
        print(f"  不划分验证集，使用全部DVS训练数据")
        print(f"  用于RGB-DVS配对训练: {len(tl_train_data.dvs_data)} (实际使用比例: {actual_dvs_train_ratio:.3f})")

    # 创建数据加载器
    # 训练：RGB+DVS配对数据，用于迁移学习
    train_dataloader = DataLoaderX(tl_train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                   pin_memory=True)
    # 测试：纯DVS数据
    test_dataloader = DataLoaderX(dvs_test_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_cifar10(batch_size, train_set_ratio=1.0):
    """
    get the train loader and test loader of cifar10.
    :return: train_loader, test_loader
    """
    trans_train = transforms.Compose([transforms.Resize(48),
                                      transforms.RandomCrop(48, padding=4),
                                      transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                      # CIFAR10Policy(),  # AutoAugment策略 - 已注释以避免影响实验效果
                                      transforms.ToTensor(),
                                      # transforms.RandomGrayscale(),  # 随机变为灰度图
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
                                      # transforms.Normalize((0., 0., 0.), (1, 1, 1)),
                                      # Cutout(n_holes=1, length=16)  # 随机挖n_holes个length * length的洞
                                      ])
    trans_test = transforms.Compose([transforms.Resize(48),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_train, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans_test, download=True)

    # take train set by train_set_ratio
    n_train = len(train_data)
    split = int(n_train * train_set_ratio)
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])

    if train_set_ratio < 1.0:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True,
                                       sampler=train_sampler, pin_memory=True)
    else:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                       pin_memory=True)
    test_dataloader = DataLoaderX(test_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader  # , test_dataloader


def get_cifar10_DVS(batch_size, T, split_ratio=0.9, train_set_ratio=1, size=32, encode_type='TET', use_eventrpg=False, eventrpg_mix_prob=0.5):
    """
    get the train loader and test loader of cifar10.
    :param batch_size:
    :param T:
    :param split_ratio: the ratio of train set: test set
    :param train_set_ratio: the real used train set ratio
    :param size:
    :param encode_type:
    :param use_eventrpg: whether to use EventRPG augmentation
    :param eventrpg_mix_prob: EventRPG mix probability
    :return: train_loader, test_loader
    """
    if encode_type == "spikingjelly":
        trans = DVSResize((size, size), T)

        train_set_pth = os.path.join(DIR['CIFAR10DVS_CATCH'], f'train_set_{T}_{split_ratio}_{size}.pt')
        test_set_pth = os.path.join(DIR['CIFAR10DVS_CATCH'], f'test_set_{T}_{split_ratio}_{size}.pt')

        if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
            train_set = torch.load(train_set_pth)
            test_set = torch.load(test_set_pth)
        else:
            origin_set = cifar10_dvs.CIFAR10DVS(root=DIR['CIFAR10DVS'], data_type='frame', frames_number=T,
                                                split_by='number', transform=trans)

            train_set, test_set = split_to_train_test_set(split_ratio, origin_set, 10)
            if not os.path.exists(DIR['CIFAR10DVS_CATCH']):
                os.makedirs(DIR['CIFAR10DVS_CATCH'])
            torch.save(train_set, train_set_pth)
            torch.save(test_set, test_set_pth)
    elif encode_type == "TET":
        path = '/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat'
        train_path = path + '/train'
        test_path = path + '/test'
        train_set = DVSCifar10v1(root=train_path, train=True, transform=True, use_eventrpg=use_eventrpg, eventrpg_mix_prob=eventrpg_mix_prob)
        test_set = DVSCifar10v1(root=test_path, train=False, transform=False, use_eventrpg=False)
    elif encode_type == "3_channel":
        path = '/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat'
        train_path = path + '/train'
        test_path = path + '/test'
        train_set = Channel_3_DVSCifar10(root=train_path)
        test_set = Channel_3_DVSCifar10(root=test_path)

    # take train set by train_set_ratio
    n_train = len(train_set)
    split = int(n_train * train_set_ratio)
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])
    # valid_sampler = SubsetRandomSampler(indices[split:])

    # generate dataloader
    # train_data_loader = DataLoaderX(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,
    #                                 num_workers=8, pin_memory=True)
    train_data_loader = DataLoaderX(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=False,
                                    sampler=train_sampler, num_workers=8,
                                    pin_memory=True)  # SubsetRandomSampler 自带shuffle，不能重复使用
    test_data_loader = DataLoaderX(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False,
                                   num_workers=8, pin_memory=True)

    return train_data_loader, test_data_loader


def get_cifar100(batch_size):
    """
    get the train loader and test loader of cifar100.
    :return: train_loader, test_loader
    """
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                       std=[n / 255. for n in [68.2, 65.4, 70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                     std=[n / 255. for n in [68.2, 65.4, 70.4]])])

    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True)

    train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_dataloader = DataLoaderX(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_dataloader, test_dataloader


class DVSCifar10(Dataset):
    # This code is form https://github.com/Gus-Lab/temporal_efficient_training
    def __init__(self, root, train=True, transform=True, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(32, 32))  # 32 32
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}_mat.pt'.format(index))
        # if self.train:
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t, ...]))))
        data = torch.stack(new_data, dim=0)

        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


class DVSCifar10v1(Dataset):
    def __init__(self, root, train=True, transform=True, target_transform=None, use_nda=False, use_eventrpg=False, eventrpg_mix_prob=0.5):
        """
        DVS CIFAR10数据集
        
        Args:
            root: 数据根目录
            train: 是否为训练集
            transform: 是否使用数据增强 (传统方法: flip + roll)
            target_transform: 目标变换
            use_nda: 是否使用NDA_SNN的数据增强方法 (roll/rotate/shear随机选择)
            use_eventrpg: 是否使用EventRPG的数据增强方法 (几何增强+RPGMix)
            eventrpg_mix_prob: EventRPG的RPGMix概率
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.use_nda = use_nda
        self.use_eventrpg = use_eventrpg
        self.resize = transforms.Resize(size=(48, 48))
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

        # 构建索引映射：每个元素是（文件路径, 类别标签）
        self.samples = []
        self.class_to_idx = {}
        self._build_index()
        
        # 初始化NDA增强器
        if self.use_nda:
            self.nda_augment = DVSAugmentCIFAR10(apply_prob=1.0)
        
        # 初始化EventRPG增强器
        if self.use_eventrpg:
            from .eventrpg_augment import EventRPGAugment
            self.eventrpg_augment = EventRPGAugment(img_size=48, mix_prob=eventrpg_mix_prob)

    def _build_index(self):
        class_dirs = sorted(os.listdir(self.root))
        for label_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_path):
                self.class_to_idx[class_dir] = label_idx
                for filename in os.listdir(class_path):
                    if filename.endswith('.pt'):
                        filepath = os.path.join(class_path, filename)
                        self.samples.append((filepath, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]  # label 已从文件夹名称映射
        data = torch.load(filepath, weights_only=True)  # 直接加载张量
        target = label  # 标签来自文件夹名称
        
        # 优化：批量resize，避免Tensor→PIL→Tensor转换
        if data.shape[2] != 48 or data.shape[3] != 48:
            data = torch.nn.functional.interpolate(
                data.float(),  # (T, C, H, W)
                size=(48, 48),
                mode='bilinear',
                align_corners=False
            )

        if self.transform:
            if self.use_eventrpg:
                # 使用EventRPG的增强方法 (几何增强+RPGMix)
                data = self.eventrpg_augment(data)
            elif self.use_nda:
                # 使用NDA_SNN的增强方法 (roll/rotate/shear随机选择)
                data = self.nda_augment(data)
            else:
                # 使用传统增强方法 (flip + roll)
                flip = random.random() > 0.5
                if flip:
                    data = torch.flip(data, dims=(3,))
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, torch.tensor(target)


class Channel_3_DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=True, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(32, 32))  # 32 32
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        T, C, H, W = data.shape
        # if self.train:
        new_data = []
        for t in range(T):
            tmp = data[t, ...]  # (2, H, W)
            tmp = torch.cat((tmp, torch.zeros(1, H, W)), dim=0)  # (3, H, W)
            mask = (torch.randn((H, W)) > 0).to(data)
            tmp[2].data = tmp[0].data * mask + tmp[1].data * (1 - mask)
            new_data.append(self.tensorx(self.resize(self.imgx(tmp))))
        data = torch.stack(new_data, dim=0)

        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


class TLCIFAR10(datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            dvs_root: str,
            train: bool = True,
            dvs_train_set_ratio: float = 1.0,
            transform: Optional[Callable] = None,
            dvs_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(TLCIFAR10, self).__init__(root=root, train=train, transform=transform,
                                        target_transform=target_transform, download=download)

        self.train = train  # training set or test set
        self.dvs_train_set_ratio = dvs_train_set_ratio
        self.dvs_transform = dvs_transform
        self.imgx = transforms.ToPILImage()
        if self.train:
            self.dvs_root = os.path.join(dvs_root, 'train')
        else:
            self.dvs_root = os.path.join(dvs_root, 'test')

        """
        准备RGB数据
        """
        # 对rgb数据按label排序
        sort_idx = sorted(range(len(self.targets)), key=lambda k: self.targets[k])
        self.data = list(np.array(self.data)[sort_idx])
        self.targets = list(np.array(self.targets)[sort_idx])

        self.cumulative_sizes = self.cumsum(self.targets)

        """
        准备DVS数据
        """
        print(self.dvs_root)
        dvs_class_list = sorted(os.listdir(self.dvs_root))
        self.dvs_data = []
        self.dvs_targets = []
        for dvs_class in dvs_class_list:
            dvs_class_path = os.path.join(self.dvs_root, dvs_class)
            file_list = sorted(os.listdir(dvs_class_path))
            file_range = int(self.dvs_train_set_ratio * len(file_list))
            for file_name in file_list[: file_range]:
                self.dvs_data.append(os.path.join(dvs_class_path, file_name))
                self.dvs_targets.append(self.class_to_idx[dvs_class])
        print(len(self.dvs_data), len(self.dvs_targets))
        self.dvs_cumulative_sizes = self.cumsum(self.dvs_targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # 获取dvs图像的索引
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)  # 输出索引对应rgb图像的类别+1
            dvs_index_start = self.dvs_cumulative_sizes[dataset_idx - 1]  # 得到该类别对应dvs图像的开始索引
            dvs_index_end = self.dvs_cumulative_sizes[dataset_idx]  # 得到该类别对应dvs图像的结束索引
            dvs_index = dvs_index_start + (index - self.cumulative_sizes[dataset_idx - 1]) % (
                int((
                            dvs_index_end - dvs_index_start) * self.dvs_train_set_ratio))  # 利用求余，得到在该类别循环0次或多次后的最终索引，self.dvs_train_set_ratio可控制选取dvs图像的比例

            # dvs图像的transform
            dvs_img = torch.load(self.dvs_data[dvs_index], weights_only=True)
            if self.dvs_transform is not None:
                dvs_img = self.dvs_trans(dvs_img)

            return (img, dvs_img), target
        else:
            # dvs图像的transform
            dvs_img = torch.load(self.dvs_data[index], weights_only=True)
            if self.dvs_transform is not None:
                dvs_img = self.dvs_trans(dvs_img)
            target = self.dvs_targets[index]  # 输入索引对应dvs图像的类别

            return dvs_img, target

    def __len__(self) -> int:
        if self.train:
            return len(self.data)
        else:
            return len(self.dvs_data)

    def dvs_trans(self, dvs_img):
        transformed_dvs_img = []
        for t in range(dvs_img.size(0)):
            data = self.imgx(dvs_img[t, ...])
            transformed_dvs_img.append(self.dvs_transform(data))
        dvs_img = torch.stack(transformed_dvs_img, dim=0)

        if self.train:
            flip = random.random() > 0.5
            if flip:
                dvs_img = torch.flip(dvs_img, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            dvs_img = torch.roll(dvs_img, shifts=(off1, off2), dims=(2, 3))
        return dvs_img

    @staticmethod
    def cumsum(targets):
        result = Counter(targets)
        r, s = [0], 0
        for e in range(len(result)):
            l = result[e]
            r.append(l + s)
            s += l
        return r

    def get_len(self):
        return len(self.data), len(self.dvs_data)


class MySubset(Subset):
    def get_len(self):
        return len(self.indices), len(self.dataset.dvs_data)


def my_random_split(dataset, lengths, generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist()
    return [MySubset(dataset, indices[offset - length: offset]) for offset, length in
            zip(_accumulate(lengths), lengths)]


# ============================================================================
# CIFAR10 Edge to DVS Transfer Learning Dataset
# ============================================================================

class EdgeDatasetCIFAR10(Dataset):
    """
    加载预处理的CIFAR10 Edge数据（按类别组织）
    
    CIFAR10有10个类别，标签从0-9
    """
    def __init__(self, root, sample_ratio=1.0):
        self.root = root
        self.samples = []
        self.class_to_idx = {}
        
        # CIFAR10类别名称（按标签顺序）
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 收集所有类别的样本
        class_dirs = sorted(os.listdir(root))
        
        for class_dir in class_dirs:
            class_path = os.path.join(root, class_dir)
            if os.path.isdir(class_path):
                # 获取类别索引
                if class_dir in cifar10_classes:
                    label = cifar10_classes.index(class_dir)
                    self.class_to_idx[class_dir] = label
                    
                    # 收集该类别的所有样本
                    pt_files = sorted([f for f in os.listdir(class_path) if f.endswith('.pt')])
                    
                    # 按比例采样
                    if sample_ratio < 1.0:
                        n_samples = int(len(pt_files) * sample_ratio)
                        pt_files = pt_files[:n_samples]
                    
                    for pt_file in pt_files:
                        filepath = os.path.join(class_path, pt_file)
                        self.samples.append(filepath)
        
        print(f"  EdgeDatasetCIFAR10: {len(self.samples)} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        filepath = self.samples[index]
        # 加载原始数据和标签
        edge, label = torch.load(filepath, weights_only=True)  # (2, H, W), label
        
        # 确保标签是标量
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        # 归一化处理：对Edge数据进行标准化
        # Edge数据通常是Sobel边缘响应，值域约为[0, 1]
        # 使用简单的归一化将其映射到标准范围
        if edge.dtype != torch.float32:
            edge = edge.float()
        
        # 对每个通道进行归一化（2通道：水平和垂直边缘）
        # 使用与CIFAR10类似的归一化参数
        mean = torch.tensor([0.5, 0.5]).view(2, 1, 1)
        std = torch.tensor([0.5, 0.5]).view(2, 1, 1)
        edge = (edge - mean) / std
        
        return edge, label


class TLEdge2DVSDatasetCIFAR10(Dataset):
    """
    CIFAR10 Edge到DVS迁移学习数据集
    使用方案1（TLCIFAR10风格）：基于累积索引的动态配对
    
    优势：
    - 内存占用低（只存储累积数组）
    - 初始化快（无需遍历构建索引映射）
    - 代码简洁
    """
    def __init__(self, edge_root, dvs_dataset, edge_ratio=1.0):
        print("  初始化TLEdge2DVSDatasetCIFAR10（方案1：累积索引动态配对）...")
        
        self.edge_dataset = EdgeDatasetCIFAR10(edge_root, edge_ratio)
        self.dvs_dataset = dvs_dataset
        
        # 按标签排序Edge数据（关键步骤）
        print("  对Edge数据按标签排序...")
        edge_samples_with_labels = []
        for filepath in self.edge_dataset.samples:
            edge, label = torch.load(filepath, weights_only=True)
            if isinstance(label, torch.Tensor):
                label = label.item()
            edge_samples_with_labels.append((filepath, label))
        
        # 按标签排序
        edge_samples_with_labels.sort(key=lambda x: x[1])
        self.edge_samples = [x[0] for x in edge_samples_with_labels]
        self.edge_labels = [x[1] for x in edge_samples_with_labels]
        
        # 构建Edge累积大小数组
        self.edge_cumulative_sizes = self.cumsum(self.edge_labels)
        
        # 构建DVS累积大小数组
        print("  构建DVS累积索引...")
        if hasattr(self.dvs_dataset, 'samples'):
            # DVSCifar10v1有samples属性，直接提取标签
            dvs_labels = [label for _, label in self.dvs_dataset.samples]
        else:
            # 备选方案：遍历加载（较慢）
            print("  警告：DVS数据集没有samples属性，遍历获取标签...")
            dvs_labels = []
            for idx in range(len(self.dvs_dataset)):
                _, label = self.dvs_dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                dvs_labels.append(label)
        
        self.dvs_cumulative_sizes = self.cumsum(dvs_labels)
        
        print(f"  ✓ Edge样本数: {len(self.edge_samples)}")
        print(f"  ✓ DVS样本数: {len(self.dvs_dataset)}")
        print(f"  ✓ 使用累积索引动态配对（类内循环采样）")
    
    @staticmethod
    def cumsum(targets):
        """计算累积和数组（与TLCIFAR10相同的实现）"""
        result = Counter(targets)
        r, s = [0], 0
        for e in range(len(result)):
            l = result[e]
            r.append(l + s)
            s += l
        return r
    
    def __len__(self):
        return len(self.edge_samples)
    
    def __getitem__(self, index):
        # 加载Edge样本
        edge_filepath = self.edge_samples[index]
        edge, edge_label = torch.load(edge_filepath, weights_only=True)
        if isinstance(edge_label, torch.Tensor):
            edge_label = edge_label.item()
        
        # 使用bisect找到Edge样本的类别（与TLCIFAR10相同的逻辑）
        dataset_idx = bisect.bisect_right(self.edge_cumulative_sizes, index)
        
        # 找到该类别对应的DVS样本范围
        dvs_index_start = self.dvs_cumulative_sizes[dataset_idx - 1]
        dvs_index_end = self.dvs_cumulative_sizes[dataset_idx]
        
        # 类内循环采样（取模运算）
        dvs_index = dvs_index_start + (index - self.edge_cumulative_sizes[dataset_idx - 1]) % (
            dvs_index_end - dvs_index_start)
        
        # 加载对应的DVS样本
        dvs_data, dvs_label = self.dvs_dataset[dvs_index]
        if isinstance(dvs_label, torch.Tensor):
            dvs_label = dvs_label.item()
        
        # 验证标签匹配（调试用）
        assert edge_label == dvs_label, f"标签不匹配: Edge={edge_label}, DVS={dvs_label}"
        
        return (edge, dvs_data), (edge_label, dvs_label)
    
    def get_len(self):
        return [len(self.edge_samples), len(self.dvs_dataset)]


def get_edge2dvs_cifar10(batch_size, edge_root, dvs_root, 
                        edge_ratio=1.0, dvs_ratio=1.0, 
                        num_workers=8, img_size=48, split_ratio=0.9):
    """
    获取CIFAR10 Edge到DVS迁移学习的数据加载器
    
    Args:
        batch_size: 批次大小
        edge_root: Edge数据根目录（按类别组织）
        dvs_root: DVS数据根目录（使用DVSCifar10v1加载方式）
        edge_ratio: Edge数据使用比例
        dvs_ratio: DVS数据使用比例
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        split_ratio: 当没有train/test划分时的训练集比例
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    from torch.utils.data.sampler import SubsetRandomSampler
    
    # 加载Edge数据集
    edge_dataset = EdgeDatasetCIFAR10(edge_root, edge_ratio)
    print(f"✓ Edge训练集: {len(edge_dataset)} 样本")
    
    # 检查Edge数据集的标签范围
    if len(edge_dataset) > 0:
        sample_edge, sample_label = edge_dataset[0]
        print(f"  Edge数据形状示例: {sample_edge.shape}")
        print(f"  Edge标签示例: {sample_label}")
        
        # 采样检查标签范围 - 均匀采样以覆盖所有类别
        labels = []
        check_samples = min(100, len(edge_dataset))
        # 均匀采样整个数据集，而不是只采样前100个
        step = max(1, len(edge_dataset) // check_samples)
        for i in range(0, len(edge_dataset), step):
            _, label = edge_dataset[i]
            labels.append(label)
            if len(labels) >= check_samples:
                break
        print(f"  Edge标签范围（采样检查）: [{min(labels)}, {max(labels)}]")
        print(f"  Edge唯一标签数: {len(set(labels))}")
    
    # 加载DVS数据集
    train_path = os.path.join(dvs_root, 'train')
    test_path = os.path.join(dvs_root, 'test')
    
    has_split = os.path.exists(train_path) and os.path.exists(test_path)
    
    if has_split:
        # 使用预先划分的train/test
        dvs_train_dataset = DVSCifar10v1(train_path, train=True, transform=True, 
                                        use_nda=False, use_eventrpg=False)
        dvs_test_dataset = DVSCifar10v1(test_path, train=False, transform=False, 
                                       use_nda=False, use_eventrpg=False)
        
        print(f"✓ DVS训练集: {len(dvs_train_dataset)} 样本")
        print(f"✓ DVS测试集: {len(dvs_test_dataset)} 样本")
        
        # DVS训练集采样
        if dvs_ratio < 1.0:
            n_dvs = len(dvs_train_dataset)
            indices = list(range(n_dvs))
            random.shuffle(indices)
            dvs_train_indices = indices[:int(n_dvs * dvs_ratio)]
            dvs_train_sampler = SubsetRandomSampler(dvs_train_indices)
        else:
            dvs_train_sampler = None
        
        # 创建Edge2DVS训练数据集
        train_dataset = TLEdge2DVSDatasetCIFAR10(edge_root, dvs_train_dataset, edge_ratio)
        
        # 训练数据加载器
        train_loader = DataLoaderX(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=(dvs_train_sampler is None),
            sampler=None,  # TLEdge2DVSDatasetCIFAR10自己处理采样
            num_workers=num_workers, 
            drop_last=True, 
            pin_memory=True
        )
        
        # 测试数据加载器（只用DVS数据）
        test_loader = DataLoaderX(
            dvs_test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            drop_last=False, 
            pin_memory=True
        )
    else:
        raise ValueError(f"CIFAR10DVS必须有train/test划分，但在{dvs_root}中未找到")
    
    # 显示最终配对信息
    print(f"\n最终数据集配对:")
    edge_len, dvs_len = train_loader.dataset.get_len()
    print(f"  Edge样本: {edge_len}")
    print(f"  DVS样本: {dvs_len}")
    if edge_len != dvs_len:
        print(f"  ⚠️  样本数不一致，训练时使用循环采样（取模）策略")
    
    return train_loader, test_loader
