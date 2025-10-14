# -*- coding: utf-8 -*-
"""
Enhanced DVS Data Augmentation for Edge-Guided Training
包含多种DVS特定的数据增强技术
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class DVSAugmentation:
    """
    DVS-specific data augmentation techniques
    """
    
    def __init__(self, 
                 enable_flip=True,
                 enable_rotation=True, 
                 enable_translation=True,
                 enable_noise=True,
                 enable_temporal_aug=True,
                 flip_prob=0.5,
                 rotation_range=15,
                 translation_range=5,
                 noise_std=0.1,
                 temporal_jitter_range=2):
        
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_translation = enable_translation
        self.enable_noise = enable_noise
        self.enable_temporal_aug = enable_temporal_aug
        
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.temporal_jitter_range = temporal_jitter_range
    
    def __call__(self, dvs_data):
        """
        Apply augmentations to DVS data
        
        Args:
            dvs_data: (T, C, H, W) - DVS event data
            
        Returns:
            augmented_data: (T, C, H, W) - Augmented DVS data
        """
        T, C, H, W = dvs_data.shape
        augmented = dvs_data.clone()
        
        # 1. Random horizontal flip
        if self.enable_flip and random.random() < self.flip_prob:
            augmented = torch.flip(augmented, dims=(3,))
        
        # 2. Random rotation (small angles)
        if self.enable_rotation and random.random() < 0.3:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            augmented = self._rotate_tensor(augmented, angle)
        
        # 3. Random translation
        if self.enable_translation and random.random() < 0.5:
            dx = random.randint(-self.translation_range, self.translation_range)
            dy = random.randint(-self.translation_range, self.translation_range)
            augmented = self._translate_tensor(augmented, dx, dy)
        
        # 4. Add noise
        if self.enable_noise and random.random() < 0.3:
            noise = torch.randn_like(augmented) * self.noise_std
            augmented = torch.clamp(augmented + noise, 0, 1)
        
        # 5. Temporal augmentation
        if self.enable_temporal_aug and random.random() < 0.4:
            augmented = self._temporal_jitter(augmented)
        
        return augmented
    
    def _rotate_tensor(self, tensor, angle):
        """Apply rotation to tensor"""
        # 使用仿射变换进行旋转
        angle_rad = angle * np.pi / 180
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # 创建旋转矩阵
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=tensor.dtype, device=tensor.device)
        
        # 对每个时间步应用旋转
        rotated = []
        for t in range(tensor.size(0)):
            frame = tensor[t:t+1]  # (1, C, H, W)
            rotated_frame = F.affine_grid(
                rotation_matrix.unsqueeze(0), 
                frame.size(), 
                align_corners=False
            )
            rotated_frame = F.grid_sample(
                frame, 
                rotated_frame, 
                mode='bilinear', 
                padding_mode='zeros',
                align_corners=False
            )
            rotated.append(rotated_frame)
        
        return torch.cat(rotated, dim=0)
    
    def _translate_tensor(self, tensor, dx, dy):
        """Apply translation to tensor"""
        if dx == 0 and dy == 0:
            return tensor
        
        # 使用roll进行平移
        return torch.roll(tensor, shifts=(dx, dy), dims=(2, 3))
    
    def _temporal_jitter(self, tensor):
        """Apply temporal jittering"""
        T = tensor.size(0)
        
        # 随机选择时间步进行重排
        if T > 1:
            indices = list(range(T))
            # 随机交换相邻时间步
            for _ in range(self.temporal_jitter_range):
                if len(indices) > 1:
                    i = random.randint(0, len(indices) - 2)
                    indices[i], indices[i + 1] = indices[i + 1], indices[i]
            
            return tensor[indices]
        
        return tensor


class EnhancedDVSCifar10(Dataset):
    """
    Enhanced DVS CIFAR10 dataset with comprehensive augmentation
    Now supports returning both RGB and DVS data for edge-guided training
    Handles mismatch between RGB and DVS data sizes using modulo indexing per class
    
    改进策略：
    1. 训练集始终以RGB为准(use_rgb_size=True) - 充分利用RGB数据
    2. DVS通过类内循环匹配 - 避免单个DVS样本过度重复
    3. 验证集以DVS为准 - 真实反映DVS泛化能力
    """
    
    def __init__(self, root, train=True, augmentation=True, target_transform=None, rgb_root=None, dvs_train_set_ratio=1.0, split='train', val_split=0.1, use_rgb_size=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.augmentation = augmentation
        self.target_transform = target_transform
        self.rgb_root = rgb_root  # Path to RGB CIFAR10 dataset
        self.dvs_train_set_ratio = dvs_train_set_ratio
        self.split = split  # 'train', 'val', or 'test'
        self.val_split = val_split  # 验证集比例
        # 改进：训练集强制以RGB为准，充分利用RGB数据
        self.use_rgb_size = use_rgb_size if split != 'train' else True  # 训练集强制True
        
        # 基础变换
        self.resize = transforms.Resize(size=(32, 32))
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()
        
        # RGB normalization (ImageNet stats)
        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 数据增强
        if self.augmentation and self.train:
            self.augmentor = DVSAugmentation(
                enable_flip=True,
                enable_rotation=True,
                enable_translation=True,
                enable_noise=True,
                enable_temporal_aug=True,
                flip_prob=0.5,
                rotation_range=10,
                translation_range=3,
                noise_std=0.05,
                temporal_jitter_range=1
            )
        else:
            self.augmentor = None
        
        # 构建DVS索引
        self.dvs_samples = []
        self.dvs_targets = []
        self.class_to_idx = {}
        self._build_dvs_index()
        
        # Load RGB CIFAR10 if path is provided
        self.rgb_data = None
        self.rgb_targets = None
        if rgb_root is not None:
            from torchvision.datasets import CIFAR10
            rgb_dataset = CIFAR10(
                root=rgb_root, 
                train=train, 
                download=False
            )
            # 按类别排序RGB数据（与TLCIFAR10一致）
            self.rgb_data = rgb_dataset.data
            self.rgb_targets = rgb_dataset.targets
            sort_idx = sorted(range(len(self.rgb_targets)), key=lambda k: self.rgb_targets[k])
            self.rgb_data = [self.rgb_data[i] for i in sort_idx]
            self.rgb_targets = [self.rgb_targets[i] for i in sort_idx]
            
            # 计算累积大小用于快速查找
            self.rgb_cumulative_sizes = self._cumsum(self.rgb_targets)
        
        # 计算DVS累积大小
        self.dvs_cumulative_sizes = self._cumsum(self.dvs_targets)
    
    def _build_dvs_index(self):
        """构建DVS样本索引（按类别排序），支持训练/验证集划分"""
        class_dirs = sorted(os.listdir(self.root))
        for label_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_path):
                self.class_to_idx[class_dir] = label_idx
                file_list = sorted([f for f in os.listdir(class_path) if f.endswith('.pt')])
                
                # 如果是测试集，使用全部数据
                if self.split == 'test':
                    selected_files = file_list
                else:
                    # 训练/验证集划分
                    total_files = int(self.dvs_train_set_ratio * len(file_list))
                    val_count = int(total_files * self.val_split)
                    train_count = total_files - val_count
                    
                    if self.split == 'train':
                        selected_files = file_list[:train_count]
                    elif self.split == 'val':
                        selected_files = file_list[train_count:total_files]
                    else:
                        selected_files = file_list[:total_files]
                
                for filename in selected_files:
                    filepath = os.path.join(class_path, filename)
                    self.dvs_samples.append(filepath)
                    self.dvs_targets.append(label_idx)
    
    def _cumsum(self, targets):
        """计算累积大小（用于按类别查找）"""
        import bisect
        cumulative = []
        count = 0
        current_class = targets[0] if targets else 0
        for i, target in enumerate(targets):
            if target != current_class:
                cumulative.append(count)
                current_class = target
            count += 1
        cumulative.append(count)
        return cumulative
    
    def __len__(self):
        # 改进策略：
        # 训练集：以RGB为准（充分利用50k RGB数据，DVS通过类内循环匹配）
        # 验证集/测试集：以DVS为准（真实评估DVS泛化能力）
        if self.train and self.rgb_data is not None and self.split == 'train':
            # 训练集：始终以RGB数据量为准（充分利用RGB数据）
            return len(self.rgb_data)
        else:
            # 验证集、测试集：以DVS数据量为准（避免虚高精度）
            return len(self.dvs_samples)
    
    def __getitem__(self, index):
        import bisect
        from PIL import Image
        
        if self.train and self.rgb_data is not None and self.split == 'train':
            # 训练集模式：以RGB为准（充分利用RGB数据），DVS通过类内循环匹配
            # 改进：添加随机性，避免固定的DVS样本匹配
            
            # 1. 获取RGB图像和类别
            rgb_img = self.rgb_data[index]
            target = self.rgb_targets[index]
            
            # 转换RGB图像
            rgb_img = Image.fromarray(rgb_img)
            rgb_img = self.tensorx(rgb_img)
            rgb_img = self.rgb_normalize(rgb_img)  # (3, H, W)
            
            # 2. 根据RGB的类别，从DVS数据中获取同类别的样本
            # 找到RGB图像对应的类别
            dataset_idx = bisect.bisect_right(self.rgb_cumulative_sizes, index)
            
            # 获取该类别在RGB中的索引范围
            rgb_class_start = self.rgb_cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
            rgb_offset_in_class = index - rgb_class_start
            
            # 获取该类别的DVS图像索引范围
            dvs_index_start = self.dvs_cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
            dvs_index_end = self.dvs_cumulative_sizes[dataset_idx]
            dvs_class_size = dvs_index_end - dvs_index_start
            
            # 改进：使用模运算+随机扰动，增加DVS样本多样性
            # 基础索引：模运算循环
            base_dvs_offset = rgb_offset_in_class % dvs_class_size
            # 随机扰动：±1范围内随机选择（如果类别有足够样本）
            if dvs_class_size > 3 and self.augmentation:
                random_offset = random.randint(-1, 1)
                final_dvs_offset = (base_dvs_offset + random_offset) % dvs_class_size
            else:
                final_dvs_offset = base_dvs_offset
            
            dvs_index = dvs_index_start + final_dvs_offset
            
            # 3. 加载DVS数据
            dvs_filepath = self.dvs_samples[dvs_index]
            dvs_data = torch.load(dvs_filepath, weights_only=True)
            
            # 处理每一帧DVS数据
            processed_frames = []
            for t in range(dvs_data.size(0)):
                frame = dvs_data[t, ...]  # (C, H, W)
                frame_pil = self.imgx(frame)
                frame_resized = self.resize(frame_pil)
                frame_tensor = self.tensorx(frame_resized)
                processed_frames.append(frame_tensor)
            
            dvs_data = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            
            # 应用数据增强
            if self.augmentor is not None:
                dvs_data = self.augmentor(dvs_data)
            
            # 目标变换
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return (rgb_img, dvs_data), torch.tensor(target)
        
        elif self.train and self.rgb_data is not None:
            # 验证集模式：以DVS为准，返回(RGB, DVS)对
            # index是基于DVS的，需要找到对应类别的RGB样本（循环使用）
            
            # 1. 获取DVS样本和类别
            dvs_filepath = self.dvs_samples[index]
            target = self.dvs_targets[index]
            
            # 2. 根据DVS的类别，从RGB数据中获取同类别的图像（使用模运算循环）
            dvs_class_idx = target
            dvs_class_start = self.dvs_cumulative_sizes[dvs_class_idx - 1] if dvs_class_idx > 0 else 0
            dvs_offset_in_class = index - dvs_class_start
            
            # 获取RGB数据中同类别的索引范围
            rgb_class_start = self.rgb_cumulative_sizes[dvs_class_idx - 1] if dvs_class_idx > 0 else 0
            rgb_class_end = self.rgb_cumulative_sizes[dvs_class_idx]
            rgb_class_size = rgb_class_end - rgb_class_start
            
            # 使用模运算在RGB的该类别内循环
            rgb_index = rgb_class_start + (dvs_offset_in_class % rgb_class_size)
            
            # 获取RGB图像
            rgb_img = self.rgb_data[rgb_index]
            rgb_img = Image.fromarray(rgb_img)
            rgb_img = self.tensorx(rgb_img)
            rgb_img = self.rgb_normalize(rgb_img)  # (3, H, W)
            
            # 3. 加载DVS数据
            dvs_data = torch.load(dvs_filepath, weights_only=True)
            
            # 处理每一帧DVS数据
            processed_frames = []
            for t in range(dvs_data.size(0)):
                frame = dvs_data[t, ...]  # (C, H, W)
                frame_pil = self.imgx(frame)
                frame_resized = self.resize(frame_pil)
                frame_tensor = self.tensorx(frame_resized)
                processed_frames.append(frame_tensor)
            
            dvs_data = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            
            # 应用数据增强
            if self.augmentor is not None:
                dvs_data = self.augmentor(dvs_data)
            
            # 目标变换
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return (rgb_img, dvs_data), torch.tensor(target)
        
        else:
            # 测试模式或无RGB数据：只返回DVS
            dvs_filepath = self.dvs_samples[index]
            target = self.dvs_targets[index]
            
            # 加载DVS数据
            dvs_data = torch.load(dvs_filepath, weights_only=True)
            
            # 处理每一帧DVS数据
            processed_frames = []
            for t in range(dvs_data.size(0)):
                frame = dvs_data[t, ...]  # (C, H, W)
                frame_pil = self.imgx(frame)
                frame_resized = self.resize(frame_pil)
                frame_tensor = self.tensorx(frame_resized)
                processed_frames.append(frame_tensor)
            
            dvs_data = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            
            # 应用数据增强
            if self.augmentor is not None:
                dvs_data = self.augmentor(dvs_data)
            
            # 目标变换
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return dvs_data, torch.tensor(target)


def get_enhanced_cifar10_DVS(batch_size, T=10, split_ratio=0.9, train_set_ratio=1.0, 
                            augmentation=True, num_workers=8, rgb_root=None, dvs_root=None, val_split=0.1, use_rgb_size=False):
    """
    获取增强版的CIFAR10-DVS数据加载器，同时返回RGB和DVS数据
    使用TLCIFAR10的索引机制处理RGB和DVS数据量不匹配问题
    
    Args:
        batch_size: 批次大小
        T: 时间步数（这里主要用于兼容性，实际从数据中读取）
        split_ratio: 训练/测试分割比例（已废弃，保留以兼容旧代码）
        train_set_ratio: 实际使用的训练集比例（DVS数据）
        augmentation: 是否启用数据增强
        num_workers: 数据加载器工作进程数
        rgb_root: RGB CIFAR10数据集路径
        dvs_root: DVS CIFAR10数据集根路径
        val_split: 验证集比例（从训练集中划分，默认10%）
        use_rgb_size: 是否以RGB数据量为准（True: RGB为准，DVS循环；False: DVS为准，RGB循环）
    
    Returns:
        train_loader, val_loader, test_loader
        - train_loader: 训练集 (返回(rgb_img, dvs_data), labels)
        - val_loader: 验证集 (从训练集划分，返回(rgb_img, dvs_data), labels)
        - test_loader: 测试集 (独立的test文件夹，返回dvs_data, labels)
    """
    from dataloader.dataloader_utils import DataLoaderX
    
    # DVS数据路径
    if dvs_root is None:
        dvs_root = '/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat'
    
    # RGB CIFAR10 路径（如果为None则不加载RGB数据）
    # 注意：不要给 rgb_root=None 设置默认值，这样才能支持 DVS-only 模式
    
    train_path = os.path.join(dvs_root, 'train')
    test_path = os.path.join(dvs_root, 'test')
    
    # 创建训练集（从train文件夹的前90%）
    train_dataset = EnhancedDVSCifar10(
        root=train_path, 
        train=True, 
        augmentation=augmentation,
        rgb_root=rgb_root,
        dvs_train_set_ratio=train_set_ratio,
        split='train',
        val_split=val_split,
        use_rgb_size=use_rgb_size  # 传递参数
    )
    
    # 创建验证集（从train文件夹的后10%）
    val_dataset = EnhancedDVSCifar10(
        root=train_path, 
        train=True, 
        augmentation=False,  # 验证时不使用增强
        rgb_root=rgb_root,
        dvs_train_set_ratio=train_set_ratio,
        split='val',
        val_split=val_split,
        use_rgb_size=False  # 验证集始终以DVS为准
    )
    
    # 创建测试集（独立的test文件夹）
    test_dataset = EnhancedDVSCifar10(
        root=test_path, 
        train=False, 
        augmentation=False,  # 测试时不使用增强
        rgb_root=rgb_root,
        dvs_train_set_ratio=1.0,
        split='test'
    )
    
    # 检查RGB数据是否被加载
    train_has_rgb = train_dataset.rgb_data is not None
    val_has_rgb = val_dataset.rgb_data is not None
    
    if train_has_rgb:
        print(f"✓ Train: {len(train_dataset.rgb_data)} RGB samples paired with {len(train_dataset.dvs_samples)} DVS samples (RGB-based, DVS循环)")
        print(f"   策略: 以RGB为准充分利用数据 - 每个epoch遍历全部{len(train_dataset.rgb_data)}个RGB样本")
    else:
        print(f"✓ Train: {len(train_dataset)} samples (DVS-only, with augmentation)")
    
    if val_has_rgb:
        print(f"✓ Val: {len(val_dataset)} DVS samples paired with {len(val_dataset.rgb_data)} RGB samples (DVS-based)")
        print(f"   策略: 以DVS为准真实评估泛化能力")
    else:
          print(f"✓ Val: {len(val_dataset)} samples (DVS-only, no augmentation)")

    print(f"✓ Test: {len(test_dataset)} samples (DVS-only, independent)")
    
    # 训练集加载器
    train_loader = DataLoaderX(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    # 验证集加载器
    val_loader = DataLoaderX(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    
    # 测试集加载器
    test_loader = DataLoaderX(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# 兼容性函数
def get_cifar10_DVS_enhanced(batch_size, T=10, split_ratio=0.9, train_set_ratio=1.0, 
                            encode_type='TET', augmentation=True, rgb_root=None, dvs_root=None, val_split=0.1, use_rgb_size=False):
    """
    兼容性函数，保持与原get_cifar10_DVS相同的接口
    现在返回三个loader：train, val, test
    - train/val: 返回 (rgb_img, dvs_data), labels
    - test: 返回 dvs_data, labels
    使用TLCIFAR10的机制处理RGB和DVS数据量不匹配
    """
    return get_enhanced_cifar10_DVS(
        batch_size=batch_size,
        T=T,
        split_ratio=split_ratio,
        train_set_ratio=train_set_ratio,
        augmentation=augmentation,
        rgb_root=rgb_root,
        dvs_root=dvs_root,
        val_split=val_split,
        use_rgb_size=use_rgb_size
    )
