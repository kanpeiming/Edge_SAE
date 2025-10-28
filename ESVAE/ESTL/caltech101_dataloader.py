# -*- coding: utf-8 -*-
"""
Caltech 101 和 N-Caltech101 数据加载器
支持RGB和DVS一对一配对的数据加载

Caltech 101: RGB图像数据集
N-Caltech101: DVS事件数据集（由Caltech 101转换而来，一对一对应）

数据集结构:
RGB:
    caltech101/
        101_ObjectCategories/
            accordion/
                image_0001.jpg
                image_0002.jpg
                ...
            airplanes/
                ...
            ...

DVS:
    N-Caltech101/
        accordion/
            image_0001.bin (或 .npy)
            image_0002.bin
            ...
        airplanes/
            ...
        ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random


class DVSAugmentation:
    """
    DVS-specific data augmentation techniques
    """
    
    def __init__(self, 
                 enable_flip=True,
                 enable_rotation=True, 
                 enable_translation=True,
                 enable_noise=True,
                 flip_prob=0.5,
                 rotation_range=15,
                 translation_range=5,
                 noise_std=0.1):
        
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_translation = enable_translation
        self.enable_noise = enable_noise
        
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.noise_std = noise_std
    
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
        
        return augmented
    
    def _rotate_tensor(self, tensor, angle):
        """Apply rotation to tensor"""
        import torch.nn.functional as F
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


class Caltech101Dataset(Dataset):
    """
    Caltech 101 + N-Caltech101 配对数据集
    支持RGB和DVS一对一配对加载
    """
    
    def __init__(self, 
                 rgb_root, 
                 dvs_root, 
                 split='train',
                 train_ratio=0.8,
                 val_ratio=0.1,
                 augmentation=False,
                 img_size=128,
                 T=10,
                 dvs_format='npy'):
        """
        Args:
            rgb_root: RGB Caltech101数据集根目录 (包含101_ObjectCategories文件夹)
            dvs_root: N-Caltech101数据集根目录 (包含类别文件夹)
            split: 'train', 'val', 或 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            augmentation: 是否使用数据增强
            img_size: 图像大小
            T: DVS时间步数
            dvs_format: DVS数据格式 ('npy', 'bin', 'pt')
        """
        self.rgb_root = rgb_root
        self.dvs_root = dvs_root
        self.split = split
        self.augmentation = augmentation
        self.img_size = img_size
        self.T = T
        self.dvs_format = dvs_format
        
        # RGB变换
        if augmentation and split == 'train':
            self.rgb_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # DVS增强
        if augmentation and split == 'train':
            self.dvs_augmentor = DVSAugmentation(
                enable_flip=True,
                enable_rotation=True,
                enable_translation=True,
                enable_noise=True,
                flip_prob=0.5,
                rotation_range=10,
                translation_range=3,
                noise_std=0.05
            )
        else:
            self.dvs_augmentor = None
        
        # 构建数据索引
        self.samples = []
        self.class_to_idx = {}
        self._build_dataset_index(train_ratio, val_ratio)
        
        print(f"✓ Caltech101 {split} dataset loaded: {len(self.samples)} samples, {len(self.class_to_idx)} classes")
    
    def _build_dataset_index(self, train_ratio, val_ratio):
        """构建数据集索引，确保RGB和DVS一对一配对"""
        
        # 获取RGB类别目录
        rgb_base = os.path.join(self.rgb_root, '101_ObjectCategories')
        if not os.path.exists(rgb_base):
            rgb_base = self.rgb_root  # 如果直接是类别目录
        
        class_dirs = sorted([d for d in os.listdir(rgb_base) 
                           if os.path.isdir(os.path.join(rgb_base, d))])
        
        # 移除背景类别（如果存在）
        if 'BACKGROUND_Google' in class_dirs:
            class_dirs.remove('BACKGROUND_Google')
        
        # 移除Faces_easy类别（重复类别）
        if 'Faces_easy' in class_dirs:
            class_dirs.remove('Faces_easy')
            print(f"  ⚠ 移除 'Faces_easy' 类别（与Faces重复）")
        
        # 为每个类别分配索引
        for idx, class_name in enumerate(class_dirs):
            self.class_to_idx[class_name] = idx
        
        # 为每个类别构建样本列表
        all_class_samples = {}
        
        for class_name in class_dirs:
            rgb_class_dir = os.path.join(rgb_base, class_name)
            dvs_class_dir = os.path.join(self.dvs_root, class_name)
            
            if not os.path.exists(dvs_class_dir):
                print(f"⚠ Warning: DVS class directory not found: {dvs_class_dir}")
                continue
            
            # 获取RGB图像列表
            rgb_files = sorted([f for f in os.listdir(rgb_class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # 获取DVS文件列表
            dvs_ext = f'.{self.dvs_format}'
            dvs_files = sorted([f for f in os.listdir(dvs_class_dir) 
                              if f.endswith(dvs_ext)])
            
            # 确保RGB和DVS文件名对应
            class_samples = []
            for rgb_file in rgb_files:
                # 构造对应的DVS文件名
                base_name = os.path.splitext(rgb_file)[0]
                dvs_file = base_name + dvs_ext
                
                rgb_path = os.path.join(rgb_class_dir, rgb_file)
                dvs_path = os.path.join(dvs_class_dir, dvs_file)
                
                # 检查DVS文件是否存在
                if os.path.exists(dvs_path):
                    class_samples.append({
                        'rgb_path': rgb_path,
                        'dvs_path': dvs_path,
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
            
            if len(class_samples) > 0:
                all_class_samples[class_name] = class_samples
        
        # 按split划分数据
        for class_name, samples in all_class_samples.items():
            n_samples = len(samples)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # 随机打乱（使用固定种子以保证可重复性）
            random.Random(42).shuffle(samples)
            
            if self.split == 'train':
                self.samples.extend(samples[:n_train])
            elif self.split == 'val':
                self.samples.extend(samples[n_train:n_train+n_val])
            elif self.split == 'test':
                self.samples.extend(samples[n_train+n_val:])
    
    def __len__(self):
        return len(self.samples)
    
    def _load_dvs_data(self, dvs_path):
        """加载DVS数据（配对数据集版本 - 修复版）"""
        if self.dvs_format == 'npy':
            dvs_data = np.load(dvs_path)
            dvs_tensor = torch.from_numpy(dvs_data).float()
        elif self.dvs_format == 'pt':
            dvs_tensor = torch.load(dvs_path, weights_only=True)
        elif self.dvs_format == 'bin':
            # ✅ 官方格式：每个事件40位（5字节）
            # bit 39-32: X坐标 (8位)
            # bit 31-24: Y坐标 (8位)
            # bit 23: 极性 (1位, 0=OFF, 1=ON)
            # bit 22-0: 时间戳 (23位, 微秒)
            
            with open(dvs_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8)
            
            # 每个事件5字节
            num_events = len(raw_data) // 5
            
            if num_events == 0:
                # 空文件，创建零数据
                dvs_data = np.zeros((self.T, 2, 180, 240), dtype=np.float32)
            else:
                # 创建累积帧 (histogram)
                # N-Caltech101实际分辨率（从官方文档）
                # X: 0-255, Y: 0-255 (但实际数据可能更小)
                H, W = 180, 240  # ATIS相机标准分辨率
                C = 2  # ON/OFF events
                frames = np.zeros((self.T, C, H, W), dtype=np.float32)
                
                # 解析事件
                events = []
                for i in range(num_events):
                    offset = i * 5
                    if offset + 5 > len(raw_data):
                        break
                    
                    # 读取5字节 = 40位
                    byte0 = int(raw_data[offset])
                    byte1 = int(raw_data[offset+1])
                    byte2 = int(raw_data[offset+2])
                    byte3 = int(raw_data[offset+3])
                    byte4 = int(raw_data[offset+4])
                    
                    # bit 39-32: X坐标
                    x = byte0
                    
                    # bit 31-24: Y坐标
                    y = byte1
                    
                    # bit 23-0: 极性(1位) + 时间戳(23位)
                    # 组合后3字节
                    pol_ts = (byte2 << 16) | (byte3 << 8) | byte4
                    
                    # bit 23: 极性
                    pol = (pol_ts >> 23) & 0x1
                    
                    # bit 22-0: 时间戳
                    ts = pol_ts & 0x7FFFFF
                    
                    # 限制坐标范围
                    if 0 <= x < W and 0 <= y < H:
                        events.append((x, y, ts, pol))
                
                if len(events) > 0:
                    events = np.array(events)
                    timestamps = events[:, 2]
                    
                    # 归一化时间戳到 [0, T-1]
                    if timestamps.max() > timestamps.min():
                        time_bins = ((timestamps - timestamps.min()) / 
                                    (timestamps.max() - timestamps.min() + 1e-6) * (self.T - 1)).astype(int)
                    else:
                        time_bins = np.zeros(len(events), dtype=int)
                    
                    # 累积事件到帧
                    for idx, (x, y, ts, pol) in enumerate(events):
                        t_bin = min(time_bins[idx], self.T - 1)
                        frames[t_bin, pol, int(y), int(x)] += 1.0
                    
                    # ✅ 修复：改进归一化策略
                    # 使用更鲁棒的归一化方法
                    frames_nonzero = frames[frames > 0]
                    if len(frames_nonzero) > 0:
                        # 方法1: 使用百分位数归一化，避免极端值影响
                        p99 = np.percentile(frames_nonzero, 99)
                        if p99 > 0:
                            frames = np.clip(frames / p99, 0, 1)
                        else:
                            # 如果99分位数为0，使用最大值
                            max_val = frames.max()
                            if max_val > 0:
                                frames = frames / max_val
                
                dvs_data = frames
            
            dvs_tensor = torch.from_numpy(dvs_data).float()
        else:
            raise ValueError(f"Unsupported DVS format: {self.dvs_format}")
        
        # 确保形状为 (T, C, H, W)
        if dvs_tensor.dim() == 3:
            dvs_tensor = dvs_tensor.unsqueeze(0)
        elif dvs_tensor.dim() == 2:
            dvs_tensor = dvs_tensor.unsqueeze(0).unsqueeze(0)
        elif dvs_tensor.dim() == 1:
            raise ValueError(f"DVS数据维度错误: {dvs_tensor.shape}")
        
        # 调整时间步数
        current_T = dvs_tensor.size(0)
        if current_T != self.T:
            if current_T > self.T:
                indices = torch.linspace(0, current_T-1, self.T).long()
                dvs_tensor = dvs_tensor[indices]
            else:
                padding = self.T - current_T
                last_frame = dvs_tensor[-1:].repeat(padding, 1, 1, 1)
                dvs_tensor = torch.cat([dvs_tensor, last_frame], dim=0)
        
        # ✅ 修复：调整空间大小时正确处理维度
        current_H = dvs_tensor.size(-2)
        current_W = dvs_tensor.size(-1)
        if current_H != self.img_size or current_W != self.img_size:
            # 需要将 (T, C, H, W) 逐时间步插值
            T, C, H, W = dvs_tensor.shape
            resized_frames = []
            for t in range(T):
                frame = dvs_tensor[t:t+1]  # (1, C, H, W)
                frame_resized = torch.nn.functional.interpolate(
                    frame, 
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                )
                resized_frames.append(frame_resized)
            dvs_tensor = torch.cat(resized_frames, dim=0)  # (T, C, img_size, img_size)
        
        return dvs_tensor
    
    def __getitem__(self, idx):
        """
        Returns:
            (rgb_img, dvs_data): RGB图像和DVS数据的元组
            label: 类别标签
        """
        sample = self.samples[idx]
        
        # 加载RGB图像
        rgb_img = Image.open(sample['rgb_path']).convert('RGB')
        rgb_img = self.rgb_transform(rgb_img)
        
        # 加载DVS数据
        dvs_data = self._load_dvs_data(sample['dvs_path'])
        
        # 应用DVS增强
        if self.dvs_augmentor is not None:
            dvs_data = self.dvs_augmentor(dvs_data)
        
        label = sample['label']
        
        return (rgb_img, dvs_data), torch.tensor(label)


def get_caltech101_dataloaders(
    rgb_root,
    dvs_root,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    augmentation=True,
    img_size=128,
    T=10,
    num_workers=4,
    dvs_format='npy'
):
    """
    获取Caltech101数据加载器
    
    Args:
        rgb_root: RGB Caltech101数据集根目录
        dvs_root: N-Caltech101数据集根目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        augmentation: 是否使用数据增强
        img_size: 图像大小
        T: DVS时间步数
        num_workers: 数据加载线程数
        dvs_format: DVS数据格式
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from dataloader.dataloader_utils import DataLoaderX
    
    # 创建数据集
    train_dataset = Caltech101Dataset(
        rgb_root=rgb_root,
        dvs_root=dvs_root,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augmentation=augmentation,
        img_size=img_size,
        T=T,
        dvs_format=dvs_format
    )
    
    val_dataset = Caltech101Dataset(
        rgb_root=rgb_root,
        dvs_root=dvs_root,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augmentation=False,
        img_size=img_size,
        T=T,
        dvs_format=dvs_format
    )
    
    test_dataset = Caltech101Dataset(
        rgb_root=rgb_root,
        dvs_root=dvs_root,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augmentation=False,
        img_size=img_size,
        T=T,
        dvs_format=dvs_format
    )
    
    # 创建数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoaderX(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  类别数: {len(train_dataset.class_to_idx)}")
    print(f"  图像大小: {img_size}x{img_size}")
    print(f"  时间步数: {T}")
    
    return train_loader, val_loader, test_loader


# DVS-only版本（用于baseline）
class Caltech101DVSOnlyDataset(Dataset):
    """
    N-Caltech101 DVS-only数据集（用于baseline实验）
    """
    
    def __init__(self, 
                 dvs_root, 
                 split='train',
                 train_ratio=0.8,
                 val_ratio=0.1,
                 augmentation=False,
                 img_size=128,
                 T=10,
                 dvs_format='npy'):
        """
        Args:
            dvs_root: N-Caltech101数据集根目录
            split: 'train', 'val', 或 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            augmentation: 是否使用数据增强
            img_size: 图像大小
            T: DVS时间步数
            dvs_format: DVS数据格式
        """
        self.dvs_root = dvs_root
        self.split = split
        self.augmentation = augmentation
        self.img_size = img_size
        self.T = T
        self.dvs_format = dvs_format
        
        # DVS增强
        if augmentation and split == 'train':
            self.dvs_augmentor = DVSAugmentation(
                enable_flip=True,
                enable_rotation=True,
                enable_translation=True,
                enable_noise=True,
                flip_prob=0.5,
                rotation_range=10,
                translation_range=3,
                noise_std=0.05
            )
        else:
            self.dvs_augmentor = None
        
        # 构建数据索引
        self.samples = []
        self.class_to_idx = {}
        self._build_dataset_index(train_ratio, val_ratio)
        
        print(f"✓ N-Caltech101 (DVS-only) {split} dataset loaded: {len(self.samples)} samples")
    
    def _build_dataset_index(self, train_ratio, val_ratio):
        """构建DVS数据集索引"""
        class_dirs = sorted([d for d in os.listdir(self.dvs_root) 
                           if os.path.isdir(os.path.join(self.dvs_root, d))])
        
        # 移除Faces_easy类别（重复类别）
        if 'Faces_easy' in class_dirs:
            class_dirs.remove('Faces_easy')
            print(f"  ⚠ 移除 'Faces_easy' 类别（与Faces重复）")
        
        # 为每个类别分配索引
        for idx, class_name in enumerate(class_dirs):
            self.class_to_idx[class_name] = idx
        
        # 为每个类别构建样本列表
        all_class_samples = {}
        
        for class_name in class_dirs:
            dvs_class_dir = os.path.join(self.dvs_root, class_name)
            
            # 获取DVS文件列表
            dvs_ext = f'.{self.dvs_format}'
            dvs_files = sorted([f for f in os.listdir(dvs_class_dir) 
                              if f.endswith(dvs_ext)])
            
            class_samples = []
            for dvs_file in dvs_files:
                dvs_path = os.path.join(dvs_class_dir, dvs_file)
                class_samples.append({
                    'dvs_path': dvs_path,
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
            
            if len(class_samples) > 0:
                all_class_samples[class_name] = class_samples
        
        # 按split划分数据
        for class_name, samples in all_class_samples.items():
            n_samples = len(samples)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # 随机打乱
            random.Random(42).shuffle(samples)
            
            if self.split == 'train':
                self.samples.extend(samples[:n_train])
            elif self.split == 'val':
                self.samples.extend(samples[n_train:n_train+n_val])
            elif self.split == 'test':
                self.samples.extend(samples[n_train+n_val:])
    
    def __len__(self):
        return len(self.samples)
    
    def _load_dvs_data(self, dvs_path):
        """加载DVS数据（修复版）"""
        if self.dvs_format == 'npy':
            dvs_data = np.load(dvs_path)
            dvs_tensor = torch.from_numpy(dvs_data).float()
        elif self.dvs_format == 'pt':
            dvs_tensor = torch.load(dvs_path, weights_only=True)
        elif self.dvs_format == 'bin':
            # ✅ 官方格式：每个事件40位（5字节）
            # bit 39-32: X坐标 (8位)
            # bit 31-24: Y坐标 (8位)
            # bit 23: 极性 (1位, 0=OFF, 1=ON)
            # bit 22-0: 时间戳 (23位, 微秒)
            
            with open(dvs_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8)
            
            # 每个事件5字节
            num_events = len(raw_data) // 5
            
            if num_events == 0:
                # 空文件，创建零数据
                dvs_data = np.zeros((self.T, 2, 180, 240), dtype=np.float32)
            else:
                # 创建累积帧 (histogram)
                # N-Caltech101实际分辨率
                H, W = 180, 240  # ATIS相机标准分辨率
                C = 2  # ON/OFF events
                
                # 初始化累积图像
                frames = np.zeros((self.T, C, H, W), dtype=np.float32)
                
                # 解析事件
                events = []
                for i in range(num_events):
                    offset = i * 5
                    if offset + 5 > len(raw_data):
                        break
                    
                    # 读取5字节 = 40位
                    byte0 = int(raw_data[offset])
                    byte1 = int(raw_data[offset+1])
                    byte2 = int(raw_data[offset+2])
                    byte3 = int(raw_data[offset+3])
                    byte4 = int(raw_data[offset+4])
                    
                    # bit 39-32: X坐标
                    x = byte0
                    
                    # bit 31-24: Y坐标
                    y = byte1
                    
                    # bit 23-0: 极性(1位) + 时间戳(23位)
                    pol_ts = (byte2 << 16) | (byte3 << 8) | byte4
                    
                    # bit 23: 极性
                    pol = (pol_ts >> 23) & 0x1
                    
                    # bit 22-0: 时间戳
                    ts = pol_ts & 0x7FFFFF
                    
                    # 限制坐标范围
                    if 0 <= x < W and 0 <= y < H:
                        events.append((x, y, ts, pol))
                
                if len(events) > 0:
                    # 将事件分配到时间bins
                    events = np.array(events)
                    timestamps = events[:, 2]
                    
                    # 归一化时间戳到 [0, T-1]
                    if timestamps.max() > timestamps.min():
                        time_bins = ((timestamps - timestamps.min()) / 
                                    (timestamps.max() - timestamps.min() + 1e-6) * (self.T - 1)).astype(int)
                    else:
                        time_bins = np.zeros(len(events), dtype=int)
                    
                    # 累积事件到帧
                    for idx, (x, y, ts, pol) in enumerate(events):
                        t_bin = min(time_bins[idx], self.T - 1)
                        frames[t_bin, pol, int(y), int(x)] += 1.0
                    
                    # ✅ 修复：改进归一化策略
                    # 使用更鲁棒的归一化方法
                    frames_nonzero = frames[frames > 0]
                    if len(frames_nonzero) > 0:
                        # 方法1: 使用百分位数归一化，避免极端值影响
                        p99 = np.percentile(frames_nonzero, 99)
                        if p99 > 0:
                            frames = np.clip(frames / p99, 0, 1)
                        else:
                            # 如果99分位数为0，使用最大值
                            max_val = frames.max()
                            if max_val > 0:
                                frames = frames / max_val
                
                dvs_data = frames
            
            dvs_tensor = torch.from_numpy(dvs_data).float()
        else:
            raise ValueError(f"Unsupported DVS format: {self.dvs_format}")
        
        # 确保形状为 (T, C, H, W)
        if dvs_tensor.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            dvs_tensor = dvs_tensor.unsqueeze(0)
        elif dvs_tensor.dim() == 2:
            # (H, W) -> (1, 1, H, W)
            dvs_tensor = dvs_tensor.unsqueeze(0).unsqueeze(0)
        elif dvs_tensor.dim() == 1:
            raise ValueError(f"DVS数据维度错误: {dvs_tensor.shape}，无法处理1维数据")
        
        # 调整时间步数
        current_T = dvs_tensor.size(0)
        if current_T != self.T:
            if current_T > self.T:
                # 下采样时间步
                indices = torch.linspace(0, current_T-1, self.T).long()
                dvs_tensor = dvs_tensor[indices]
            else:
                # 上采样时间步（重复最后一帧）
                padding = self.T - current_T
                last_frame = dvs_tensor[-1:].repeat(padding, 1, 1, 1)
                dvs_tensor = torch.cat([dvs_tensor, last_frame], dim=0)
        
        # ✅ 修复：调整空间大小时正确处理维度
        current_H = dvs_tensor.size(-2)
        current_W = dvs_tensor.size(-1)
        if current_H != self.img_size or current_W != self.img_size:
            # 需要将 (T, C, H, W) 逐时间步插值
            # 不能直接reshape，会破坏空间结构
            T, C, H, W = dvs_tensor.shape
            resized_frames = []
            for t in range(T):
                frame = dvs_tensor[t:t+1]  # (1, C, H, W)
                frame_resized = torch.nn.functional.interpolate(
                    frame, 
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                )
                resized_frames.append(frame_resized)
            dvs_tensor = torch.cat(resized_frames, dim=0)  # (T, C, img_size, img_size)
        
        return dvs_tensor
    
    def __getitem__(self, idx):
        """
        Returns:
            dvs_data: DVS数据
            label: 类别标签
        """
        sample = self.samples[idx]
        
        # 加载DVS数据
        dvs_data = self._load_dvs_data(sample['dvs_path'])
        
        # 应用DVS增强
        if self.dvs_augmentor is not None:
            dvs_data = self.dvs_augmentor(dvs_data)
        
        label = sample['label']
        
        return dvs_data, torch.tensor(label)


def get_caltech101_dvs_only_dataloaders(
    dvs_root,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    augmentation=True,
    img_size=128,
    T=10,
    num_workers=4,
    dvs_format='npy'
):
    """
    获取N-Caltech101 DVS-only数据加载器（用于baseline）
    
    Args:
        dvs_root: N-Caltech101数据集根目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        augmentation: 是否使用数据增强
        img_size: 图像大小
        T: DVS时间步数
        num_workers: 数据加载线程数
        dvs_format: DVS数据格式
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from dataloader.dataloader_utils import DataLoaderX
    
    # 创建数据集
    train_dataset = Caltech101DVSOnlyDataset(
        dvs_root=dvs_root,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augmentation=augmentation,
        img_size=img_size,
        T=T,
        dvs_format=dvs_format
    )
    
    val_dataset = Caltech101DVSOnlyDataset(
        dvs_root=dvs_root,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augmentation=False,
        img_size=img_size,
        T=T,
        dvs_format=dvs_format
    )
    
    test_dataset = Caltech101DVSOnlyDataset(
        dvs_root=dvs_root,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augmentation=False,
        img_size=img_size,
        T=T,
        dvs_format=dvs_format
    )
    
    # 创建数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoaderX(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\n数据集统计 (DVS-only):")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  类别数: {len(train_dataset.class_to_idx)}")
    
    return train_loader, val_loader, test_loader

