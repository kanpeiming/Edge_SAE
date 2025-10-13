#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
边缘预训练数据加载器（3通道多阈值Canny）

加载RGB图像和对应的3通道边缘标签（弱/中/强边缘，npz格式）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class EdgeDataset(Dataset):
    """
    边缘检测数据集
    返回RGB图像和对应的边缘标签
    """
    
    def __init__(self, rgb_root, edge_root, train=True, transform=None):
        """
        Args:
            rgb_root: RGB CIFAR10路径
            edge_root: 边缘数据路径（npz文件）
            train: 训练集或测试集
            transform: RGB图像变换
        """
        self.rgb_root = rgb_root
        self.edge_root = edge_root
        self.train = train
        self.transform = transform
        
        # 加载RGB数据集
        self.rgb_dataset = CIFAR10(
            root=rgb_root,
            train=train,
            download=False
        )
        
        # 构建边缘数据路径
        split = 'train' if train else 'test'
        self.edge_dir = os.path.join(edge_root, split)
        
        # 验证数据存在
        if not os.path.exists(self.edge_dir):
            raise FileNotFoundError(f"边缘数据目录不存在: {self.edge_dir}")
        
        print(f"✓ 加载边缘数据集: {split}")
        print(f"  RGB: {len(self.rgb_dataset)} 样本")
        print(f"  Edge: {self.edge_dir}")
    
    def __len__(self):
        return len(self.rgb_dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb_img: (3, H, W) tensor, normalized
            edge_label: (3, H, W) tensor, [弱边缘, 中等边缘, 强边缘], float32 [0,1]
            label: 类别标签（用于分析，训练时不用）
        """
        # 加载RGB图像
        rgb_img, label = self.rgb_dataset[idx]  # PIL Image
        
        # 应用变换
        if self.transform:
            rgb_img = self.transform(rgb_img)
        else:
            rgb_img = transforms.ToTensor()(rgb_img)
        
        # 加载3通道边缘标签
        edge_path = os.path.join(self.edge_dir, f'{idx:05d}.npz')
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"边缘文件不存在: {edge_path}\n请先运行 preprocess_edges.py 生成边缘数据")
        
        edge_data = np.load(edge_path)
        edge_label = torch.from_numpy(edge_data['edges']).float()  # (3, H, W)
        
        # 验证形状
        if edge_label.shape[0] != 3:
            raise ValueError(
                f"边缘数据形状错误: {edge_label.shape}，期望 (3, H, W)\n"
                f"当前边缘数据可能是旧版本（2通道），请重新运行 preprocess_edges.py 生成3通道边缘数据"
            )
        
        # 验证索引一致性（确保标签不被破坏）
        assert edge_data['index'] == idx, f"索引不匹配: {idx} vs {edge_data['index']}"
        assert edge_data['label'] == label, f"标签不匹配: {label} vs {edge_data['label']}"
        
        return rgb_img, edge_label, label


def get_edge_dataloaders(
    rgb_root='/home/user/kpm/kpm//Dataset/CIFAR10/cifar10',
    edge_root='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10-edge',
    batch_size=32,
    num_workers=8,
    augmentation=False
):
    """
    获取边缘预训练数据加载器
    
    Args:
        rgb_root: RGB CIFAR10路径
        edge_root: 边缘数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        augmentation: 是否使用数据增强
    
    Returns:
        train_loader, test_loader
    """
    # 数据变换
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = EdgeDataset(
        rgb_root=rgb_root,
        edge_root=edge_root,
        train=True,
        transform=train_transform
    )
    
    test_dataset = EdgeDataset(
        rgb_root=rgb_root,
        edge_root=edge_root,
        train=False,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ 数据加载器创建完成")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  测试批次: {len(test_loader)}")
    
    return train_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载器
    print("测试边缘数据加载器...")
    
    train_loader, test_loader = get_edge_dataloaders(batch_size=4)
    
    # 测试一个batch
    for rgb, edge, label in train_loader:
        print(f"\n✓ Batch shapes:")
        print(f"  RGB: {rgb.shape}")      # (4, 3, 32, 32)
        print(f"  Edge: {edge.shape}")    # (4, 3, 32, 32) - 3通道！
        print(f"  Label: {label.shape}")  # (4,)
        print(f"  Edge channels:")
        print(f"    Channel 0 (弱边缘) range: [{edge[:, 0].min():.3f}, {edge[:, 0].max():.3f}]")
        print(f"    Channel 1 (中等边缘) range: [{edge[:, 1].min():.3f}, {edge[:, 1].max():.3f}]")
        print(f"    Channel 2 (强边缘) range: [{edge[:, 2].min():.3f}, {edge[:, 2].max():.3f}]")
        break
    
    print("\n✓ 数据加载器测试通过!")

