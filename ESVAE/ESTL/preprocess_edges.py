#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成CIFAR10边缘数据集（3通道多阈值Canny）

处理CIFAR10 RGB图像，生成3通道边缘图（弱/中/强边缘），保存为npz格式
确保与原始RGB样本一一对应
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import cv2

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class MultiThresholdEdgeExtractor:
    """
    3通道多阈值Canny边缘提取器
    
    输出3通道：
    - Channel 0: 弱边缘（低阈值，捕获更多细节）
    - Channel 1: 中等边缘（中阈值，平衡）
    - Channel 2: 强边缘（高阈值，主要结构）
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 三组不同的Canny阈值
        self.thresholds = [
            (10, 50),    # 弱边缘：低阈值，捕获细节
            (50, 150),   # 中等边缘：平衡
            (100, 200),  # 强边缘：只保留显著边缘
        ]
    
    def extract_canny_single(self, img_np, low_thresh, high_thresh):
        """
        提取单个阈值的Canny边缘
        Args:
            img_np: (H, W, 3) numpy array, RGB, uint8
            low_thresh: 低阈值
            high_thresh: 高阈值
        Returns:
            edge: (H, W) 边缘二值图, float32 [0,1]
        """
        # 转灰度
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Canny边缘检测
        edges = cv2.Canny(gray, threshold1=low_thresh, threshold2=high_thresh)
        
        # 归一化到[0,1]
        edges = edges.astype(np.float32) / 255.0
        
        return edges
    
    def extract_edges(self, img_tensor, img_np):
        """
        提取3通道多阈值Canny边缘
        Args:
            img_tensor: (3, H, W) tensor (未使用，保持接口一致)
            img_np: (H, W, 3) numpy array, RGB, uint8
        Returns:
            edges: (3, H, W) numpy array, [弱边缘, 中等边缘, 强边缘]
        """
        # 提取三个不同强度的边缘
        edge_weak = self.extract_canny_single(img_np, *self.thresholds[0])      # (H, W)
        edge_medium = self.extract_canny_single(img_np, *self.thresholds[1])    # (H, W)
        edge_strong = self.extract_canny_single(img_np, *self.thresholds[2])    # (H, W)
        
        # 堆叠为3通道
        edges = np.stack([edge_weak, edge_medium, edge_strong], axis=0)  # (3, H, W)
        
        return edges


def process_cifar10_edges(
    rgb_root='/home/user/kpm/kpm//Dataset/CIFAR10/cifar10',
    edge_root='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10-edge',
    device='cuda'
):
    """
    处理CIFAR10生成边缘数据集
    
    Args:
        rgb_root: RGB CIFAR10路径
        edge_root: 边缘数据保存路径
        device: 计算设备
    """
    print("="*70)
    print("CIFAR10 3通道边缘数据集生成".center(70))
    print("="*70)
    print("边缘通道设计:")
    print("  Channel 0: 弱边缘（阈值10-50）- 捕获细节")
    print("  Channel 1: 中等边缘（阈值50-150）- 平衡")
    print("  Channel 2: 强边缘（阈值100-200）- 主要结构")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(os.path.join(edge_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(edge_root, 'test'), exist_ok=True)
    
    # 初始化边缘提取器
    extractor = MultiThresholdEdgeExtractor(device=device)
    
    # 处理训练集和测试集
    for split in ['train', 'test']:
        print(f"\n处理 {split} 集...")
        
        is_train = (split == 'train')
        
        # 加载CIFAR10数据集
        dataset = CIFAR10(
            root=rgb_root,
            train=is_train,
            download=False,
            transform=transforms.ToTensor()
        )
        
        output_dir = os.path.join(edge_root, split)
        
        # 处理每张图像
        for idx in tqdm(range(len(dataset)), desc=f"生成{split}边缘"):
            img_tensor, label = dataset[idx]  # img_tensor: (3, 32, 32), [0,1]
            
            # 转换为numpy用于Canny
            img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # 提取3通道边缘 (3, 32, 32)
            edges = extractor.extract_edges(img_tensor, img_np)
            
            # 保存为npz格式
            save_path = os.path.join(output_dir, f'{idx:05d}.npz')
            np.savez_compressed(
                save_path,
                edges=edges,  # (3, 32, 32) float32 [弱/中/强边缘]
                label=label,  # int
                index=idx     # 索引，确保对应
            )
        
        print(f"✓ {split} 集完成: {len(dataset)} 样本")
        print(f"  保存路径: {output_dir}")
    
    print("\n" + "="*70)
    print("边缘数据集生成完成！".center(70))
    print("="*70)
    
    # 验证数据
    print("\n验证生成的数据...")
    sample_path = os.path.join(edge_root, 'train', '00000.npz')
    data = np.load(sample_path)
    print(f"✓ 样本形状: edges={data['edges'].shape}, label={data['label']}")
    print(f"✓ Channel 0 (弱边缘)范围: [{data['edges'][0].min():.3f}, {data['edges'][0].max():.3f}]")
    print(f"✓ Channel 1 (中等边缘)范围: [{data['edges'][1].min():.3f}, {data['edges'][1].max():.3f}]")
    print(f"✓ Channel 2 (强边缘)范围: [{data['edges'][2].min():.3f}, {data['edges'][2].max():.3f}]")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成CIFAR10边缘数据集')
    parser.add_argument('--rgb_root', type=str, 
                       default='/home/user/kpm/kpm//Dataset/CIFAR10/cifar10',
                       help='RGB CIFAR10路径')
    parser.add_argument('--edge_root', type=str,
                       default='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10-edge',
                       help='边缘数据保存路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    process_cifar10_edges(
        rgb_root=args.rgb_root,
        edge_root=args.edge_root,
        device=args.device
    )

