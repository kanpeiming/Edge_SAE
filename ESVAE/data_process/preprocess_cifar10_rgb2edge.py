# -*- coding: utf-8 -*-
"""
CIFAR10 RGB to Edge 预处理脚本
将CIFAR10 RGB图像转换为Sobel边缘图并保存

功能：
- 使用Sobel算子提取边缘（2通道：水平+垂直）
- 保存为.pt文件，格式与DVS数据一致
- 按类别组织输出目录

使用方法：
python preprocess_cifar10_rgb2edge.py --output_dir /path/to/output --img_size 48
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.nn.functional as F

def sobel_edge_extraction(img_tensor):
    """
    使用Sobel算子提取边缘（简化版，不需要GPU）
    
    Args:
        img_tensor: (3, H, W) RGB图像
    
    Returns:
        edge_tensor: (2, H, W) 边缘图（通道0=水平边缘，通道1=垂直边缘）
    """
    # Sobel核
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 转为灰度图
    if img_tensor.shape[0] == 3:
        # RGB to grayscale
        gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
    else:
        gray = img_tensor[0]
    
    # 添加batch和channel维度
    gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # 计算梯度
    edge_x = F.conv2d(gray, sobel_x, padding=1)  # (1, 1, H, W)
    edge_y = F.conv2d(gray, sobel_y, padding=1)  # (1, 1, H, W)
    
    # 拼接为2通道
    edge = torch.cat([edge_x, edge_y], dim=1)  # (1, 2, H, W)
    edge = edge.squeeze(0)  # (2, H, W)
    
    # 归一化到[0, 1]
    edge = torch.abs(edge)
    edge = edge / (edge.max() + 1e-8)
    
    return edge


def preprocess_cifar10_to_edge(cifar10_root, output_dir, img_size=48):
    """
    将CIFAR10数据集转换为edge数据集（按类别保存）
    
    Args:
        cifar10_root: CIFAR10根目录
        output_dir: 输出目录
        img_size: 图像尺寸
    """
    print(f"\n{'='*80}")
    print("CIFAR10 RGB to Edge 预处理")
    print(f"{'='*80}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # CIFAR10类别名称
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 加载CIFAR10数据集（只使用训练集）
    print(f"加载CIFAR10数据集: {cifar10_root}")
    
    try:
        # 只加载训练集（50,000样本）
        train_dataset = datasets.CIFAR10(cifar10_root, train=True, transform=transform, download=False)
        print(f"✓ 成功加载CIFAR10数据集")
    except Exception as e:
        print(f"错误: 无法加载CIFAR10数据集: {e}")
        print(f"请确保数据集已下载到: {cifar10_root}")
        return
    
    print(f"训练集大小: {len(train_dataset)} 样本")
    print(f"类别数: {len(cifar10_classes)}")
    
    # 为每个类别创建输出目录
    print(f"\n创建类别目录...")
    for class_name in cifar10_classes:
        category_dir = os.path.join(output_dir, class_name)
        os.makedirs(category_dir, exist_ok=True)
        print(f"  创建目录: {class_name}")
    
    # 统计每个类别的样本数
    category_counts = {cat: 0 for cat in cifar10_classes}
    
    # 处理每个样本
    print(f"\n开始转换 RGB -> Edge...")
    
    for idx in tqdm(range(len(train_dataset)), desc="处理中", ncols=100):
        try:
            # 加载RGB图像和标签
            img, label = train_dataset[idx]
            
            # 获取类别名称
            category_name = cifar10_classes[label]
            
            # 转换为边缘图
            edge = sobel_edge_extraction(img)  # (2, H, W)
            
            # 保存为.pt文件（按类别组织）
            category_dir = os.path.join(output_dir, category_name)
            sample_idx = category_counts[category_name]
            output_path = os.path.join(category_dir, f"{sample_idx}.pt")
            torch.save((edge, torch.tensor(label)), output_path)
            
            # 更新计数
            category_counts[category_name] += 1
            
        except Exception as e:
            print(f"\n警告: 处理样本 {idx} 时出错: {e}")
            continue
    
    print(f"\n✓ 转换完成！")
    print(f"  输出目录: {output_dir}")
    
    # 统计信息
    total_files = sum(category_counts.values())
    print(f"  生成文件总数: {total_files}")
    print(f"\n各类别样本数:")
    for category, count in sorted(category_counts.items()):
        if count > 0:
            print(f"    {category}: {count}")
    
    # 验证生成的数据
    print(f"\n验证生成的数据...")
    first_category = cifar10_classes[0]
    test_file = os.path.join(output_dir, first_category, "0.pt")
    if os.path.exists(test_file):
        data, label = torch.load(test_file)
        print(f"  样本 ({first_category}/0.pt):")
        print(f"    边缘图形状: {data.shape}")
        print(f"    数据类型: {data.dtype}")
        print(f"    数据范围: [{data.min():.3f}, {data.max():.3f}]")
        print(f"    标签: {label}")
    
    print(f"\n{'='*80}")
    print("预处理完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10 RGB to Edge 预处理')
    parser.add_argument('--cifar10_root', type=str,
                       default='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10',
                       help='CIFAR10数据集根目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10_edge',
                       help='输出目录')
    parser.add_argument('--img_size', type=int, default=48,
                       help='图像尺寸')
    
    args = parser.parse_args()
    
    preprocess_cifar10_to_edge(args.cifar10_root, args.output_dir, args.img_size)

