# -*- coding: utf-8 -*-
"""
@file: compare_rgb_dvs_samples.py
@description: 输出CIFAR10和DVS-CIFAR10数据集的同一个样本进行对比
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dataloader.cifar import TLCIFAR10

# 数据集路径配置
USER_NAME = 'zhan'
DIR = {
    'CIFAR10': f'/data/{USER_NAME}/Event_Camera_Datasets/CIFAR10/cifar10',
    'CIFAR10DVS': f'/data/{USER_NAME}/Event_Camera_Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat',
}


# CIFAR10类别名称
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_rgb_dvs_comparison(num_samples=9, save_path='/data/kpm/picture/rgb_dvs_comparison_hd.png', figsize=(24, 12), dpi=300):
    """
    可视化RGB和DVS的高分辨率对比
    
    Args:
        num_samples: 要显示的样本数量（每个类别一个样本）
        save_path: 保存图片的路径
        figsize: 图像大小
        dpi: 图像分辨率
    """
    
    # 定义变换
    rgb_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dvs_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集
    dataset = TLCIFAR10(
        root=DIR['CIFAR10'],
        dvs_root=DIR['CIFAR10DVS'],
        train=True,
        transform=rgb_transform,
        dvs_transform=dvs_transform,
        download=True
    )
    
    print("开始生成RGB-DVS对比图像...")
    
    # 为每个类别选择一个代表性样本
    print("正在选择代表性样本...")
    class_samples = []
    found_classes = set()
    max_search = min(5000, len(dataset))  # 限制搜索范围，避免遍历整个数据集
    
    for idx in range(max_search):
        if len(class_samples) >= min(num_samples, 10):
            break
            
        if idx % 1000 == 0:
            print(f"已搜索 {idx}/{max_search} 个样本，找到 {len(class_samples)} 个类别")
            
        try:
            (_, _), label = dataset[idx]
            if label not in found_classes and label < min(num_samples, 10):
                class_samples.append(idx)
                found_classes.add(label)
                print(f"找到类别 {label} ({CIFAR10_CLASSES[label]}) 的样本，索引: {idx}")
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            continue
    
    print(f"选择了 {len(class_samples)} 个样本，涵盖类别: {sorted(found_classes)}")
    
    if len(class_samples) == 0:
        print("错误：未找到任何有效样本")
        return None
    
    # 创建图形 - 2行：RGB、DVS累积
    fig, axes = plt.subplots(2, len(class_samples), figsize=figsize)
    fig.suptitle('RGB vs DVS Comparison (High Resolution)', fontsize=20, y=0.98)
    
    # 确保axes是2维数组，即使只有一列
    if len(class_samples) == 1:
        axes = axes.reshape(2, 1)
    
    # 设置行标题
    row_labels = ['RGB Image', 'DVS Cumulative']
    for row_idx, label in enumerate(row_labels):
        if len(class_samples) > 0:
            axes[row_idx, 0].set_ylabel(label, fontsize=14, rotation=90, labelpad=30)
    
    for i, sample_idx in enumerate(class_samples):
        # 获取样本
        (rgb_img, dvs_img), label = dataset[sample_idx]
        
        # 反归一化RGB图像用于显示
        rgb_img_display = rgb_img.clone()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        rgb_img_display = rgb_img_display * std + mean
        rgb_img_display = torch.clamp(rgb_img_display, 0, 1)
        
        # 显示RGB图像
        print(f"正在处理第 {i+1}/{len(class_samples)} 个样本...")
        axes[0, i].imshow(rgb_img_display.permute(1, 2, 0))
        axes[0, i].set_title(f'{CIFAR10_CLASSES[label]}', fontsize=12, pad=10)
        axes[0, i].axis('off')
        
        # 显示DVS累积图像（所有时间步的累积）- 使用蓝绿色渲染
        dvs_cumsum = torch.sum(dvs_img, dim=0)  # 在时间维度上累积
        if dvs_cumsum.shape[0] == 2:
            # 创建蓝绿色渲染：正事件用青色，负事件用蓝色
            dvs_cumsum_display = torch.zeros(3, dvs_cumsum.shape[1], dvs_cumsum.shape[2])
            dvs_cumsum_display[1] = dvs_cumsum[0]  # 绿色通道 - 正事件
            dvs_cumsum_display[2] = dvs_cumsum[0]  # 蓝色通道 - 正事件（青色 = 绿+蓝）
            dvs_cumsum_display[2] += dvs_cumsum[1]  # 蓝色通道 - 负事件
        else:
            dvs_cumsum_display = dvs_cumsum
            
        axes[1, i].imshow(dvs_cumsum_display.permute(1, 2, 0))
        axes[1, i].set_title(f'DVS Cumulative', fontsize=12, pad=10)
        axes[1, i].axis('off')
        print(f"样本 {i+1} 处理完成")
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 调整布局并保存图片
    print("正在保存图像...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为标题留出空间
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"高分辨率RGB-DVS对比图已保存到: {save_path}")

def save_individual_samples(num_classes=5, save_dir='/data/kpm/picture/individual_samples/'):
    """
    保存单个样本的RGB和DVS版本
    
    Args:
        num_classes: 要保存的类别数量
        save_dir: 保存目录
    """
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义变换（不进行归一化，便于保存原始图像）
    rgb_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    
    dvs_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集
    dataset = TLCIFAR10(
        root=DIR['CIFAR10'],
        dvs_root=DIR['CIFAR10DVS'],
        train=True,
        transform=rgb_transform,
        dvs_transform=dvs_transform,
        download=True
    )
    
    print("开始保存单个样本...")
    
    # 为每个类别选择一个代表性样本
    class_samples = []
    for class_id in range(min(num_classes, 10)):
        for idx in range(len(dataset)):
            (_, _), label = dataset[idx]
            if label == class_id:
                class_samples.append(idx)
                break
    
    for sample_idx in class_samples:
        (rgb_img, dvs_img), label = dataset[sample_idx]
        class_name = CIFAR10_CLASSES[label]
        
        # 创建子图 - 2列：RGB、DVS累积
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f'Sample {sample_idx} - {class_name}', fontsize=16)
        
        # 反归一化RGB图像用于显示
        rgb_img_display = rgb_img.clone()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        rgb_img_display = rgb_img_display * std + mean
        rgb_img_display = torch.clamp(rgb_img_display, 0, 1)
        
        # RGB图像
        axes[0].imshow(rgb_img_display.permute(1, 2, 0))
        axes[0].set_title('RGB Image', fontsize=12)
        axes[0].axis('off')
        
        # DVS累积图像 - 使用蓝绿色渲染
        dvs_cumsum = torch.sum(dvs_img, dim=0)
        if dvs_cumsum.shape[0] == 2:
            # 创建蓝绿色渲染：正事件用青色，负事件用蓝色
            dvs_cumsum_display = torch.zeros(3, dvs_cumsum.shape[1], dvs_cumsum.shape[2])
            dvs_cumsum_display[1] = dvs_cumsum[0]  # 绿色通道 - 正事件
            dvs_cumsum_display[2] = dvs_cumsum[0]  # 蓝色通道 - 正事件（青色 = 绿+蓝）
            dvs_cumsum_display[2] += dvs_cumsum[1]  # 蓝色通道 - 负事件
        else:
            dvs_cumsum_display = dvs_cumsum
        
        axes[1].imshow(dvs_cumsum_display.permute(1, 2, 0))
        axes[1].set_title('DVS Cumulative', fontsize=12)
        axes[1].axis('off')
        
        # 保存图像
        sample_path = os.path.join(save_dir, f'sample_{sample_idx}_{class_name}.png')
        plt.tight_layout()
        plt.savefig(sample_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"样本 {sample_idx} ({class_name}) 已保存（包含边缘检测）")

def print_dataset_info():
    """
    打印数据集信息
    """
    
    # 简单的变换
    simple_transform = transforms.ToTensor()
    
    dataset = TLCIFAR10(
        root=DIR['CIFAR10'],
        dvs_root=DIR['CIFAR10DVS'],
        train=True,
        transform=simple_transform,
        dvs_transform=simple_transform,
        download=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"RGB数据大小: {dataset.get_len()[0]}")
    print(f"DVS数据大小: {dataset.get_len()[1]}")
    
    # 获取一个样本查看数据格式
    (rgb_img, dvs_img), label = dataset[0]
    print(f"\n样本信息:")
    print(f"RGB图像形状: {rgb_img.shape}")
    print(f"DVS图像形状: {dvs_img.shape}")
    print(f"标签: {label} ({CIFAR10_CLASSES[label]})")

if __name__ == '__main__':
    print("=== CIFAR10 vs DVS-CIFAR10 数据集对比 ===")
    
    # 打印数据集信息
    print_dataset_info()
    
    print("\n=== 生成高分辨率RGB-DVS对比可视化 ===")
    visualize_rgb_dvs_comparison(
        num_samples=15,
        save_path='/data/kpm/picture/rgb_dvs_comparison_hd.png',
        figsize=(30, 12),
        dpi=300
    )
    
    print("\n=== 保存单个样本 ===")
    save_individual_samples(
        num_classes=10,
        save_dir='/data/kpm/picture/individual_samples/'
    )
    
    print("\n完成！")