# -*- coding: utf-8 -*-
"""
CEP-DVS RGB to Edge 预处理脚本
将CEP-DVS RGB图像转换为Sobel边缘图并保存

功能：
- 使用Sobel算子提取边缘（2通道：水平+垂直）
- 保存为.pt文件，格式与DVS数据一致
- 节省训练时的GPU显存和计算时间

使用方法：
python preprocess_cepdvs_rgb2edge.py --output_dir /home/user/kpm/kpm/Dataset/CEP-DVS/data/edge --img_size 48
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import re

# CEP-DVS类别映射（20个类别）
CEPDVS_CLASSES = {
    'aquatic_mammals': 0,
    'fish': 1,
    'flowers': 2,
    'food_containers': 3,
    'fruit_and_vegetables': 4,
    'household_electrical_devices': 5,
    'household_furniture': 6,
    'insects': 7,
    'large_carnivores': 8,
    'large_man-made_outdoor_things': 9,
    'large_natural_outdoor_scenes': 10,
    'large_omnivores_and_herbivores': 11,
    'medium_mammals': 12,
    'non-insect_invertebrates': 13,
    'people': 14,
    'reptiles': 15,
    'small_mammals': 16,
    'trees': 17,
    'vehicles_1': 18,
    'vehicles_2': 19,
}


def load_pathfile_csv(csv_path):
    """
    加载pathFile.csv获取文件名到类别的映射
    
    Args:
        csv_path: pathFile.csv路径
    
    Returns:
        label_dict: {file_index: label} 的字典
    """
    label_dict = {}
    
    if not os.path.exists(csv_path):
        print(f"错误: pathFile.csv 不存在: {csv_path}")
        return label_dict
    
    record_file = np.genfromtxt(open(csv_path, "rb"), delimiter=",", skip_header=1, dtype='U')
    
    for i, row in enumerate(record_file):
        # 从CSV行中提取类别名称（格式：序号 空格 空格 空格 空格 类别名）
        match = re.search(r'(?:\S*\s){4}(\S+)', row)
        if match:
            label_name = match.group(1)
            if label_name in CEPDVS_CLASSES:
                label_dict[i] = CEPDVS_CLASSES[label_name]
    
    print(f"✓ 从pathFile.csv加载了 {len(label_dict)} 个标签映射")
    return label_dict


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


def preprocess_cepdvs_rgb_to_edge(rgb_root, output_dir, csv_root, img_size=48):
    """
    将CEP-DVS RGB数据集转换为edge数据集（扁平结构，使用pathFile.csv获取正确标签）
    
    Args:
        rgb_root: RGB数据根目录
        output_dir: 输出目录
        csv_root: CEP-DVS根目录（包含pathFile.csv）
        img_size: 图像尺寸
    """
    print(f"\n{'='*80}")
    print("CEP-DVS RGB to Edge 预处理")
    print(f"{'='*80}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载pathFile.csv获取标签映射
    csv_path = os.path.join(csv_root, 'pathFile.csv')
    print(f"加载标签映射: {csv_path}")
    label_dict = load_pathfile_csv(csv_path)
    
    if not label_dict:
        raise ValueError("无法加载pathFile.csv或标签映射为空")
    
    # 加载所有RGB图像
    print(f"\n加载RGB数据集: {rgb_root}")
    
    if not os.path.exists(rgb_root):
        raise FileNotFoundError(f"RGB数据目录不存在: {rgb_root}")
    
    # 获取所有.jpg文件
    files = sorted([f for f in os.listdir(rgb_root) if f.endswith('.jpg')])
    print(f"数据集大小: {len(files)} 样本")
    
    # 处理每个样本
    print(f"\n开始转换 RGB -> Edge...")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    processed_count = 0
    skipped_count = 0
    
    for idx, file_name in enumerate(tqdm(files, desc="处理中", ncols=100)):
        try:
            # 从文件名提取索引
            file_idx = int(file_name.split('.')[0])
            
            # 从label_dict获取标签
            if file_idx not in label_dict:
                skipped_count += 1
                continue
            
            label = label_dict[file_idx]
            
            # 加载RGB图像
            file_path = os.path.join(rgb_root, file_name)
            img = Image.open(file_path).convert('RGB')
            
            # 转换为tensor
            img_tensor = transform(img)
            
            # 转换为边缘图
            edge = sobel_edge_extraction(img_tensor)  # (2, H, W)
            
            # 保存为.pt文件（保持原始文件名）
            output_file_name = file_name.replace('.jpg', '.pt')
            output_path = os.path.join(output_dir, output_file_name)
            torch.save((edge, torch.tensor(label)), output_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"\n警告: 处理样本 {file_name} 时出错: {e}")
            skipped_count += 1
            continue
    
    print(f"\n✓ 转换完成！")
    print(f"  输出目录: {output_dir}")
    print(f"  成功处理: {processed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件（未在pathFile.csv中找到标签）")
    
    # 验证生成的数据
    print(f"\n验证生成的数据...")
    output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])
    if output_files:
        test_file = os.path.join(output_dir, output_files[0])
        data, label = torch.load(test_file)
        print(f"  样本 ({output_files[0]}):")
        print(f"    边缘图形状: {data.shape}")
        print(f"    数据类型: {data.dtype}")
        print(f"    数据范围: [{data.min():.3f}, {data.max():.3f}]")
        print(f"    标签: {label}")
        
        # 统计标签分布
        label_counts = {}
        for pt_file in output_files[:100]:  # 采样前100个文件
            _, lbl = torch.load(os.path.join(output_dir, pt_file))
            lbl = lbl.item() if torch.is_tensor(lbl) else lbl
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"\n  标签分布（前100个样本）:")
        for lbl in sorted(label_counts.keys()):
            print(f"    类别 {lbl}: {label_counts[lbl]} 个样本")
    
    print(f"\n{'='*80}")
    print("预处理完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 请根据您的实际路径修改以下配置
    CEPDVS_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS'  # CEP-DVS根目录（包含pathFile.csv）
    RGB_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/img'  # RGB图像目录
    OUTPUT_DIR = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/edge'  # Edge输出目录
    IMG_SIZE = 48  # 图像尺寸
    # =================================================
    
    parser = argparse.ArgumentParser(description='CEP-DVS RGB to Edge 预处理')
    parser.add_argument('--rgb_root', type=str, default=RGB_ROOT,
                       help='CEP-DVS RGB数据集根目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='输出目录')
    parser.add_argument('--csv_root', type=str, default=CEPDVS_ROOT,
                       help='CEP-DVS根目录（包含pathFile.csv）')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                       help='图像尺寸')
    
    args = parser.parse_args()
    
    print(f"配置信息:")
    print(f"  CEP-DVS根目录: {args.csv_root}")
    print(f"  RGB数据目录: {args.rgb_root}")
    print(f"  Edge输出目录: {args.output_dir}")
    print(f"  图像尺寸: {args.img_size}×{args.img_size}")
    print(f"  pathFile.csv路径: {os.path.join(args.csv_root, 'pathFile.csv')}\n")
    
    preprocess_cepdvs_rgb_to_edge(args.rgb_root, args.output_dir, args.csv_root, args.img_size)

