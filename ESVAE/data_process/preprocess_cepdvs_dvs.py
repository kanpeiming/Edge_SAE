# -*- coding: utf-8 -*-
"""
CEP-DVS DVS数据预处理脚本
将CEP-DVS的.mat文件转换为.pt文件以加速训练

功能：
- 将.mat文件中的事件数据转换为时间序列帧
- 保存为.pt文件，避免训练时重复转换
- 大幅提升训练速度

使用方法：
python preprocess_cepdvs_dvs.py --time_bins 10 --img_size 48
"""

import argparse
import os
import torch
import numpy as np
import scipy.io as scio
from tqdm import tqdm
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
        # 从CSV行中提取类别名称
        match = re.search(r'(?:\S*\s){4}(\S+)', row)
        if match:
            label_name = match.group(1)
            if label_name in CEPDVS_CLASSES:
                label_dict[i] = CEPDVS_CLASSES[label_name]
    
    print(f"✓ 从pathFile.csv加载了 {len(label_dict)} 个标签映射")
    return label_dict


def load_mat_file(file_path):
    """
    加载.mat文件中的DVS事件数据
    
    Args:
        file_path: .mat文件路径
    
    Returns:
        event_dict: 包含 {'x', 'y', 'p', 'ts'} 的字典
    """
    data = scio.loadmat(file_path, verify_compressed_data_integrity=False)
    event_dict = {}
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            event_dict[key] = np.squeeze(data[key].astype(np.int64))
            # 将极性 -1 转换为 0 (标准化为0/1)
            if key == "p":
                event_dict[key] = np.where(event_dict[key] == -1, 0, event_dict[key])
    return event_dict


def events_to_frames(events, height=180, width=240, time_bins=10):
    """
    将DVS事件转换为时间序列帧
    
    Args:
        events: 事件字典，包含 {'x', 'y', 'p', 'ts'}
        height: 图像高度
        width: 图像宽度
        time_bins: 时间分箱数量
    
    Returns:
        frames: (time_bins, 2, H, W) 的张量，2通道分别代表正负极性
    """
    x = events['x']
    y = events['y']
    p = events['p']
    ts = events['ts']
    
    # 归一化时间戳到[0, time_bins-1]
    if len(ts) > 0:
        ts_min = ts.min()
        ts_max = ts.max()
        if ts_max > ts_min:
            ts_norm = ((ts - ts_min) / (ts_max - ts_min) * (time_bins - 1)).astype(np.int32)
        else:
            ts_norm = np.zeros_like(ts, dtype=np.int32)
    else:
        # 空事件：返回全零张量
        return torch.zeros((time_bins, 2, height, width), dtype=torch.float32)
    
    # 初始化帧
    frames = np.zeros((time_bins, 2, height, width), dtype=np.float32)
    
    # 填充事件到对应的时间帧和极性通道
    for i in range(len(x)):
        xi, yi, pi, ti = x[i], y[i], p[i], ts_norm[i]
        if 0 <= xi < width and 0 <= yi < height and 0 <= ti < time_bins:
            # pi=0 -> 通道0 (负极性), pi=1 -> 通道1 (正极性)
            frames[ti, pi, yi, xi] += 1
    
    # 归一化
    frames = frames / (frames.max() + 1e-8)
    
    return torch.from_numpy(frames).float()


def resize_frames_fast(frames, target_size):
    """
    快速Resize帧序列（使用torch的插值，避免PIL转换）
    
    Args:
        frames: (T, C, H, W) 的张量
        target_size: 目标尺寸
    
    Returns:
        resized_frames: (T, C, target_size, target_size) 的张量
    """
    import torch.nn.functional as F
    T, C, H, W = frames.shape
    
    # 将 (T, C, H, W) reshape为 (T*C, 1, H, W) 以便批量resize
    frames_flat = frames.view(T * C, 1, H, W)
    
    # 使用双线性插值resize
    resized_flat = F.interpolate(frames_flat, size=(target_size, target_size), 
                                 mode='bilinear', align_corners=False)
    
    # reshape回 (T, C, target_size, target_size)
    resized_frames = resized_flat.view(T, C, target_size, target_size)
    
    return resized_frames


def preprocess_cepdvs_dvs(dvs_root, output_dir, csv_root, time_bins=10, img_size=48):
    """
    将CEP-DVS DVS数据集预处理为.pt文件
    
    Args:
        dvs_root: DVS .mat文件目录
        output_dir: 输出目录
        csv_root: CEP-DVS根目录（包含pathFile.csv）
        time_bins: 时间分箱数量
        img_size: 图像尺寸
    """
    print(f"\n{'='*80}")
    print("CEP-DVS DVS数据预处理")
    print(f"{'='*80}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载pathFile.csv获取标签映射
    csv_path = os.path.join(csv_root, 'pathFile.csv')
    print(f"加载标签映射: {csv_path}")
    label_dict = load_pathfile_csv(csv_path)
    
    if not label_dict:
        raise ValueError("无法加载pathFile.csv或标签映射为空")
    
    # 加载所有.mat文件
    print(f"\n加载DVS数据集: {dvs_root}")
    
    if not os.path.exists(dvs_root):
        raise FileNotFoundError(f"DVS数据目录不存在: {dvs_root}")
    
    # 获取所有.mat文件
    files = sorted([f for f in os.listdir(dvs_root) if f.endswith('.mat')])
    print(f"数据集大小: {len(files)} 样本")
    print(f"时间分箱数: {time_bins}")
    print(f"目标尺寸: {img_size}×{img_size}")
    
    # 处理每个样本
    print(f"\n开始转换 DVS (.mat) -> 帧序列 (.pt)...")
    
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
            
            # 加载.mat文件
            file_path = os.path.join(dvs_root, file_name)
            events = load_mat_file(file_path)
            
            # 转换为帧序列 (T, 2, 180, 240)
            frames = events_to_frames(events, height=180, width=240, time_bins=time_bins)
            
            # Resize到目标尺寸（使用快速方法）
            if img_size != 180:
                frames = resize_frames_fast(frames, img_size)
            
            # 保存为.pt文件（保持原始文件名）
            output_file_name = file_name.replace('.mat', '.pt')
            output_path = os.path.join(output_dir, output_file_name)
            torch.save((frames, torch.tensor(label)), output_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"\n警告: 处理样本 {file_name} 时出错: {e}")
            skipped_count += 1
            continue
    
    print(f"\n✓ 转换完成！")
    print(f"  输出目录: {output_dir}")
    print(f"  成功处理: {processed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件（未在pathFile.csv中找到标签或处理失败）")
    
    # 验证生成的数据
    print(f"\n验证生成的数据...")
    output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])
    if output_files:
        test_file = os.path.join(output_dir, output_files[0])
        data, label = torch.load(test_file)
        print(f"  样本 ({output_files[0]}):")
        print(f"    帧序列形状: {data.shape}")
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
    DVS_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/MAT/img'  # DVS .mat文件目录
    OUTPUT_DIR = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/dvs_processed'  # DVS处理后输出目录
    TIME_BINS = 10  # 时间分箱数量
    IMG_SIZE = 48  # 图像尺寸
    # =================================================
    
    parser = argparse.ArgumentParser(description='CEP-DVS DVS数据预处理')
    parser.add_argument('--dvs_root', type=str, default=DVS_ROOT,
                       help='CEP-DVS DVS .mat文件目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='输出目录')
    parser.add_argument('--csv_root', type=str, default=CEPDVS_ROOT,
                       help='CEP-DVS根目录（包含pathFile.csv）')
    parser.add_argument('--time_bins', type=int, default=TIME_BINS,
                       help='时间分箱数量')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                       help='图像尺寸')
    
    args = parser.parse_args()
    
    print(f"配置信息:")
    print(f"  CEP-DVS根目录: {args.csv_root}")
    print(f"  DVS数据目录: {args.dvs_root}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  时间分箱数: {args.time_bins}")
    print(f"  图像尺寸: {args.img_size}×{args.img_size}")
    print(f"  pathFile.csv路径: {os.path.join(args.csv_root, 'pathFile.csv')}\n")
    
    preprocess_cepdvs_dvs(args.dvs_root, args.output_dir, args.csv_root, args.time_bins, args.img_size)

