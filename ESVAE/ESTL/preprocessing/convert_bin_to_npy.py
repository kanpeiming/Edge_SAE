# -*- coding: utf-8 -*-
"""
Convert N-Caltech101 DVS data from .bin to .npy format
将N-Caltech101 DVS数据从.bin格式转换为.npy格式

使用方法:
    python -m ESTL.preprocessing.convert_bin_to_npy \
        --input_dir /path/to/n-caltech101/Caltech101 \
        --output_dir /path/to/n-caltech101_npy/Caltech101 \
        --img_size 48 \
        --T 10

优势:
    - .bin格式: 需要实时解析，慢
    - .npy格式: 预处理好的数组，快10-20倍
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def parse_bin_file(bin_path, img_size=48, T=10):
    """
    解析.bin格式的DVS文件
    
    Args:
        bin_path: .bin文件路径
        img_size: 目标图像大小
        T: 时间步数
        
    Returns:
        dvs_data: (T, 2, img_size, img_size) numpy array
    """
    with open(bin_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)
    
    # 每个事件5字节
    num_events = len(raw_data) // 5
    
    if num_events == 0:
        # 空文件，返回零数组
        return np.zeros((T, 2, img_size, img_size), dtype=np.float32)
    
    # 原始分辨率
    H_orig, W_orig = 180, 240
    C = 2  # ON/OFF events
    
    # 初始化累积帧
    frames = np.zeros((T, C, H_orig, W_orig), dtype=np.float32)
    
    # 解析事件
    events = []
    for i in range(num_events):
        offset = i * 5
        if offset + 5 > len(raw_data):
            break
        
        # 读取5字节
        byte0 = int(raw_data[offset])
        byte1 = int(raw_data[offset+1])
        byte2 = int(raw_data[offset+2])
        byte3 = int(raw_data[offset+3])
        byte4 = int(raw_data[offset+4])
        
        # 解析坐标和极性
        x = byte0
        y = byte1
        pol_ts = (byte2 << 16) | (byte3 << 8) | byte4
        pol = (pol_ts >> 23) & 0x1
        ts = pol_ts & 0x7FFFFF
        
        # 限制坐标范围
        if 0 <= x < W_orig and 0 <= y < H_orig:
            events.append((x, y, ts, pol))
    
    if len(events) > 0:
        events = np.array(events)
        timestamps = events[:, 2]
        
        # 归一化时间戳到 [0, T-1]
        if timestamps.max() > timestamps.min():
            time_bins = ((timestamps - timestamps.min()) / 
                        (timestamps.max() - timestamps.min() + 1e-6) * (T - 1)).astype(int)
        else:
            time_bins = np.zeros(len(events), dtype=int)
        
        # 累积事件到帧
        for idx, (x, y, ts, pol) in enumerate(events):
            t_bin = min(time_bins[idx], T - 1)
            frames[t_bin, pol, int(y), int(x)] += 1.0
        
        # 对数变换 + 归一化
        frames_nonzero = frames[frames > 0]
        if len(frames_nonzero) > 0:
            frames = np.log1p(frames)
            max_val = frames.max()
            if max_val > 0:
                frames = frames / max_val
    
    # Resize到目标大小
    if H_orig != img_size or W_orig != img_size:
        import cv2
        resized_frames = []
        for t in range(T):
            frame = frames[t]  # (2, H_orig, W_orig)
            # 转置为 (H, W, C) 用于cv2
            frame_hwc = frame.transpose(1, 2, 0)  # (H_orig, W_orig, 2)
            frame_resized = cv2.resize(
                frame_hwc,
                (img_size, img_size),
                interpolation=cv2.INTER_LINEAR
            )
            # 转回 (C, H, W)
            frame_chw = frame_resized.transpose(2, 0, 1)  # (2, img_size, img_size)
            resized_frames.append(frame_chw)
        frames = np.stack(resized_frames, axis=0)  # (T, 2, img_size, img_size)
    
    return frames.astype(np.float32)


def convert_dataset(input_dir, output_dir, img_size=48, T=10, skip_existing=True):
    """
    转换整个数据集
    
    Args:
        input_dir: 输入目录（.bin格式）
        output_dir: 输出目录（.npy格式）
        img_size: 目标图像大小
        T: 时间步数
        skip_existing: 是否跳过已存在的文件
    """
    print("="*80)
    print("N-Caltech101 DVS数据格式转换: .bin → .npy".center(80))
    print("="*80)
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取类别列表
    class_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    
    # 移除不需要的类别
    if 'Faces_easy' in class_dirs:
        class_dirs.remove('Faces_easy')
        print(f"⚠ 跳过 'Faces_easy' 类别")
    
    print(f"\n配置:")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  类别数: {len(class_dirs)}")
    print(f"  目标大小: {img_size}×{img_size}")
    print(f"  时间步数: {T}")
    print(f"  跳过已存在: {skip_existing}")
    
    # 统计信息
    total_files = 0
    converted_files = 0
    skipped_files = 0
    failed_files = 0
    
    # 转换每个类别
    print(f"\n开始转换...")
    for class_name in tqdm(class_dirs, desc="处理类别"):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        # 创建输出类别目录
        os.makedirs(output_class_dir, exist_ok=True)
        
        # 获取所有.bin文件
        bin_files = sorted([
            f for f in os.listdir(input_class_dir)
            if f.endswith('.bin')
        ])
        
        total_files += len(bin_files)
        
        # 转换每个文件
        for bin_file in bin_files:
            bin_path = os.path.join(input_class_dir, bin_file)
            npy_file = bin_file.replace('.bin', '.npy')
            npy_path = os.path.join(output_class_dir, npy_file)
            
            # 跳过已存在的文件
            if skip_existing and os.path.exists(npy_path):
                skipped_files += 1
                continue
            
            try:
                # 解析.bin文件
                dvs_data = parse_bin_file(bin_path, img_size=img_size, T=T)
                
                # 保存为.npy
                np.save(npy_path, dvs_data)
                
                converted_files += 1
                
            except Exception as e:
                print(f"\n✗ 转换失败: {bin_path}")
                print(f"  错误: {e}")
                failed_files += 1
    
    # 统计信息
    print("\n" + "="*80)
    print("转换完成 - 统计信息".center(80))
    print("="*80)
    print(f"  总文件数: {total_files}")
    print(f"  已转换: {converted_files}")
    print(f"  跳过: {skipped_files}")
    print(f"  失败: {failed_files}")
    print(f"  输出目录: {output_dir}")
    print("="*80)
    
    # 保存转换信息
    info_file = os.path.join(output_dir, 'conversion_info.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("N-Caltech101 DVS数据格式转换信息\n")
        f.write("="*80 + "\n\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"图像大小: {img_size}×{img_size}\n")
        f.write(f"时间步数: {T}\n\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"已转换: {converted_files}\n")
        f.write(f"跳过: {skipped_files}\n")
        f.write(f"失败: {failed_files}\n\n")
        f.write("使用方法:\n")
        f.write("  在训练脚本中设置:\n")
        f.write(f"  --dvs_root {output_dir}\n")
        f.write("  --dvs_format npy\n")
    
    print(f"\n✓ 转换信息已保存到: {info_file}")
    
    return {
        'total': total_files,
        'converted': converted_files,
        'skipped': skipped_files,
        'failed': failed_files
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='将N-Caltech101 DVS数据从.bin转换为.npy格式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 路径参数
    parser.add_argument('--input_dir', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101/Caltech101',
                        help='输入目录（.bin格式）')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101_npy/Caltech101',
                        help='输出目录（.npy格式）')
    
    # 转换参数
    parser.add_argument('--img_size', type=int, default=48,
                        help='目标图像大小')
    parser.add_argument('--T', type=int, default=10,
                        help='时间步数')
    
    # 其他选项
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已存在的文件')
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false',
                        help='重新转换所有文件')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查依赖
    try:
        import cv2
    except ImportError:
        print("错误: 需要安装opencv-python")
        print("请运行: pip install opencv-python")
        return
    
    # 转换数据集
    results = convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        T=args.T,
        skip_existing=args.skip_existing
    )
    
    # 显示使用说明
    print("\n" + "="*80)
    print("使用转换后的数据训练".center(80))
    print("="*80)
    print("\n训练命令:")
    print(f"""
python train_caltech101_baseline.py \\
    --model spikformer \\
    --dvs_root {args.output_dir} \\
    --dvs_format npy \\
    --img_shape {args.img_size} \\
    --T {args.T} \\
    --enable_augmentation
""")
    print("预期速度提升: 10-20倍！")
    print("="*80)


if __name__ == '__main__':
    main()

