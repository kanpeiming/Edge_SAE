# -*- coding: utf-8 -*-
"""
Extract VLM Features for Caltech101
为Caltech101数据集提取VLM特征

使用方法:
    python -m ESTL.preprocessing.extract_vlm_features \
        --rgb_root /path/to/caltech101 \
        --output_dir /path/to/vlm_features \
        --model_name openai/clip-vit-base-patch16
"""

import argparse
import os
import sys
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ESTL.vlm.feature_extractor import VLMFeatureExtractor


def extract_caltech101_vlm_features(
    rgb_root: str,
    output_dir: str,
    model_name: str = "openai/clip-vit-base-patch16",
    img_size: int = 224,
    skip_existing: bool = True
):
    """
    为Caltech101数据集提取VLM特征
    
    Args:
        rgb_root: RGB Caltech101数据集根目录
        output_dir: VLM特征输出目录
        model_name: VLM模型名称
        img_size: 图像大小
        skip_existing: 是否跳过已存在的特征
    """
    print("="*80)
    print("Caltech101 VLM特征提取".center(80))
    print("="*80)
    
    # 1. 初始化VLM特征提取器
    print(f"\n[1/4] 加载VLM模型: {model_name}")
    extractor = VLMFeatureExtractor(model_name=model_name)
    model_info = extractor.get_model_info()
    print(f"  模型信息:")
    print(f"    - 特征维度: {model_info['feature_dim']}")
    print(f"    - 图像大小: {model_info['image_size']}")
    print(f"    - 设备: {model_info['device']}")
    
    # 2. 扫描数据集
    print(f"\n[2/4] 扫描数据集: {rgb_root}")
    
    # 检查目录结构
    rgb_base = os.path.join(rgb_root, '101_ObjectCategories')
    if not os.path.exists(rgb_base):
        rgb_base = rgb_root
    
    if not os.path.exists(rgb_base):
        raise ValueError(f"数据集目录不存在: {rgb_base}")
    
    # 获取类别列表
    class_dirs = sorted([
        d for d in os.listdir(rgb_base)
        if os.path.isdir(os.path.join(rgb_base, d))
    ])
    
    # 移除背景和重复类别
    if 'BACKGROUND_Google' in class_dirs:
        class_dirs.remove('BACKGROUND_Google')
    if 'Faces_easy' in class_dirs:
        class_dirs.remove('Faces_easy')
    
    print(f"  找到 {len(class_dirs)} 个类别")
    
    # 3. 创建输出目录
    print(f"\n[3/4] 创建输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个类别创建子目录
    for class_name in class_dirs:
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
    
    # 4. 提取特征
    print(f"\n[4/4] 提取VLM特征...")
    
    total_images = 0
    extracted_images = 0
    skipped_images = 0
    failed_images = 0
    
    for class_name in tqdm(class_dirs, desc="处理类别"):
        class_dir = os.path.join(rgb_base, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        
        # 获取图像列表
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        total_images += len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            # 输出路径
            base_name = os.path.splitext(img_file)[0]
            output_path = os.path.join(class_output_dir, f"{base_name}.pkl")
            
            # 跳过已存在的特征
            if skip_existing and os.path.exists(output_path):
                skipped_images += 1
                continue
            
            try:
                # 加载图像
                rgb_image = Image.open(img_path).convert('RGB')
                rgb_image = rgb_image.resize((img_size, img_size), Image.BILINEAR)
                rgb_array = np.array(rgb_image)
                
                # 提取特征
                features = extractor.extract_features(
                    rgb_array,
                    class_name,
                    return_attention=True
                )
                
                # 保存特征
                with open(output_path, 'wb') as f:
                    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                extracted_images += 1
                
            except Exception as e:
                print(f"\n  ✗ 处理失败: {img_path}")
                print(f"    错误: {e}")
                failed_images += 1
    
    # 5. 统计信息
    print("\n" + "="*80)
    print("提取完成 - 统计信息".center(80))
    print("="*80)
    print(f"  总图像数: {total_images}")
    print(f"  已提取: {extracted_images}")
    print(f"  跳过: {skipped_images}")
    print(f"  失败: {failed_images}")
    print(f"  输出目录: {output_dir}")
    print("="*80)
    
    return {
        'total': total_images,
        'extracted': extracted_images,
        'skipped': skipped_images,
        'failed': failed_images,
        'output_dir': output_dir
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='提取Caltech101的VLM特征',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据路径
    parser.add_argument('--rgb_root', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/caltech101/caltech-101',
                        help='RGB Caltech101数据集根目录')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/vlm_features',
                        help='VLM特征输出目录')
    
    # VLM配置
    parser.add_argument('--model_name', type=str,
                        default='openai/clip-vit-base-patch16',
                        help='VLM模型名称')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像大小')
    
    # 其他选项
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已存在的特征文件')
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false',
                        help='重新提取所有特征')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 提取特征
    results = extract_caltech101_vlm_features(
        rgb_root=args.rgb_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        img_size=args.img_size,
        skip_existing=args.skip_existing
    )
    
    # 保存统计信息
    stats_path = os.path.join(args.output_dir, 'extraction_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Caltech101 VLM特征提取统计\n")
        f.write("="*80 + "\n\n")
        f.write(f"模型: {args.model_name}\n")
        f.write(f"图像大小: {args.img_size}\n")
        f.write(f"RGB数据路径: {args.rgb_root}\n")
        f.write(f"输出目录: {args.output_dir}\n\n")
        f.write(f"总图像数: {results['total']}\n")
        f.write(f"已提取: {results['extracted']}\n")
        f.write(f"跳过: {results['skipped']}\n")
        f.write(f"失败: {results['failed']}\n")
    
    print(f"\n✓ 统计信息已保存到: {stats_path}")


if __name__ == '__main__':
    main()

