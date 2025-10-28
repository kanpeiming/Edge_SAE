# -*- coding: utf-8 -*-
"""
Generate Semantic Edges for Caltech101
为Caltech101数据集生成语义边缘

使用方法:
    python -m ESTL.preprocessing.generate_semantic_edges \
        --rgb_root /path/to/caltech101 \
        --vlm_features_dir /path/to/vlm_features \
        --output_dir /path/to/semantic_edges
"""

import argparse
import os
import sys
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ESTL.vlm.semantic_edge import SemanticEdgeGenerator, MultiScaleEdgeDetector


def generate_caltech101_semantic_edges(
    rgb_root: str,
    vlm_features_dir: str,
    output_dir: str,
    img_size: int = 224,
    sobel_weight: float = 0.3,
    canny_weight: float = 0.7,
    skip_existing: bool = True,
    save_visualization: bool = False
):
    """
    为Caltech101数据集生成语义边缘
    
    Args:
        rgb_root: RGB Caltech101数据集根目录
        vlm_features_dir: VLM特征目录
        output_dir: 语义边缘输出目录
        img_size: 图像大小
        sobel_weight: Sobel权重（密集边缘）
        canny_weight: Canny权重（稀疏边缘）
        skip_existing: 是否跳过已存在的边缘
        save_visualization: 是否保存可视化结果
    """
    print("="*80)
    print("Caltech101 语义边缘生成".center(80))
    print("="*80)
    
    # 1. 初始化边缘生成器
    print(f"\n[1/4] 初始化边缘生成器")
    print(f"  Sobel权重: {sobel_weight} (密集边缘)")
    print(f"  Canny权重: {canny_weight} (稀疏边缘，匹配DVS)")
    
    edge_detector = MultiScaleEdgeDetector(
        use_sobel=True,
        use_canny=True,
        sobel_weight=sobel_weight,
        canny_weight=canny_weight,
        canny_threshold1=50,
        canny_threshold2=150
    )
    
    edge_generator = SemanticEdgeGenerator(
        edge_detector=edge_detector,
        use_vlm_weighting=True,
        use_class_enhancement=True,
        attention_power=2.0,
        min_edge_threshold=0.1
    )
    
    # 2. 扫描数据集
    print(f"\n[2/4] 扫描数据集")
    print(f"  RGB数据: {rgb_root}")
    print(f"  VLM特征: {vlm_features_dir}")
    
    # 检查目录
    rgb_base = os.path.join(rgb_root, '101_ObjectCategories')
    if not os.path.exists(rgb_base):
        rgb_base = rgb_root
    
    if not os.path.exists(rgb_base):
        raise ValueError(f"RGB数据集目录不存在: {rgb_base}")
    
    if not os.path.exists(vlm_features_dir):
        raise ValueError(
            f"VLM特征目录不存在: {vlm_features_dir}\n"
            f"请先运行 extract_vlm_features.py 提取VLM特征"
        )
    
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
    
    if save_visualization:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  可视化目录: {vis_dir}")
    
    # 为每个类别创建子目录
    for class_name in class_dirs:
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        if save_visualization:
            class_vis_dir = os.path.join(vis_dir, class_name)
            os.makedirs(class_vis_dir, exist_ok=True)
    
    # 4. 生成语义边缘
    print(f"\n[4/4] 生成语义边缘...")
    
    total_images = 0
    generated_images = 0
    skipped_images = 0
    failed_images = 0
    missing_vlm_features = 0
    
    for class_name in tqdm(class_dirs, desc="处理类别"):
        class_dir = os.path.join(rgb_base, class_name)
        class_vlm_dir = os.path.join(vlm_features_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        
        # 获取图像列表
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        total_images += len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            # 输出路径
            output_path = os.path.join(class_output_dir, f"{base_name}.npy")
            
            # 跳过已存在的边缘
            if skip_existing and os.path.exists(output_path):
                skipped_images += 1
                continue
            
            try:
                # 1. 加载RGB图像
                rgb_image = Image.open(img_path).convert('RGB')
                rgb_image = rgb_image.resize((img_size, img_size), Image.BILINEAR)
                rgb_array = np.array(rgb_image)
                
                # 2. 加载VLM特征
                vlm_feature_path = os.path.join(class_vlm_dir, f"{base_name}.pkl")
                
                if not os.path.exists(vlm_feature_path):
                    missing_vlm_features += 1
                    failed_images += 1
                    continue
                
                with open(vlm_feature_path, 'rb') as f:
                    vlm_features = pickle.load(f)
                
                # 3. 生成语义边缘
                result = edge_generator.generate(
                    rgb_array,
                    vlm_features,
                    class_name
                )
                
                semantic_edges = result['semantic_edges']
                
                # 4. 保存边缘图
                np.save(output_path, semantic_edges)
                
                # 5. 保存可视化（可选）
                if save_visualization and generated_images < 10:  # 只保存前10个样本
                    vis_path = os.path.join(
                        vis_dir, class_name, f"{base_name}_comparison.png"
                    )
                    save_edge_comparison(rgb_array, result, vis_path)
                
                generated_images += 1
                
            except Exception as e:
                print(f"\n  ✗ 处理失败: {img_path}")
                print(f"    错误: {e}")
                failed_images += 1
    
    # 5. 统计信息
    print("\n" + "="*80)
    print("生成完成 - 统计信息".center(80))
    print("="*80)
    print(f"  总图像数: {total_images}")
    print(f"  已生成: {generated_images}")
    print(f"  跳过: {skipped_images}")
    print(f"  失败: {failed_images}")
    if missing_vlm_features > 0:
        print(f"  缺失VLM特征: {missing_vlm_features}")
    print(f"  输出目录: {output_dir}")
    print("="*80)
    
    return {
        'total': total_images,
        'generated': generated_images,
        'skipped': skipped_images,
        'failed': failed_images,
        'missing_vlm': missing_vlm_features,
        'output_dir': output_dir
    }


def save_edge_comparison(rgb_image: np.ndarray,
                        result_dict: dict,
                        save_path: str):
    """
    保存边缘对比可视化
    
    Args:
        rgb_image: 原始RGB图像
        result_dict: 边缘生成结果字典
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original RGB', fontsize=12)
    axes[0, 0].axis('off')
    
    # Sobel
    if result_dict.get('sobel_edges') is not None:
        axes[0, 1].imshow(result_dict['sobel_edges'], cmap='gray')
        axes[0, 1].set_title('Sobel (Dense)', fontsize=12)
        axes[0, 1].axis('off')
    
    # Canny
    if result_dict.get('canny_edges') is not None:
        axes[0, 2].imshow(result_dict['canny_edges'], cmap='gray')
        axes[0, 2].set_title('Canny (Sparse)', fontsize=12)
        axes[0, 2].axis('off')
    
    # Raw combined
    axes[1, 0].imshow(result_dict['raw_edges'], cmap='gray')
    axes[1, 0].set_title('Combined Edges', fontsize=12)
    axes[1, 0].axis('off')
    
    # VLM attention
    if result_dict.get('attention_map') is not None:
        axes[1, 1].imshow(result_dict['attention_map'], cmap='hot')
        axes[1, 1].set_title('VLM Attention', fontsize=12)
        axes[1, 1].axis('off')
    
    # Semantic edges
    axes[1, 2].imshow(result_dict['semantic_edges'], cmap='gray')
    axes[1, 2].set_title('Semantic Edges (Final)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='生成Caltech101的语义边缘',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据路径
    parser.add_argument('--rgb_root', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/caltech101/caltech-101',
                        help='RGB Caltech101数据集根目录')
    parser.add_argument('--vlm_features_dir', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/vlm_features',
                        help='VLM特征目录')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/semantic_edges',
                        help='语义边缘输出目录')
    
    # 边缘检测配置
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像大小')
    parser.add_argument('--sobel_weight', type=float, default=0.3,
                        help='Sobel权重（密集边缘）')
    parser.add_argument('--canny_weight', type=float, default=0.7,
                        help='Canny权重（稀疏边缘，匹配DVS）')
    
    # 其他选项
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已存在的边缘文件')
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false',
                        help='重新生成所有边缘')
    parser.add_argument('--save_visualization', action='store_true',
                        help='保存可视化结果（前10个样本）')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 生成语义边缘
    results = generate_caltech101_semantic_edges(
        rgb_root=args.rgb_root,
        vlm_features_dir=args.vlm_features_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        sobel_weight=args.sobel_weight,
        canny_weight=args.canny_weight,
        skip_existing=args.skip_existing,
        save_visualization=args.save_visualization
    )
    
    # 保存统计信息
    stats_path = os.path.join(args.output_dir, 'generation_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Caltech101 语义边缘生成统计\n")
        f.write("="*80 + "\n\n")
        f.write(f"RGB数据路径: {args.rgb_root}\n")
        f.write(f"VLM特征路径: {args.vlm_features_dir}\n")
        f.write(f"输出目录: {args.output_dir}\n\n")
        f.write(f"边缘检测配置:\n")
        f.write(f"  Sobel权重: {args.sobel_weight} (密集)\n")
        f.write(f"  Canny权重: {args.canny_weight} (稀疏)\n\n")
        f.write(f"总图像数: {results['total']}\n")
        f.write(f"已生成: {results['generated']}\n")
        f.write(f"跳过: {results['skipped']}\n")
        f.write(f"失败: {results['failed']}\n")
        if results['missing_vlm'] > 0:
            f.write(f"缺失VLM特征: {results['missing_vlm']}\n")
    
    print(f"\n✓ 统计信息已保存到: {stats_path}")


if __name__ == '__main__':
    main()

