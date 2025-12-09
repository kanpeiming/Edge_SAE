"""
特征提取与t-SNE可视化脚本 (Feature Extraction and t-SNE Visualization Script)

功能：
1. 从两个模型（baseline和pretrained+finetuned）中提取DVS特征
2. 使用t-SNE降维到2D
3. 可视化特征分布，对比两个模型的特征空间结构

使用方法:
    python -m ESVAE.analysis.extract_features_tsne --layer_name classifier
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ESVAE.models.snn_models.VGG import VGGSNN
from ESVAE.dataloader.caltech101 import create_caltech101_dataloaders


def load_checkpoint(model, ckpt_path, device):
    """加载模型checkpoint"""
    print(f"Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 提取state_dict
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除'module.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("✓ Loaded successfully\n")
    return model


def extract_features(model, dataloader, layer_name, device, max_samples=1000):
    """
    提取指定层的特征
    
    Args:
        model: 模型
        dataloader: 数据加载器
        layer_name: 层名称 ('bottleneck', 'classifier', 'features.9')
        device: 设备
        max_samples: 最大样本数
    
    Returns:
        features: (N, D) 特征数组
        labels: (N,) 标签数组
    """
    model.eval()
    
    # 注册hook
    features_dict = {}
    
    def hook_fn(module, input, output):
        # 处理不同的输出格式
        if isinstance(output, tuple):
            feat = output[0].detach()
        else:
            feat = output.detach()
        
        # 处理时序特征: (N, T, D) -> (N, D)
        if len(feat.shape) == 3:
            feat = feat.mean(dim=1)  # 对时间维度求平均
        elif len(feat.shape) == 2:
            pass  # 已经是 (N, D)
        else:
            # 展平其他维度
            feat = feat.reshape(feat.size(0), -1)
        
        features_dict['output'] = feat
    
    # 找到目标层并注册hook
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        print(f"\n可用的层名称:")
        for name, _ in model.named_modules():
            if name:
                print(f"  - {name}")
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    hook = target_module.register_forward_hook(hook_fn)
    
    print(f"Extracting features from layer: {layer_name}")
    print(f"Maximum samples: {max_samples}")
    
    all_features = []
    all_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
            
            data = data.to(device)
            
            # 前向传播
            _ = model(data)
            
            # 获取特征
            if 'output' in features_dict:
                feat = features_dict['output'].cpu().numpy()
                all_features.append(feat)
                all_labels.append(target.cpu().numpy())
                sample_count += len(data)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {sample_count}/{max_samples} samples...")
    
    hook.remove()
    
    # 合并所有batch
    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"✓ Labels shape: {labels.shape}\n")
    
    return features, labels


def compute_metrics(features_2d, labels):
    """计算聚类质量指标"""
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
    
    silhouette = silhouette_score(features_2d, labels)
    davies_bouldin = davies_bouldin_score(features_2d, labels)
    calinski_harabasz = calinski_harabasz_score(features_2d, labels)
    
    # 计算类内平均距离
    unique_labels = np.unique(labels)
    intra_dists = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 1:
            class_features = features_2d[mask]
            centroid = class_features.mean(axis=0)
            dists = np.linalg.norm(class_features - centroid, axis=1)
            intra_dists.append(dists.mean())
    
    avg_intra_dist = np.mean(intra_dists) if intra_dists else 0
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'avg_intra_dist': avg_intra_dist
    }


def visualize_comparison(features_b_2d, features_p_2d, labels, 
                        metrics_b, metrics_p, output_dir, num_display=10):
    """创建对比可视化"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择样本数最多的类别
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_indices = np.argsort(counts)[-num_display:]
    selected_labels = unique_labels[top_indices]
    
    # 使用更鲜艳的颜色方案
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
    
    # ========================================================================
    # 图1：并排对比图（带指标）
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Baseline
    ax = axes[0]
    for i, label in enumerate(selected_labels):
        mask = labels == label
        if mask.sum() > 0:
            ax.scatter(features_b_2d[mask, 0], features_b_2d[mask, 1],
                      c=colors[i], label=f'Class {label} (n={mask.sum()})',
                      alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            # 质心
            centroid = features_b_2d[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=colors[i], 
                      marker='*', s=600, edgecolors='black', linewidths=2.5, zorder=10)
    
    metrics_text = (f"Silhouette: {metrics_b['silhouette']:.3f}\n"
                   f"DB Index: {metrics_b['davies_bouldin']:.3f}\n"
                   f"Intra-dist: {metrics_b['avg_intra_dist']:.2f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.set_title('Baseline Model (DVS-only)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_facecolor('#f8f9fa')
    
    # Pretrained
    ax = axes[1]
    for i, label in enumerate(selected_labels):
        mask = labels == label
        if mask.sum() > 0:
            ax.scatter(features_p_2d[mask, 0], features_p_2d[mask, 1],
                      c=colors[i], label=f'Class {label} (n={mask.sum()})',
                      alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            # 质心
            centroid = features_p_2d[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=colors[i], 
                      marker='*', s=600, edgecolors='black', linewidths=2.5, zorder=10)
    
    metrics_text = (f"Silhouette: {metrics_p['silhouette']:.3f}\n"
                   f"DB Index: {metrics_p['davies_bouldin']:.3f}\n"
                   f"Intra-dist: {metrics_p['avg_intra_dist']:.2f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.set_title('Pretrained + Finetuned Model', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'tsne_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # 图2：单独的Baseline图（大图）
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, label in enumerate(selected_labels):
        mask = labels == label
        if mask.sum() > 0:
            ax.scatter(features_b_2d[mask, 0], features_b_2d[mask, 1],
                      c=colors[i], label=f'Class {label} (n={mask.sum()})',
                      alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            centroid = features_b_2d[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=colors[i], 
                      marker='*', s=600, edgecolors='black', linewidths=2.5, zorder=10)
    
    metrics_text = (f"Silhouette Score: {metrics_b['silhouette']:.4f}\n"
                   f"Davies-Bouldin Index: {metrics_b['davies_bouldin']:.4f}\n"
                   f"Avg Intra-class Distance: {metrics_b['avg_intra_dist']:.2f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.set_title('Baseline Model (DVS-only)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'tsne_baseline.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # 图3：单独的Pretrained图（大图）
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, label in enumerate(selected_labels):
        mask = labels == label
        if mask.sum() > 0:
            ax.scatter(features_p_2d[mask, 0], features_p_2d[mask, 1],
                      c=colors[i], label=f'Class {label} (n={mask.sum()})',
                      alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            centroid = features_p_2d[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=colors[i], 
                      marker='*', s=600, edgecolors='black', linewidths=2.5, zorder=10)
    
    metrics_text = (f"Silhouette Score: {metrics_p['silhouette']:.4f}\n"
                   f"Davies-Bouldin Index: {metrics_p['davies_bouldin']:.4f}\n"
                   f"Avg Intra-class Distance: {metrics_p['avg_intra_dist']:.2f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.set_title('Pretrained + Finetuned Model', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'tsne_pretrained.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def visualize_3d(features_b, features_p, labels, output_dir, num_display=10):
    """创建3D t-SNE可视化"""
    
    print("\nCreating 3D visualizations...")
    
    # 选择样本数最多的类别
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_indices = np.argsort(counts)[-num_display:]
    selected_labels = unique_labels[top_indices]
    
    # 使用更鲜艳的颜色方案
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
    
    # 3D t-SNE降维
    print("Running 3D t-SNE on baseline features...")
    tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200,
                   init='pca', random_state=42, verbose=0)
    features_b_3d = tsne_3d.fit_transform(features_b)
    
    print("Running 3D t-SNE on pretrained features...")
    tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200,
                   init='pca', random_state=42, verbose=0)
    features_p_3d = tsne_3d.fit_transform(features_p)
    
    # 创建3D对比图 - 侧面视角优化版本
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(28, 12))
    fig.patch.set_facecolor('white')
    
    # Baseline 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('white')
    
    for i, label in enumerate(selected_labels):
        mask = labels == label
        if mask.sum() > 0:
            scatter1 = ax1.scatter(features_b_3d[mask, 0], 
                                  features_b_3d[mask, 1], 
                                  features_b_3d[mask, 2],
                                  c=colors[i], label=f'Class {label}',
                                  alpha=0.7, s=50, edgecolors='none',
                                  depthshade=True)
    
    ax1.set_title('Baseline Model (DVS-only) - 3D', fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel('t-SNE Dim 1', fontsize=12, labelpad=8)
    ax1.set_ylabel('t-SNE Dim 2', fontsize=12, labelpad=8)
    ax1.set_zlabel('t-SNE Dim 3', fontsize=12, labelpad=8)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
    
    # 关键：设置侧面视角（类似你的图）
    ax1.view_init(elev=15, azim=-60)
    
    # 设置网格样式
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    
    # 设置坐标轴背景面板为浅色
    ax1.xaxis.pane.fill = True
    ax1.yaxis.pane.fill = True
    ax1.zaxis.pane.fill = True
    ax1.xaxis.pane.set_facecolor('#f0f0f0')
    ax1.yaxis.pane.set_facecolor('#f0f0f0')
    ax1.zaxis.pane.set_facecolor('#f0f0f0')
    ax1.xaxis.pane.set_alpha(0.3)
    ax1.yaxis.pane.set_alpha(0.3)
    ax1.zaxis.pane.set_alpha(0.3)
    
    # 设置坐标轴边框颜色
    ax1.xaxis.pane.set_edgecolor('#cccccc')
    ax1.yaxis.pane.set_edgecolor('#cccccc')
    ax1.zaxis.pane.set_edgecolor('#cccccc')
    
    # Pretrained 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('white')
    
    for i, label in enumerate(selected_labels):
        mask = labels == label
        if mask.sum() > 0:
            scatter2 = ax2.scatter(features_p_3d[mask, 0], 
                                  features_p_3d[mask, 1], 
                                  features_p_3d[mask, 2],
                                  c=colors[i], label=f'Class {label}',
                                  alpha=0.7, s=50, edgecolors='none',
                                  depthshade=True)
    
    ax2.set_title('Pretrained + Finetuned Model - 3D', fontsize=20, fontweight='bold', pad=20)
    ax2.set_xlabel('t-SNE Dim 1', fontsize=12, labelpad=8)
    ax2.set_ylabel('t-SNE Dim 2', fontsize=12, labelpad=8)
    ax2.set_zlabel('t-SNE Dim 3', fontsize=12, labelpad=8)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
    
    # 关键：设置侧面视角（类似你的图）
    ax2.view_init(elev=15, azim=-60)
    
    # 设置网格样式
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    
    # 设置坐标轴背景面板为浅色
    ax2.xaxis.pane.fill = True
    ax2.yaxis.pane.fill = True
    ax2.zaxis.pane.fill = True
    ax2.xaxis.pane.set_facecolor('#f0f0f0')
    ax2.yaxis.pane.set_facecolor('#f0f0f0')
    ax2.zaxis.pane.set_facecolor('#f0f0f0')
    ax2.xaxis.pane.set_alpha(0.3)
    ax2.yaxis.pane.set_alpha(0.3)
    ax2.zaxis.pane.set_alpha(0.3)
    
    # 设置坐标轴边框颜色
    ax2.xaxis.pane.set_edgecolor('#cccccc')
    ax2.yaxis.pane.set_edgecolor('#cccccc')
    ax2.zaxis.pane.set_edgecolor('#cccccc')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'tsne_3d_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved 3D comparison: {save_path}")
    plt.close()
    
    print("✓ 3D visualization complete!")


def main():
    parser = argparse.ArgumentParser(description='Feature extraction and t-SNE visualization')
    parser.add_argument('--baseline_ckpt', type=str,
                       default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints/NCaltech101_baseline_lr0.001_T10_bs32_seed1000.pth')
    parser.add_argument('--pretrained_ckpt', type=str,
                       default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints/Fine-tuning_NCaltech101_baseline_lr0.001_T10_bs32_seed1000.pth')
    parser.add_argument('--data_path', type=str,
                       default='/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101')
    parser.add_argument('--output_dir', type=str,
                       default='/home/user/kpm/kpm/results/analysis/tsne_visualization')
    parser.add_argument('--layer_name', type=str, default='features.9',
                       help='Layer to extract features from (bottleneck, classifier, features.9)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=48)
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity (try 30-100)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_display', type=int, default=10,
                       help='Number of classes to display')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("特征提取与t-SNE可视化 (Feature Extraction and t-SNE Visualization)")
    print("=" * 80)
    print(f"Layer: {args.layer_name}")
    print(f"Perplexity: {args.perplexity}")
    print(f"Max samples: {args.max_samples}\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # 加载数据
    print("Loading dataset...")
    _, test_loader = create_caltech101_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        train_ratio=1.0,
        num_workers=4,
        img_size=args.img_size
    )
    print("✓ Dataset loaded\n")
    
    # 加载模型
    print("=" * 80)
    print("Loading models...")
    print("=" * 80)
    
    model_baseline = VGGSNN(in_channel=2, cls_num=101, img_shape=args.img_size)
    model_baseline = load_checkpoint(model_baseline, args.baseline_ckpt, device)
    
    model_pretrained = VGGSNN(in_channel=2, cls_num=101, img_shape=args.img_size)
    model_pretrained = load_checkpoint(model_pretrained, args.pretrained_ckpt, device)
    
    # 提取特征
    print("=" * 80)
    print("Extracting features...")
    print("=" * 80)
    
    print("[1/2] Baseline model:")
    features_baseline, labels = extract_features(
        model_baseline, test_loader, args.layer_name, device, args.max_samples
    )
    
    print("[2/2] Pretrained model:")
    features_pretrained, _ = extract_features(
        model_pretrained, test_loader, args.layer_name, device, args.max_samples
    )
    
    # 特征归一化（重要！）
    print("Normalizing features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_baseline = scaler.fit_transform(features_baseline)
    features_pretrained = scaler.fit_transform(features_pretrained)
    print("✓ Features normalized\n")
    
    # t-SNE降维
    print("=" * 80)
    print("Running t-SNE...")
    print("=" * 80)
    
    print("[1/2] Baseline features...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=200,
                init='pca', random_state=42, verbose=1)
    features_baseline_2d = tsne.fit_transform(features_baseline)
    print("✓ Baseline t-SNE completed\n")
    
    print("[2/2] Pretrained features...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=200,
                init='pca', random_state=42, verbose=1)
    features_pretrained_2d = tsne.fit_transform(features_pretrained)
    print("✓ Pretrained t-SNE completed\n")
    
    # 计算指标
    print("=" * 80)
    print("Computing metrics...")
    print("=" * 80)
    
    metrics_baseline = compute_metrics(features_baseline_2d, labels)
    metrics_pretrained = compute_metrics(features_pretrained_2d, labels)
    
    print(f"\nBaseline:")
    print(f"  Silhouette: {metrics_baseline['silhouette']:.4f}")
    print(f"  DB Index: {metrics_baseline['davies_bouldin']:.4f}")
    print(f"  Intra-dist: {metrics_baseline['avg_intra_dist']:.4f}")
    
    print(f"\nPretrained:")
    print(f"  Silhouette: {metrics_pretrained['silhouette']:.4f}")
    print(f"  DB Index: {metrics_pretrained['davies_bouldin']:.4f}")
    print(f"  Intra-dist: {metrics_pretrained['avg_intra_dist']:.4f}\n")
    
    # 保存指标报告
    report_path = os.path.join(args.output_dir, 'metrics_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("聚类质量对比报告 (Clustering Quality Comparison Report)\n")
        f.write(f"提取层: {args.layer_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Metric':<30} {'Baseline':<20} {'Pretrained':<20} {'Change':<15}\n")
        f.write("-" * 80 + "\n")
        
        sil_change = (metrics_pretrained['silhouette'] - metrics_baseline['silhouette']) * 100
        db_change = (metrics_baseline['davies_bouldin'] - metrics_pretrained['davies_bouldin']) / metrics_baseline['davies_bouldin'] * 100
        ch_change = (metrics_pretrained['calinski_harabasz'] - metrics_baseline['calinski_harabasz']) / metrics_baseline['calinski_harabasz'] * 100
        intra_change = (metrics_baseline['avg_intra_dist'] - metrics_pretrained['avg_intra_dist']) / metrics_baseline['avg_intra_dist'] * 100
        
        f.write(f"{'Silhouette Score (↑)':<30} {metrics_baseline['silhouette']:>19.4f} {metrics_pretrained['silhouette']:>19.4f} {sil_change:>13.2f}%\n")
        f.write(f"{'Davies-Bouldin Index (↓)':<30} {metrics_baseline['davies_bouldin']:>19.4f} {metrics_pretrained['davies_bouldin']:>19.4f} {db_change:>13.2f}%\n")
        f.write(f"{'Calinski-Harabasz Index (↑)':<30} {metrics_baseline['calinski_harabasz']:>19.2f} {metrics_pretrained['calinski_harabasz']:>19.2f} {ch_change:>13.2f}%\n")
        f.write(f"{'Avg Intra-class Dist (↓)':<30} {metrics_baseline['avg_intra_dist']:>19.4f} {metrics_pretrained['avg_intra_dist']:>19.4f} {intra_change:>13.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("结论分析:\n")
        f.write("-" * 80 + "\n")
        
        if sil_change > 0:
            f.write("✓ Silhouette Score提升，表明预训练后类别分离度更好\n")
        else:
            f.write("✗ Silhouette Score下降，可能是t-SNE投影失真\n")
        
        if db_change > 0:
            f.write("✓ Davies-Bouldin Index降低，表明类内更紧凑、类间分离更好\n")
        else:
            f.write("✗ Davies-Bouldin Index上升，表明聚类质量在2D投影中下降\n")
        
        if intra_change > 0:
            f.write("✓ 类内距离减小，表明预训练后同类样本更紧凑\n")
        else:
            f.write("✗ 类内距离增大，可能是特征空间扩展\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("指标说明:\n")
        f.write("- Silhouette Score: 轮廓系数，[-1,1]，越接近1越好\n")
        f.write("- Davies-Bouldin Index: DB指数，越小越好（类内紧凑、类间分离）\n")
        f.write("- Calinski-Harabasz Index: CH指数，越大越好\n")
        f.write("- Avg Intra-class Distance: 类内平均距离，越小表示类内越紧凑\n")
        f.write("\n重要提示:\n")
        f.write("- t-SNE是非线性降维，2D/3D投影可能无法完全反映高维空间的真实结构\n")
        f.write("- bottleneck层: 压缩特征，可能更能反映模型学到的紧凑表示\n")
        f.write("- classifier层: 分类器输入，更直接反映分类性能\n")
        f.write("- features.9层: 最后卷积层，包含更多空间细节\n")
    
    print(f"✓ Saved metrics report: {report_path}\n")
    
    # 可视化
    print("=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    
    visualize_comparison(
        features_baseline_2d, features_pretrained_2d, labels,
        metrics_baseline, metrics_pretrained,
        args.output_dir, args.num_display
    )
    
    # 创建3D可视化
    visualize_3d(
        features_baseline, features_pretrained, labels,
        args.output_dir, args.num_display
    )
    
    print("\n" + "=" * 80)
    print("完成！(Complete!)")
    print("=" * 80)
    print(f"\n建议:")
    print(f"  1. 如果结果与分类准确率不一致，尝试其他层:")
    print(f"     --layer_name classifier  (分类器前的特征)")
    print(f"     --layer_name features.9  (最后一个卷积层)")
    print(f"  2. 调整perplexity参数: --perplexity 50 或 --perplexity 100")
    print(f"  3. 增加样本数: --max_samples 2000")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
