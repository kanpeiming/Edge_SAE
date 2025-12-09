"""
模型参数对比脚本 (Model Parameter Comparison Script)

功能：
1. 加载两个模型的checkpoint
2. 逐层对比参数差异（L2范数、余弦相似度）
3. 生成对比报告并保存为CSV文件

使用方法:
    python -m analysis.compare_parameters

或者从项目根目录运行:
    python ESVAE/analysis/compare_parameters.py
"""

import os
import sys
import torch
import pandas as pd
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ESVAE.models.snn_models.VGG import VGGSNN
from ESVAE.analysis.utils import compute_parameter_difference


def compare_model_parameters(
    baseline_ckpt_path: str,
    finetuned_ckpt_path: str,
    output_dir: str = "results/parameter_comparison",
    device: str = "cuda"
):
    """
    对比两个模型的参数差异
    
    Args:
        baseline_ckpt_path: baseline模型checkpoint路径
        finetuned_ckpt_path: 预训练+微调模型checkpoint路径
        output_dir: 输出目录
        device: 设备 (cuda/cpu)
    """
    
    print("=" * 80)
    print("模型参数对比分析 (Model Parameter Comparison)")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 加载checkpoint
    print("Loading checkpoints...")
    print(f"  Baseline: {baseline_ckpt_path}")
    
    if not os.path.exists(baseline_ckpt_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt_path}")
    
    checkpoint_a = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    
    print(f"  Finetuned: {finetuned_ckpt_path}")
    
    if not os.path.exists(finetuned_ckpt_path):
        raise FileNotFoundError(f"Finetuned checkpoint not found: {finetuned_ckpt_path}")
    
    checkpoint_b = torch.load(finetuned_ckpt_path, map_location=device, weights_only=False)
    
    # 提取state_dict
    if 'model_state_dict' in checkpoint_a:
        state_dict_a = checkpoint_a['model_state_dict']
    elif 'state_dict' in checkpoint_a:
        state_dict_a = checkpoint_a['state_dict']
    else:
        state_dict_a = checkpoint_a
    
    if 'model_state_dict' in checkpoint_b:
        state_dict_b = checkpoint_b['model_state_dict']
    elif 'state_dict' in checkpoint_b:
        state_dict_b = checkpoint_b['state_dict']
    else:
        state_dict_b = checkpoint_b
    
    # 处理DataParallel保存的模型
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    state_dict_a = remove_module_prefix(state_dict_a)
    state_dict_b = remove_module_prefix(state_dict_b)
    
    print(f"✓ Loaded {len(state_dict_a)} parameters from baseline model")
    print(f"✓ Loaded {len(state_dict_b)} parameters from finetuned model\n")
    
    # 计算参数差异
    print("Computing parameter differences...")
    differences = compute_parameter_difference(state_dict_a, state_dict_b)
    
    # 整理结果
    results = []
    for layer_name, diff_info in differences.items():
        results.append({
            'Layer': layer_name,
            'L2_Difference': diff_info['l2_diff'],
            'Cosine_Similarity': diff_info['cosine_sim'],
            'Shape': str(diff_info['shape']),
            'Num_Params': diff_info['num_params']
        })
    
    # 创建DataFrame
    df = pd.DataFrame(results)

    # 当没有可比较的参数时给出友好提示并提前返回
    if df.empty or 'L2_Difference' not in df.columns:
        print("警告: 未找到可比较的公共张量参数，无法计算参数差异。")
        print("Warning: No common tensor parameters found between the two checkpoints; nothing to compare.")
        csv_path = os.path.join(output_dir, 'parameter_comparison_empty.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Empty results saved to: {csv_path}\n")
        return df
    
    # 按L2差异排序
    df = df.sort_values('L2_Difference', ascending=False)
    
    # 保存完整结果
    csv_path = os.path.join(output_dir, 'parameter_comparison_full.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Full results saved to: {csv_path}\n")
    
    # 打印统计信息
    print("=" * 80)
    print("统计摘要 (Statistical Summary)")
    print("=" * 80)
    print(f"Total parameters compared: {len(df)}")
    print(f"Average L2 difference: {df['L2_Difference'].mean():.6f}")
    print(f"Average cosine similarity: {df['Cosine_Similarity'].mean():.6f}")
    print(f"Min cosine similarity: {df['Cosine_Similarity'].min():.6f}")
    print(f"Max cosine similarity: {df['Cosine_Similarity'].max():.6f}\n")
    
    # 打印Top 10差异最大的层
    print("=" * 80)
    print("Top 10 最大差异层 (Top 10 Layers with Largest Differences)")
    print("=" * 80)
    print(df.head(10).to_string(index=False))
    print()
    
    # 筛选并保存关键层的差异
    print("=" * 80)
    print("关键层分析 (Key Layer Analysis)")
    print("=" * 80)
    
    # dvs_input相关层
    dvs_input_layers = df[df['Layer'].str.contains('dvs_input', case=False, na=False)]
    if not dvs_input_layers.empty:
        print("\n[DVS Input Layers]")
        print(dvs_input_layers.to_string(index=False))
        dvs_csv_path = os.path.join(output_dir, 'parameter_comparison_dvs_input.csv')
        dvs_input_layers.to_csv(dvs_csv_path, index=False)
        print(f"✓ Saved to: {dvs_csv_path}")
    
    # features相关层
    features_layers = df[df['Layer'].str.contains('features', case=False, na=False)]
    if not features_layers.empty:
        print("\n[Feature Layers]")
        print(features_layers.to_string(index=False))
        features_csv_path = os.path.join(output_dir, 'parameter_comparison_features.csv')
        features_layers.to_csv(features_csv_path, index=False)
        print(f"✓ Saved to: {features_csv_path}")
    
    # bottleneck相关层
    bottleneck_layers = df[df['Layer'].str.contains('bottleneck', case=False, na=False)]
    if not bottleneck_layers.empty:
        print("\n[Bottleneck Layers]")
        print(bottleneck_layers.to_string(index=False))
        bottleneck_csv_path = os.path.join(output_dir, 'parameter_comparison_bottleneck.csv')
        bottleneck_layers.to_csv(bottleneck_csv_path, index=False)
        print(f"✓ Saved to: {bottleneck_csv_path}")
    
    # classifier相关层
    classifier_layers = df[df['Layer'].str.contains('classifier', case=False, na=False)]
    if not classifier_layers.empty:
        print("\n[Classifier Layers]")
        print(classifier_layers.to_string(index=False))
        classifier_csv_path = os.path.join(output_dir, 'parameter_comparison_classifier.csv')
        classifier_layers.to_csv(classifier_csv_path, index=False)
        print(f"✓ Saved to: {classifier_csv_path}")
    
    print("\n" + "=" * 80)
    print("分析完成！(Analysis Complete!)")
    print("=" * 80)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compare parameters between two models')
    parser.add_argument('--baseline_ckpt', type=str, required=False,
                        default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints/NCaltech101_baseline_lr0.001_T10_bs32_seed1000.pth',
                        help='Path to baseline DVS-only model checkpoint')
    parser.add_argument('--finetuned_ckpt', type=str, required=False,
                        default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints/Fine-tuning_NCaltech101_baseline_lr0.001_T10_bs32_seed1000.pth',
                        help='Path to pretrained+finetuned model checkpoint')
    parser.add_argument('--output_dir', type=str, default='/home/user/kpm/kpm/results/analysis/compare_parameters/',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 提示用户修改路径
    if args.baseline_ckpt == 'path/to/baseline_dvs_model.pth':
        print("\n" + "!" * 80)
        print("请修改checkpoint路径！(Please modify checkpoint paths!)")
        print("!" * 80)
        print("\n使用方法 (Usage):")
        print("  python -m analysis.compare_parameters \\")
        print("      --baseline_ckpt /path/to/your/baseline_model.pth \\")
        print("      --finetuned_ckpt /path/to/your/finetuned_model.pth \\")
        print("      --output_dir results/parameter_comparison\n")
        
        print("或者直接在脚本中修改默认路径 (Or modify default paths in the script)")
        print("=" * 80 + "\n")
        return
    
    # 运行对比分析
    compare_model_parameters(
        baseline_ckpt_path=args.baseline_ckpt,
        finetuned_ckpt_path=args.finetuned_ckpt,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()

