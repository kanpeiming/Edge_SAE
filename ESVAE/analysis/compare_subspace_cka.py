"""
子空间CKA/TCKA对比脚本 (Subspace CKA/TCKA Comparison Script)

功能：
1. 从两个模型中提取时序特征
2. 使用CKA和TCKA计算特征子空间的相似度
3. 对比编码层和高层特征的差异

使用方法:
    python -m analysis.compare_subspace_cka

或者从项目根目录运行:
    python ESVAE/analysis/compare_subspace_cka.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ESVAE.models.snn_models.VGG import VGGSNN
from ESVAE.dataloader.caltech101 import create_caltech101_dataloaders
from ESVAE.analysis.utils import load_model_checkpoint, get_layer_output_with_mem
from ESVAE.tl_utils.loss_function import linear_CKA, temporal_linear_CKA


def extract_temporal_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    device: torch.device,
    max_samples: int = 2000
) -> torch.Tensor:
    """
    提取时序特征 (N, T, D)
    
    Args:
        model: 模型
        dataloader: 数据加载器
        layer_name: 层名称
        device: 设备
        max_samples: 最大样本数
    
    Returns:
        features: (N, T, D) 时序特征张量
    """
    model.eval()
    
    all_features = []
    sample_count = 0
    
    # 创建hook来捕获中间层输出
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    captured_output = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_output.append(output[0].detach())
        else:
            captured_output.append(output.detach())
    
    hook = target_module.register_forward_hook(hook_fn)
    
    print(f"Extracting temporal features from layer: {layer_name}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Processing batches")):
            if sample_count >= max_samples:
                break
            
            data = data.to(device)
            captured_output = []
            
            # 前向传播
            _ = model(data)
            
            if captured_output:
                features = captured_output[0]
                
                # 确保是时序格式 (N, T, ...)
                if len(features.shape) >= 3:
                    N, T = features.shape[:2]
                    # Flatten除了N和T之外的所有维度
                    features = features.reshape(N, T, -1)
                    all_features.append(features.cpu())
                    sample_count += len(data)
    
    hook.remove()
    
    # 合并所有batch
    all_features = torch.cat(all_features, dim=0)[:max_samples]
    
    print(f"✓ Extracted temporal features shape: {all_features.shape}")
    
    return all_features


def compare_subspace_cka(
    baseline_ckpt_path: str,
    pretrained_ckpt_path: str,
    data_path: str,
    output_dir: str = "results/cka_comparison",
    batch_size: int = 32,
    max_samples: int = 2000,
    img_size: int = 48,
    device: str = "cuda"
):
    """
    使用CKA/TCKA对比两个模型的特征子空间
    
    Args:
        baseline_ckpt_path: baseline模型checkpoint路径
        pretrained_ckpt_path: 预训练+微调模型checkpoint路径
        data_path: 数据集路径
        output_dir: 输出目录
        batch_size: 批次大小
        max_samples: 最大样本数
        img_size: 图像尺寸
        device: 设备
    """
    
    print("=" * 80)
    print("子空间CKA/TCKA对比分析 (Subspace CKA/TCKA Comparison)")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 加载数据
    print("Loading DVS dataset...")
    print(f"Dataset path: {data_path}")
    
    _, test_loader = create_caltech101_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        train_ratio=1.0,
        num_workers=4,
        img_size=img_size
    )
    
    print(f"✓ Dataset loaded\n")
    
    # 加载模型
    print("Loading models...")
    
    # Baseline模型
    print(f"  [1/2] Loading baseline model from: {baseline_ckpt_path}")
    model_baseline = VGGSNN(in_channel=2, cls_num=101, img_shape=img_size)
    model_baseline = load_model_checkpoint(model_baseline, baseline_ckpt_path, device)
    
    # Pretrained+Finetuned模型
    print(f"  [2/2] Loading pretrained model from: {pretrained_ckpt_path}")
    model_pretrained = VGGSNN(in_channel=2, cls_num=101, img_shape=img_size)
    model_pretrained = load_model_checkpoint(model_pretrained, pretrained_ckpt_path, device)
    
    print()
    
    # 定义要对比的层
    layers_to_compare = [
        ('dvs_input', 'DVS Input Layer (Encoder)'),
        ('bottleneck', 'Bottleneck Layer (High-level Features)')
    ]
    
    results = {}
    
    print("=" * 80)
    print("开始提取特征并计算CKA (Extracting Features and Computing CKA)")
    print("=" * 80)
    
    for layer_name, layer_description in layers_to_compare:
        print(f"\n{'=' * 80}")
        print(f"处理层: {layer_description} ({layer_name})")
        print(f"{'=' * 80}")
        
        try:
            # 从baseline模型提取特征
            print(f"\n[1/2] Extracting from baseline model...")
            features_baseline = extract_temporal_features(
                model=model_baseline,
                dataloader=test_loader,
                layer_name=layer_name,
                device=device,
                max_samples=max_samples
            )
            
            # 从pretrained模型提取特征
            print(f"\n[2/2] Extracting from pretrained model...")
            features_pretrained = extract_temporal_features(
                model=model_pretrained,
                dataloader=test_loader,
                layer_name=layer_name,
                device=device,
                max_samples=max_samples
            )
            
            # 确保特征在同一设备上
            features_baseline = features_baseline.to(device)
            features_pretrained = features_pretrained.to(device)
            
            print(f"\n计算CKA指标 (Computing CKA metrics)...")
            
            # 计算Temporal Linear CKA (时序CKA)
            print("  Computing Temporal Linear CKA...")
            tcka_score = temporal_linear_CKA(features_baseline, features_pretrained, debiased=False)
            tcka_score = tcka_score.item() if isinstance(tcka_score, torch.Tensor) else tcka_score
            
            # 计算Linear CKA (对时间维度求和后的CKA)
            print("  Computing Linear CKA (time-averaged)...")
            cka_sum_score = linear_CKA(features_baseline, features_pretrained, ftype="SUM", debiased=False)
            cka_sum_score = cka_sum_score.item() if isinstance(cka_sum_score, torch.Tensor) else cka_sum_score
            
            # 计算Linear CKA (对时间维度flatten后的CKA)
            print("  Computing Linear CKA (time-flattened)...")
            cka_flatten_score = linear_CKA(features_baseline, features_pretrained, ftype="FLATTEN", debiased=False)
            cka_flatten_score = cka_flatten_score.item() if isinstance(cka_flatten_score, torch.Tensor) else cka_flatten_score
            
            # 保存结果
            results[layer_name] = {
                'description': layer_description,
                'temporal_cka': tcka_score,
                'linear_cka_sum': cka_sum_score,
                'linear_cka_flatten': cka_flatten_score,
                'feature_shape': list(features_baseline.shape)
            }
            
            print(f"\n✓ Results for {layer_description}:")
            print(f"    Temporal Linear CKA: {tcka_score:.6f}")
            print(f"    Linear CKA (SUM):    {cka_sum_score:.6f}")
            print(f"    Linear CKA (FLATTEN): {cka_flatten_score:.6f}")
            
        except Exception as e:
            print(f"\n✗ Error processing layer {layer_name}: {e}")
            results[layer_name] = {
                'description': layer_description,
                'error': str(e)
            }
    
    # 打印最终结果摘要
    print("\n" + "=" * 80)
    print("CKA对比结果摘要 (CKA Comparison Summary)")
    print("=" * 80)
    print(f"\nDataset: N-Caltech101 (DVS)")
    print(f"Number of samples: {max_samples}")
    print(f"Models compared:")
    print(f"  - Baseline: DVS-only trained model")
    print(f"  - Pretrained: RGB→Edge pretrained + DVS finetuned model")
    print("\n" + "-" * 80)
    
    for layer_name, result in results.items():
        if 'error' not in result:
            print(f"\n{result['description']} ({layer_name}):")
            print(f"  Feature shape: {result['feature_shape']}")
            print(f"  Temporal Linear CKA:  {result['temporal_cka']:.6f}")
            print(f"  Linear CKA (SUM):     {result['linear_cka_sum']:.6f}")
            print(f"  Linear CKA (FLATTEN): {result['linear_cka_flatten']:.6f}")
        else:
            print(f"\n{result['description']} ({layer_name}):")
            print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 80)
    
    # 保存结果到文件
    result_file = os.path.join(output_dir, 'cka_comparison_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CKA/TCKA Subspace Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: N-Caltech101 (DVS)\n")
        f.write(f"Number of samples: {max_samples}\n")
        f.write(f"Models compared:\n")
        f.write(f"  - Baseline: DVS-only trained model\n")
        f.write(f"  - Pretrained: RGB→Edge pretrained + DVS finetuned model\n\n")
        f.write("-" * 80 + "\n\n")
        
        for layer_name, result in results.items():
            if 'error' not in result:
                f.write(f"{result['description']} ({layer_name}):\n")
                f.write(f"  Feature shape: {result['feature_shape']}\n")
                f.write(f"  Temporal Linear CKA:  {result['temporal_cka']:.6f}\n")
                f.write(f"  Linear CKA (SUM):     {result['linear_cka_sum']:.6f}\n")
                f.write(f"  Linear CKA (FLATTEN): {result['linear_cka_flatten']:.6f}\n\n")
            else:
                f.write(f"{result['description']} ({layer_name}):\n")
                f.write(f"  Error: {result['error']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("\nInterpretation:\n")
        f.write("- CKA值范围: [0, 1]\n")
        f.write("- CKA = 1: 两个特征子空间完全相同\n")
        f.write("- CKA = 0: 两个特征子空间完全不相关\n")
        f.write("- 较高的CKA值表明预训练保留了相似的特征表示\n")
        f.write("- 较低的CKA值表明预训练显著改变了特征空间结构\n")
    
    print(f"\n✓ Results saved to: {result_file}")
    print("\n" + "=" * 80)
    print("分析完成！(Analysis Complete!)")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare feature subspaces using CKA/TCKA')
    parser.add_argument('--baseline_ckpt', type=str, required=False,
                        default='path/to/baseline_dvs_model.pth',
                        help='Path to baseline DVS-only model checkpoint')
    parser.add_argument('--pretrained_ckpt', type=str, required=False,
                        default='path/to/pretrain_then_dvs_model.pth',
                        help='Path to pretrained+finetuned model checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101',
                        help='Path to N-Caltech101 DVS dataset')
    parser.add_argument('--output_dir', type=str, default='results/cka_comparison',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Maximum number of samples to use')
    parser.add_argument('--img_size', type=int, default=48,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 提示用户修改路径
    if args.baseline_ckpt == 'path/to/baseline_dvs_model.pth':
        print("\n" + "!" * 80)
        print("请修改checkpoint路径！(Please modify checkpoint paths!)")
        print("!" * 80)
        print("\n使用方法 (Usage):")
        print("  python -m analysis.compare_subspace_cka \\")
        print("      --baseline_ckpt /path/to/your/baseline_model.pth \\")
        print("      --pretrained_ckpt /path/to/your/pretrained_model.pth \\")
        print("      --data_path /path/to/n-caltech101 \\")
        print("      --output_dir results/cka_comparison \\")
        print("      --max_samples 2000\n")
        
        print("或者直接在脚本中修改默认路径 (Or modify default paths in the script)")
        print("=" * 80 + "\n")
        return
    
    # 运行CKA对比分析
    compare_subspace_cka(
        baseline_ckpt_path=args.baseline_ckpt,
        pretrained_ckpt_path=args.pretrained_ckpt,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        img_size=args.img_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

