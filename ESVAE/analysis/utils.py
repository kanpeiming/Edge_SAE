"""
共享工具函数 (Shared Utility Functions)
提供模型加载、特征提取等通用功能
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    加载模型checkpoint
    
    Args:
        model: 模型实例
        checkpoint_path: checkpoint文件路径
        device: 设备 (cuda/cpu)
    
    Returns:
        加载了参数的模型
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'net' in checkpoint:
        # 处理包含'net'键的checkpoint格式
        state_dict = checkpoint['net']
    else:
        # 检查是否是直接的state_dict（包含模型参数）
        # 通过检查是否有常见的非参数键来判断
        non_param_keys = {'T', 'batch_size', 'epoch', 'lr', 'seed', 'device', 
                         'checkpoint', 'data_set', 'log_dir', 'parallel', 
                         'pretrained_path', 'weight_decay', 'size', 'id',
                         'dvs_sample_ratio', 'caltech101_dvs_path', 'best_test_acc'}
        
        # 如果checkpoint中有这些非参数键，说明不是直接的state_dict
        if any(key in checkpoint for key in non_param_keys):
            raise ValueError(
                f"Checkpoint format not recognized. Available keys: {list(checkpoint.keys())}\n"
                f"Expected keys: 'model_state_dict', 'state_dict', or 'net'"
            )
        else:
            state_dict = checkpoint
    
    # 处理DataParallel保存的模型
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # 移除'module.'前缀
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


class FeatureExtractor:
    """
    特征提取器 - 使用hook机制从指定层提取特征
    """
    
    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        初始化特征提取器
        
        Args:
            model: 要提取特征的模型
            layer_names: 要提取特征的层名称列表
        """
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        
        # 注册hook
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)
                print(f"✓ Registered hook for layer: {name}")
    
    def _get_hook(self, name: str):
        """创建hook函数"""
        def hook(module, input, output):
            # 保存输出特征
            if isinstance(output, tuple):
                # 如果返回多个值，保存第一个
                self.features[name] = output[0].detach()
            else:
                self.features[name] = output.detach()
        return hook
    
    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取特征
        
        Args:
            x: 输入数据
        
        Returns:
            特征字典 {layer_name: features}
        """
        self.features = {}
        with torch.no_grad():
            _ = self.model(x)
        return self.features
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def extract_features_from_dataloader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    device: torch.device,
    max_samples: int = 2000,
    average_time: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从dataloader中提取指定层的特征
    
    Args:
        model: 模型
        dataloader: 数据加载器
        layer_name: 要提取的层名称
        device: 设备
        max_samples: 最大样本数
        average_time: 是否对时间维度求平均 (针对SNN的时序特征)
    
    Returns:
        features: 特征数组 (N, D)
        labels: 标签数组 (N,)
    """
    model.eval()
    extractor = FeatureExtractor(model, [layer_name])
    
    all_features = []
    all_labels = []
    sample_count = 0
    
    print(f"Extracting features from layer: {layer_name}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
            
            data = data.to(device)
            
            # 提取特征
            features_dict = extractor.extract(data)
            features = features_dict[layer_name]
            
            # 处理时序特征 (N, T, D) -> (N, D)
            if len(features.shape) == 3 and average_time:
                features = features.mean(dim=1)  # 对时间维度求平均
            elif len(features.shape) > 3:
                # 如果是 (N, T, C, H, W) 格式，先flatten空间维度
                N, T = features.shape[:2]
                features = features.reshape(N, T, -1)
                if average_time:
                    features = features.mean(dim=1)  # (N, T, D) -> (N, D)
                else:
                    features = features.reshape(N, -1)  # (N, T*D)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            
            sample_count += len(data)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {sample_count} samples...")
    
    extractor.remove_hooks()
    
    # 合并所有batch
    all_features = np.concatenate(all_features, axis=0)[:max_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:max_samples]
    
    print(f"✓ Extracted features shape: {all_features.shape}")
    print(f"✓ Labels shape: {all_labels.shape}")
    
    return all_features, all_labels


def get_layer_output_with_mem(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    device: torch.device,
    max_samples: int = 2000,
    return_mem: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取指定层的输出（包括membrane potential）
    专门用于SNN模型的时序特征提取
    
    Args:
        model: 模型
        dataloader: 数据加载器
        layer_name: 层名称
        device: 设备
        max_samples: 最大样本数
        return_mem: 是否返回membrane potential
    
    Returns:
        features: (N, T, D) 时序特征
        labels: (N,) 标签
    """
    model.eval()
    
    all_features = []
    all_labels = []
    sample_count = 0
    
    # 创建hook来捕获中间层输出
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    captured_output = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_output.append(output[0].detach())
        else:
            captured_output.append(output.detach())
    
    hook = target_module.register_forward_hook(hook_fn)
    
    print(f"Extracting temporal features from layer: {layer_name}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
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
                    all_labels.append(target.cpu())
                    sample_count += len(data)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {sample_count} samples...")
    
    hook.remove()
    
    # 合并所有batch
    all_features = torch.cat(all_features, dim=0)[:max_samples]
    all_labels = torch.cat(all_labels, dim=0)[:max_samples]
    
    print(f"✓ Extracted temporal features shape: {all_features.shape}")
    print(f"✓ Labels shape: {all_labels.shape}")
    
    return all_features, all_labels


def compute_parameter_difference(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """
    计算两个模型参数的差异
    
    Args:
        state_dict_a: 模型A的state_dict
        state_dict_b: 模型B的state_dict
    
    Returns:
        差异字典 {layer_name: {'l2_diff': float, 'cosine_sim': float}}
    """
    differences = {}
    
    # 获取共同的参数名
    common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
    
    for key in sorted(common_keys):
        # 跳过非张量参数（如整数、字符串等）
        if not isinstance(state_dict_a[key], torch.Tensor) or not isinstance(state_dict_b[key], torch.Tensor):
            continue
        
        param_a = state_dict_a[key].flatten().float()
        param_b = state_dict_b[key].flatten().float()
        
        # 计算L2差异
        l2_diff = torch.norm(param_a - param_b, p=2).item()
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            param_a.unsqueeze(0),
            param_b.unsqueeze(0)
        ).item()
        
        differences[key] = {
            'l2_diff': l2_diff,
            'cosine_sim': cos_sim,
            'shape': list(state_dict_a[key].shape),
            'num_params': param_a.numel()
        }
    
    return differences


def print_layer_names(model: nn.Module):
    """
    打印模型的所有层名称（用于调试）
    
    Args:
        model: 模型
    """
    print("\n=== Model Layer Names ===")
    for name, module in model.named_modules():
        if name:  # 跳过空名称
            print(f"  {name}: {module.__class__.__name__}")
    print("=" * 50 + "\n")

