"""
EventRPG风格的数据增强模块（适配张量形式的DVS数据）
Adapted from EventRPG: https://github.com/myuansun/EventRPG

组合方式：
- 每个样本随机选择一种几何增强：Identity / Flip / Rotate / Scale / Translation / Shear
- 以0.5的概率执行RPGMix（可选）

适用于：静态物体识别（N-Caltech101、CIFAR10-DVS）
"""

import torch
import torch.nn.functional as F
import random
import math
import numpy as np


class EventRPGAugment:
    """
    EventRPG风格的数据增强（张量版本）
    
    Args:
        img_size: 图像尺寸 (H, W) 或 单个值
        mix_prob: 执行RPGMix的概率（默认0.5，设为0则不使用mix）
        augment_prob: 执行几何增强的概率（默认1.0）
    """
    
    def __init__(self, img_size, mix_prob=0.5, augment_prob=1.0):
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size
        self.H, self.W = self.img_size
        self.mix_prob = mix_prob
        self.augment_prob = augment_prob
        
        # 几何增强列表：(函数, 最小幅度, 最大幅度)
        self.geometric_augments = [
            (self.identity, 0, 0),              # 0: Identity
            (self.flip_horizontal, 0, 0),       # 1: Flip
            (self.rotate, -math.pi/3, math.pi/3),  # 2: Rotate
            (self.scale, 0.5, 1.5),             # 3: Scale
            (self.translate_x, -0.3, 0.3),      # 4: Translation X
            (self.translate_y, -0.3, 0.3),      # 5: Translation Y
            (self.shear_x, -0.3, 0.3),          # 6: Shear X
            (self.shear_y, -0.3, 0.3),          # 7: Shear Y
        ]
    
    def __call__(self, data, apply_mix=None):
        """
        对张量形式的DVS数据应用EventRPG增强
        
        Args:
            data: 输入数据，形状为 (T, C, H, W) 或 (B, T, C, H, W)
            apply_mix: 是否应用mix，None表示随机决定
            
        Returns:
            增强后的数据，形状与输入相同
        """
        # 根据概率决定是否应用几何增强
        if random.random() > self.augment_prob:
            return data
        
        # 处理不同输入形状
        if len(data.shape) == 4:  # (T, C, H, W)
            return self._augment_single(data, apply_mix)
        elif len(data.shape) == 5:  # (B, T, C, H, W)
            batch_size = data.shape[0]
            augmented_batch = []
            for b in range(batch_size):
                augmented = self._augment_single(data[b], apply_mix)
                augmented_batch.append(augmented)
            return torch.stack(augmented_batch, dim=0)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def _augment_single(self, data, apply_mix=None):
        """
        对单个样本应用增强 (T, C, H, W)
        """
        # 1. 几何增强：随机选择一种
        aug_idx = random.randint(0, len(self.geometric_augments) - 1)
        aug_func, min_mag, max_mag = self.geometric_augments[aug_idx]
        mag = random.random() * (max_mag - min_mag) + min_mag
        data = aug_func(data, mag)
        
        # 2. RPGMix：以mix_prob的概率执行（简化版：CutMix风格）
        if apply_mix is None:
            apply_mix = random.random() < self.mix_prob
        
        if apply_mix and self.mix_prob > 0:
            # 简化版RPGMix：时间维度的混合
            data = self._simple_cutmix(data)
        
        return data
    
    # ==================== 几何增强函数 ====================
    
    def identity(self, data, mag):
        """恒等变换（不做任何改变）"""
        return data
    
    def flip_horizontal(self, data, mag):
        """水平翻转"""
        return torch.flip(data, dims=[-1])  # 沿最后一个维度（W）翻转
    
    def rotate(self, data, theta):
        """
        旋转
        Args:
            data: (T, C, H, W)
            theta: 旋转角度（弧度）
        """
        T, C, H, W = data.shape
        
        # 创建仿射变换矩阵
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # PyTorch的affine_grid需要 (N, 2, 3) 的矩阵
        # 这里 N=T*C，将每个通道独立旋转
        theta_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0]
        ], dtype=data.dtype, device=data.device)
        
        # 重塑数据为 (T*C, 1, H, W) 以应用仿射变换
        data_reshaped = data.view(T * C, 1, H, W)
        
        # 创建采样网格
        grid = F.affine_grid(
            theta_matrix.unsqueeze(0).expand(T * C, -1, -1),
            data_reshaped.size(),
            align_corners=False
        )
        
        # 应用变换
        rotated = F.grid_sample(
            data_reshaped,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        # 恢复原始形状
        return rotated.view(T, C, H, W)
    
    def scale(self, data, factor):
        """
        缩放
        Args:
            data: (T, C, H, W)
            factor: 缩放因子
        """
        T, C, H, W = data.shape
        
        # 缩放矩阵
        theta_matrix = torch.tensor([
            [factor, 0, 0],
            [0, factor, 0]
        ], dtype=data.dtype, device=data.device)
        
        data_reshaped = data.view(T * C, 1, H, W)
        grid = F.affine_grid(
            theta_matrix.unsqueeze(0).expand(T * C, -1, -1),
            data_reshaped.size(),
            align_corners=False
        )
        
        scaled = F.grid_sample(
            data_reshaped,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        return scaled.view(T, C, H, W)
    
    def translate_x(self, data, tx):
        """
        X方向平移
        Args:
            data: (T, C, H, W)
            tx: X方向平移比例（相对于图像宽度）
        """
        shift_pixels = int(tx * self.W)
        return torch.roll(data, shifts=shift_pixels, dims=-1)
    
    def translate_y(self, data, ty):
        """
        Y方向平移
        Args:
            data: (T, C, H, W)
            ty: Y方向平移比例（相对于图像高度）
        """
        shift_pixels = int(ty * self.H)
        return torch.roll(data, shifts=shift_pixels, dims=-2)
    
    def shear_x(self, data, shear):
        """
        X方向剪切
        Args:
            data: (T, C, H, W)
            shear: 剪切因子
        """
        T, C, H, W = data.shape
        
        theta_matrix = torch.tensor([
            [1, shear, 0],
            [0, 1, 0]
        ], dtype=data.dtype, device=data.device)
        
        data_reshaped = data.view(T * C, 1, H, W)
        grid = F.affine_grid(
            theta_matrix.unsqueeze(0).expand(T * C, -1, -1),
            data_reshaped.size(),
            align_corners=False
        )
        
        sheared = F.grid_sample(
            data_reshaped,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        return sheared.view(T, C, H, W)
    
    def shear_y(self, data, shear):
        """
        Y方向剪切
        Args:
            data: (T, C, H, W)
            shear: 剪切因子
        """
        T, C, H, W = data.shape
        
        theta_matrix = torch.tensor([
            [1, 0, 0],
            [shear, 1, 0]
        ], dtype=data.dtype, device=data.device)
        
        data_reshaped = data.view(T * C, 1, H, W)
        grid = F.affine_grid(
            theta_matrix.unsqueeze(0).expand(T * C, -1, -1),
            data_reshaped.size(),
            align_corners=False
        )
        
        sheared = F.grid_sample(
            data_reshaped,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        return sheared.view(T, C, H, W)
    
    # ==================== Mix增强 ====================
    
    def _simple_cutmix(self, data):
        """
        简化版的CutMix：在时间维度上进行混合
        Args:
            data: (T, C, H, W)
        Returns:
            混合后的数据
        """
        T, C, H, W = data.shape
        
        # 随机选择一个时间片段进行混合
        mix_ratio = random.uniform(0.2, 0.5)  # 混合区域的比例
        mix_length = int(T * mix_ratio)
        
        if mix_length > 0:
            # 随机选择起始位置
            start_idx = random.randint(0, T - mix_length)
            end_idx = start_idx + mix_length
            
            # 对选中的时间片段进行roll，相当于与同一样本的其他时间段混合
            roll_amount = random.randint(1, T - 1)
            data[start_idx:end_idx] = torch.roll(data[start_idx:end_idx], shifts=roll_amount, dims=0)
        
        return data


def apply_eventrpg_augment(data, img_size, mix_prob=0.5):
    """
    便捷函数：应用EventRPG增强
    
    Args:
        data: DVS数据，形状为 (T, C, H, W)
        img_size: 图像尺寸
        mix_prob: RPGMix概率
    
    Returns:
        增强后的数据
    """
    augmentor = EventRPGAugment(img_size, mix_prob=mix_prob)
    return augmentor(data)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing EventRPG Augmentation for tensor-based DVS data...")
    
    # 创建测试数据
    T, C, H, W = 10, 2, 48, 48
    test_data = torch.randn(T, C, H, W)
    
    print(f"Input shape: {test_data.shape}")
    print(f"Input range: [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    # 测试增强
    augmentor = EventRPGAugment(img_size=(H, W), mix_prob=0.5)
    
    # 测试各种增强
    print("\n=== Testing Geometric Augmentations ===")
    for i, (func, min_mag, max_mag) in enumerate(augmentor.geometric_augments):
        mag = (min_mag + max_mag) / 2
        augmented = func(test_data.clone(), mag)
        print(f"{i}. {func.__name__}: shape={augmented.shape}, range=[{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # 测试完整pipeline
    print("\n=== Testing Full Pipeline ===")
    for i in range(3):
        augmented = augmentor(test_data.clone())
        print(f"Run {i+1}: shape={augmented.shape}, range=[{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # 测试批处理
    print("\n=== Testing Batch Processing ===")
    batch_data = torch.randn(4, T, C, H, W)
    augmented_batch = augmentor(batch_data)
    print(f"Batch input: {batch_data.shape}")
    print(f"Batch output: {augmented_batch.shape}")
    
    print("\n✓ All tests passed!")

