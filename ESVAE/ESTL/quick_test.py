# -*- coding: utf-8 -*-
"""
快速测试修复效果
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from ESTL.models import BaselineVGGSNN
from ESTL.caltech101_dataloader import get_caltech101_dvs_only_dataloaders
from tl_utils.loss_function import TET_loss

def quick_test():
    """快速测试修复效果"""
    
    print("="*80)
    print("快速测试修复效果")
    print("="*80)
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    dvs_root = '/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101/Caltech101'
    
    train_loader, val_loader, _ = get_caltech101_dvs_only_dataloaders(
        dvs_root=dvs_root,
        batch_size=16,
        train_ratio=0.8,
        val_ratio=0.1,
        augmentation=False,
        img_size=48,
        T=10,
        num_workers=0,
        dvs_format='bin'
    )
    
    # 创建模型
    print("\n创建模型...")
    model = BaselineVGGSNN(cls_num=100, img_shape=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试一个batch
    print("\n测试训练一个batch...")
    data_iter = iter(train_loader)
    dvs_data, labels = next(data_iter)
    
    dvs_data = dvs_data.to(device).float()
    labels = labels.to(device)
    
    print(f"  数据形状: {dvs_data.shape}")
    print(f"  数据范围: [{dvs_data.min():.4f}, {dvs_data.max():.4f}]")
    print(f"  数据均值: {dvs_data.mean():.4f}")
    print(f"  非零比例: {(dvs_data != 0).float().mean():.4f}")
    
    # 前向传播
    model.train()
    optimizer.zero_grad()
    outputs = model(dvs_data)
    
    print(f"\n  输出形状: {outputs.shape}")
    print(f"  输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"  输出均值: {outputs.mean():.4f}")
    
    # 计算损失
    loss = TET_loss(outputs, labels)
    print(f"\n  损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  梯度总范数: {total_norm:.6f}")
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 优化器步骤
    optimizer.step()
    
    # 再次前向传播
    with torch.no_grad():
        outputs_after = model(dvs_data)
        diff = (outputs_after - outputs).abs().mean()
        print(f"  参数更新后输出变化: {diff:.6f}")
    
    # 计算准确率
    outputs_mean = outputs.mean(dim=1)
    _, predicted = outputs_mean.max(1)
    accuracy = predicted.eq(labels).float().mean()
    print(f"\n  当前准确率: {accuracy*100:.2f}%")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
    
    # 诊断结果
    print("\n诊断结果:")
    if dvs_data.mean() < 0.01:
        print("  ⚠️  警告: DVS数据均值过小，可能数据太稀疏")
    elif dvs_data.mean() > 0.5:
        print("  ⚠️  警告: DVS数据均值过大，归一化可能有问题")
    else:
        print("  ✓ DVS数据均值正常")
    
    if (dvs_data != 0).float().mean() < 0.01:
        print("  ⚠️  警告: 非零比例过低，数据太稀疏")
    elif (dvs_data != 0).float().mean() > 0.5:
        print("  ⚠️  警告: 非零比例过高，可能不是真实的DVS数据")
    else:
        print("  ✓ 非零比例正常")
    
    if total_norm < 1e-6:
        print("  ❌ 错误: 梯度过小，模型无法学习")
    elif total_norm > 100:
        print("  ⚠️  警告: 梯度过大，可能需要调整学习率或梯度裁剪阈值")
    else:
        print("  ✓ 梯度范数正常")
    
    if diff < 1e-6:
        print("  ❌ 错误: 参数更新后输出几乎没变化，优化器可能有问题")
    else:
        print("  ✓ 参数更新正常")
    
    print("\n建议:")
    if dvs_data.mean() < 0.01 or (dvs_data != 0).float().mean() < 0.01:
        print("  - 数据太稀疏，考虑:")
        print("    1. 增加时间窗口")
        print("    2. 调整归一化方法")
        print("    3. 检查DVS数据是否正确加载")
    
    if total_norm > 10:
        print("  - 梯度较大，考虑:")
        print("    1. 降低学习率 (如 0.0001)")
        print("    2. 调整梯度裁剪阈值")
    
    if accuracy < 0.05:
        print("  - 准确率过低（低于5%），考虑:")
        print("    1. 检查标签是否正确")
        print("    2. 增加训练时间")
        print("    3. 调整模型架构或超参数")

if __name__ == "__main__":
    quick_test()

