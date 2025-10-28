# -*- coding: utf-8 -*-
"""
训练调试脚本 - 检查数据和模型输出
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
from ESTL.models import BaselineVGGSNN
from ESTL.caltech101_dataloader import get_caltech101_dvs_only_dataloaders

def check_data_and_model():
    """检查数据加载和模型输出"""
    
    print("="*80)
    print("训练调试诊断")
    print("="*80)
    
    # 加载数据
    print("\n1. 加载数据...")
    dvs_root = '/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101/Caltech101'
    
    try:
        train_loader, val_loader, test_loader = get_caltech101_dvs_only_dataloaders(
            dvs_root=dvs_root,
            batch_size=4,  # 小批次用于调试
            train_ratio=0.8,
            val_ratio=0.1,
            augmentation=False,
            img_size=48,
            T=10,
            num_workers=0,
            dvs_format='bin'
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    print(f"✓ 数据加载成功")
    
    # 检查一个batch的数据
    print("\n2. 检查数据形状和统计...")
    data_iter = iter(train_loader)
    dvs_data, labels = next(data_iter)
    
    print(f"  DVS数据形状: {dvs_data.shape}")  # 应该是 (B, T, C, H, W)
    print(f"  标签形状: {labels.shape}")
    print(f"  DVS数据范围: [{dvs_data.min():.4f}, {dvs_data.max():.4f}]")
    print(f"  DVS数据均值: {dvs_data.mean():.4f}")
    print(f"  DVS数据标准差: {dvs_data.std():.4f}")
    print(f"  非零元素比例: {(dvs_data != 0).float().mean():.4f}")
    
    # 检查每个时间步
    print("\n  各时间步统计:")
    for t in range(dvs_data.shape[1]):
        frame = dvs_data[:, t, :, :, :]
        print(f"    T={t}: 均值={frame.mean():.4f}, 非零比例={((frame != 0).float().mean()):.4f}")
    
    # 创建模型
    print("\n3. 创建模型...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaselineVGGSNN(cls_num=100, img_shape=48).to(device)
    print(f"✓ 模型创建成功，设备: {device}")
    
    # 前向传播
    print("\n4. 测试前向传播...")
    model.eval()
    dvs_data = dvs_data.to(device).float()
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(dvs_data)
    
    print(f"  输出形状: {outputs.shape}")  # 应该是 (B, T, num_classes)
    print(f"  输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"  输出均值: {outputs.mean():.4f}")
    print(f"  输出标准差: {outputs.std():.4f}")
    
    # 检查每个时间步的输出
    print("\n  各时间步输出统计:")
    for t in range(outputs.shape[1]):
        out_t = outputs[:, t, :]
        print(f"    T={t}: 均值={out_t.mean():.4f}, 标准差={out_t.std():.4f}, 范围=[{out_t.min():.4f}, {out_t.max():.4f}]")
    
    # 计算预测准确率
    outputs_mean = outputs.mean(dim=1)  # (B, T, C) -> (B, C)
    _, predicted = outputs_mean.max(1)
    accuracy = predicted.eq(labels).float().mean()
    print(f"\n  随机初始化准确率: {accuracy*100:.2f}% (期望约1%对于100类)")
    
    # 检查梯度
    print("\n5. 检查梯度...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 前向传播
    outputs = model(dvs_data)
    
    # 计算TET loss
    T = outputs.size(1)
    loss = 0
    for t in range(T):
        loss += criterion(outputs[:, t, :], labels)
    loss = loss / T
    
    print(f"  损失值: {loss.item():.4f}")
    print(f"  期望损失 (随机): {np.log(100):.4f}")  # -log(1/100) ≈ 4.605
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 0:
                print(f"  {name}: grad_norm={grad_norm:.6f}")
    
    if len(grad_norms) == 0:
        print("  ❌ 警告: 没有梯度!")
    elif max(grad_norms) < 1e-6:
        print("  ❌ 警告: 梯度过小!")
    else:
        print(f"  ✓ 梯度正常，平均梯度范数: {np.mean(grad_norms):.6f}")
    
    # 优化器步骤
    optimizer.step()
    
    # 再次前向传播检查参数是否更新
    with torch.no_grad():
        outputs_after = model(dvs_data)
        diff = (outputs_after - outputs).abs().mean()
        print(f"\n  参数更新后输出变化: {diff:.6f}")
        if diff < 1e-6:
            print("  ❌ 警告: 参数更新后输出几乎没有变化!")
        else:
            print("  ✓ 参数更新正常")
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)

if __name__ == "__main__":
    check_data_and_model()

