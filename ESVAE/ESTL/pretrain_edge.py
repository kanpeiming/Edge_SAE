#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
边缘分支预训练主脚本

用法:
python pretrain_edge.py --epochs 50 --batch_size 64 --lr 0.001
"""

import argparse
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch.utils.tensorboard import SummaryWriter

from ESTL.models import VGGEdgeBranch
from ESTL.edge_pretrainer import EdgePretrainer
from ESTL.edge_dataloader import get_edge_dataloaders
from tl_utils.common_utils import seed_all


def parse_args():
    parser = argparse.ArgumentParser(description='边缘分支预训练')
    
    # 数据参数
    parser.add_argument('--rgb_root', type=str,
                       default='/home/user/kpm/kpm//Dataset/CIFAR10/cifar10',
                       help='RGB CIFAR10路径')
    parser.add_argument('--edge_root', type=str,
                       default='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10-edge',
                       help='边缘数据路径')
    parser.add_argument('--batch_size', default=64, type=int,
                       help='批次大小')
    parser.add_argument('--augmentation', action='store_true',
                       help='启用数据增强')
    
    # 模型参数
    parser.add_argument('--img_shape', default=32, type=int,
                       help='图像尺寸')
    
    # 训练参数
    parser.add_argument('--epochs', default=50, type=int,
                       help='训练轮数')
    parser.add_argument('--lr', default=0.001, type=float,
                       help='学习率')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                       help='权重衰减')
    parser.add_argument('--seed', type=int, default=1000,
                       help='随机种子')
    
    # 路径参数
    parser.add_argument('--log_dir', type=str,
                       default='/home/user/kpm/kpm/results/ESTL/edge-branch/log_dir',
                       help='TensorBoard日志目录')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/user/kpm/kpm/results/ESTL/edge-branch/checkpoints',
                       help='检查点目录')
    parser.add_argument('--GPU_id', type=int, default=0,
                       help='GPU ID')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 设备
    device = torch.device(f"cuda:{args.GPU_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 随机种子
    seed_all(args.seed)
    
    # 实验名称
    exp_name = f"EdgeBranch_Pretrain_lr{args.lr}_seed{args.seed}"
    
    # 路径
    log_dir = os.path.join(args.log_dir, exp_name)
    checkpoint_dir = os.path.join(args.checkpoint, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    print("\n" + "="*70)
    print("边缘分支预训练".center(70))
    print("="*70)
    print(f"实验: {exp_name}")
    print(f"日志目录: {log_dir}")
    print(f"检查点目录: {checkpoint_dir}")
    print("="*70 + "\n")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader = get_edge_dataloaders(
        rgb_root=args.rgb_root,
        edge_root=args.edge_root,
        batch_size=args.batch_size,
        num_workers=8,
        augmentation=args.augmentation
    )
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")
    print(f"  数据增强: {'启用' if args.augmentation else '禁用'}")
    
    # 创建模型（预训练模式）
    print("\n创建模型...")
    model = VGGEdgeBranch(
        img_shape=args.img_shape,
        pretrain_mode=True  # 预训练模式
    ).to(device)
    
    print(f"  EdgeBranch (预训练模式)")
    print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # 训练器
    trainer = EdgePretrainer(
        args=args,
        device=device,
        writer=writer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path=checkpoint_dir
    )
    
    # 训练
    best_val_loss = trainer.train(train_loader, val_loader)
    
    # 总结
    print("\n" + "="*70)
    print("训练总结".center(70))
    print("="*70)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型路径: {checkpoint_dir}/best_edge_branch.pth")
    print("="*70 + "\n")
    
    writer.close()
    print("完成!")


if __name__ == "__main__":
    main()

