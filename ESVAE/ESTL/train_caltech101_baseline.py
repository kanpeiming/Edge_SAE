# -*- coding: utf-8 -*-
"""
N-Caltech101 DVS 分类训练脚本 (Baseline - 仅使用DVS数据)

使用TET (Temporal Efficient Training)训练SNN进行N-Caltech101分类

数据集说明:
- N-Caltech101: 从Caltech101转换的DVS事件数据
- 原始101类，移除Faces类别后剩余100类（符合N-Caltech101标准）
- 数据路径: /home/user/kpm/kpm/Dataset/Caltech101/caltech101/caltech-101

使用方法:
    # 基础训练 (自动检测DVS数据格式)
    python train_caltech101_baseline.py --epochs 150 --batch_size 32 --lr 0.001
    
    # 启用数据增强
    python train_caltech101_baseline.py --epochs 150 --enable_augmentation
    
    # 调整数据集划分比例
    python train_caltech101_baseline.py --train_ratio 0.7 --val_ratio 0.15
    
    # 指定DVS数据格式
    python train_caltech101_baseline.py --dvs_format npy  # 或 pt, bin
"""

import argparse
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ESTL.models import BaselineVGGSNN, BaselineSpikformer
from ESTL.caltech101_dataloader import get_caltech101_dvs_only_dataloaders
from tl_utils.loss_function import TET_loss
from tl_utils.common_utils import seed_all


def parse_args():
    parser = argparse.ArgumentParser(
        description='N-Caltech101 DVS 分类训练 (Baseline)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==== 数据集配置 ====
    parser.add_argument('--batch_size', default=32, type=int,
                        help='批次大小 (显存不足可用8-16)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例 (0.7 = 70%用于训练)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例 (0.15 = 15%用于验证，剩余15%用于测试)')
    parser.add_argument('--enable_augmentation', action='store_true',
                        help='启用DVS数据增强 (翻转、旋转、平移、噪声)')
    parser.add_argument('--dvs_format', type=str, default='bin',
                        choices=['auto', 'npy', 'pt', 'bin'],
                        help='DVS数据格式 (默认bin)')

    # ==== 模型配置 ====
    parser.add_argument('--model', type=str, default='vgg',
                        choices=['vgg', 'spikformer'],
                        help='模型架构: vgg (CNN-SNN) 或 spikformer (ViT-SNN)')
    parser.add_argument('--img_shape', default=48, type=int,
                        help='输入图像大小 (N-Caltech101推荐48/64/128, 显存不足请用48)')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='类别数量 (Caltech101移除BACKGROUND和Faces_easy后为100类)')
    parser.add_argument('--T', default=10, type=int,
                        help='SNN时间步数 (显存不足可用4-6, 标准是10)')
    
    # Spikformer特定参数
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Spikformer patch大小 (仅当model=spikformer时有效)')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Spikformer嵌入维度 (仅当model=spikformer时有效)')
    parser.add_argument('--depth', type=int, default=4,
                        help='Spikformer层数 (仅当model=spikformer时有效)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Spikformer注意力头数 (仅当model=spikformer时有效)')

    # ==== 训练配置 ====
    parser.add_argument('--epochs', default=100, type=int,
                        help='训练轮数')
    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['SGD', 'Adam'],
                        help='优化器')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='学习率')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='权重衰减 (L2正则化)')
    parser.add_argument('--scheduler_step', default=30, type=int,
                        help='学习率衰减步长 (每N个epoch衰减一次)')
    parser.add_argument('--scheduler_gamma', default=0.5, type=float,
                        help='学习率衰减系数')
    parser.add_argument('--seed', type=int, default=1000,
                        help='随机种子')

    # ==== 路径配置 ====
    parser.add_argument('--log_dir', type=str, 
                        default='/home/user/kpm/kpm/results/ESTL/ncaltech101/log_dir',
                        help='TensorBoard日志目录')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/user/kpm/kpm/results/ESTL/ncaltech101/checkpoints',
                        help='模型检查点保存目录')
    parser.add_argument('--dvs_root', type=str,
                        default='/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101/Caltech101',
                        help='N-Caltech101 DVS数据集根目录')
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU设备ID')
    
    # ==== 实验命名 ====
    parser.add_argument('--exp_suffix', type=str, default='',
                        help='实验名称后缀 (用于区分不同实验)')

    args = parser.parse_args()
    return args


def auto_detect_dvs_format(dvs_root):
    """
    自动检测DVS数据格式
    
    Args:
        dvs_root: DVS数据根目录
    
    Returns:
        format: 'npy', 'pt', 或 'bin'
    """
    print("\n正在自动检测DVS数据格式...")
    
    # 获取第一个类别目录
    class_dirs = [d for d in os.listdir(dvs_root) 
                  if os.path.isdir(os.path.join(dvs_root, d))]
    
    if len(class_dirs) == 0:
        raise ValueError(f"未找到类别目录: {dvs_root}")
    
    # 检查第一个类别目录中的文件扩展名
    first_class = os.path.join(dvs_root, class_dirs[0])
    files = os.listdir(first_class)
    
    for ext, format_name in [('.npy', 'npy'), ('.pt', 'pt'), ('.bin', 'bin')]:
        if any(f.endswith(ext) for f in files):
            print(f"✓ 检测到DVS数据格式: {format_name}")
            return format_name
    
    # 如果没有找到，尝试检测是否是图像文件（需要转换）
    if any(f.endswith(('.jpg', '.png', '.jpeg')) for f in files):
        raise ValueError(
            f"检测到图像文件，但需要DVS事件数据。\n"
            f"请确保路径指向N-Caltech101 DVS数据集，而非RGB Caltech101。"
        )
    
    raise ValueError(f"无法检测DVS数据格式，请手动指定 --dvs_format")


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader),
        desc=f'Epoch [{epoch+1:3d}/{total_epochs}] Train',
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for batch_idx, (data, labels) in pbar:
        # DVS数据
        dvs_data = data.to(device).float()
        labels = labels.to(device)
        
        # 调试：打印第一个batch的形状
        if batch_idx == 0 and epoch == 0:
            print(f"\n[DEBUG] DVS数据形状: {dvs_data.shape}")
            print(f"[DEBUG] DVS数据范围: [{dvs_data.min():.4f}, {dvs_data.max():.4f}]")
            print(f"[DEBUG] DVS数据均值: {dvs_data.mean():.4f}")
            print(f"[DEBUG] 非零比例: {(dvs_data != 0).float().mean():.4f}")
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(dvs_data)
        
        # 调试：打印第一个batch的输出
        if batch_idx == 0 and epoch == 0:
            print(f"[DEBUG] 模型输出形状: {outputs.shape}")
            print(f"[DEBUG] 输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
            print(f"[DEBUG] 输出均值: {outputs.mean():.4f}\n")
        
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        outputs_mean = outputs.mean(dim=1)  # (B, T, C) -> (B, C) 时间维度平均
        _, predicted = outputs_mean.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    pbar.close()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, phase='Val'):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    correct_top5 = 0
    
    pbar = tqdm(
        val_loader,
        desc=f'{phase:>5s}',
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]',
        leave=False
    )
    
    with torch.no_grad():
        for data, labels in pbar:
            dvs_data = data.to(device).float()
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(dvs_data)
            loss = criterion(outputs, labels)
            
            # 统计
            total_loss += loss.item()
            outputs_mean = outputs.mean(dim=1)  # 时间维度平均
            _, predicted = outputs_mean.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5准确率
            _, top5_pred = outputs_mean.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    pbar.close()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    accuracy_top5 = correct_top5 / total
    
    return avg_loss, accuracy, accuracy_top5


def main():
    args = parse_args()
    
    # 设备配置
    device = torch.device(f"cuda:{args.GPU_id}" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*100)
    print("N-Caltech101 DVS 分类训练 (Baseline - 仅使用DVS数据)".center(100))
    print("="*100)
    print(f"设备: {device}")
    
    # 设置随机种子
    seed_all(args.seed)
    print(f"随机种子: {args.seed}")
    
    # 自动检测DVS数据格式（如果设置为auto）
    if args.dvs_format == 'auto':
        try:
            args.dvs_format = auto_detect_dvs_format(args.dvs_root)
        except Exception as e:
            print(f"❌ 错误: {e}")
            print(f"请手动指定DVS数据格式: --dvs_format [npy|pt|bin]")
            return
    else:
        print(f"\n使用指定的DVS数据格式: {args.dvs_format}")
    
    # 实验名称
    model_name = args.model.upper()
    exp_name = (
        f"NCaltech101_{model_name}_"
        f"T{args.T}_lr{args.lr}_wd{args.weight_decay}_"
        f"bs{args.batch_size}_ep{args.epochs}_"
        f"train{int(args.train_ratio*100)}_seed{args.seed}"
    )
    if args.model == 'spikformer':
        exp_name += f"_patch{args.patch_size}_dim{args.embed_dim}_depth{args.depth}"
    if args.exp_suffix:
        exp_name += f"_{args.exp_suffix}"
    
    # 创建输出目录
    log_dir = os.path.join(args.log_dir, exp_name)
    checkpoint_dir = os.path.join(args.checkpoint, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n实验配置:")
    print(f"  实验名称: {exp_name}")
    print(f"  日志目录: {log_dir}")
    print(f"  检查点目录: {checkpoint_dir}")
    
    # 加载数据
    print(f"\n{'='*100}")
    print("加载数据集...")
    print(f"{'='*100}")
    print(f"DVS数据路径: {args.dvs_root}")
    print(f"DVS数据格式: {args.dvs_format}")
    print(f"数据划分: 训练={args.train_ratio:.0%}, 验证={args.val_ratio:.0%}, 测试={1-args.train_ratio-args.val_ratio:.0%}")
    
    try:
        train_loader, val_loader, test_loader = get_caltech101_dvs_only_dataloaders(
            dvs_root=args.dvs_root,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            augmentation=args.enable_augmentation,
            img_size=args.img_shape,
            T=args.T,
            num_workers=8,  # 设置为0避免共享内存问题
            dvs_format=args.dvs_format
        )
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        print(f"\n请检查:")
        print(f"  1. DVS数据路径是否正确: {args.dvs_root}")
        print(f"  2. 数据格式是否正确: {args.dvs_format}")
        print(f"  3. 目录结构是否为: dvs_root/类别名/数据文件.{args.dvs_format}")
        return
    
    print(f"\n✓ 数据集加载成功:")
    print(f"  训练集: {len(train_loader.dataset)} 样本 ({len(train_loader)} 批次)")
    print(f"  验证集: {len(val_loader.dataset)} 样本 ({len(val_loader)} 批次)")
    print(f"  测试集: {len(test_loader.dataset)} 样本 ({len(test_loader)} 批次)")
    print(f"  类别数: {args.num_classes}")
    print(f"  图像大小: {args.img_shape}x{args.img_shape}")
    print(f"  时间步数: {args.T}")
    print(f"  数据增强: {'✓ 启用' if args.enable_augmentation else '✗ 禁用'}")
    
    # 创建模型
    print(f"\n{'='*100}")
    print("创建模型...")
    print(f"{'='*100}")
    
    if args.model == 'vgg':
        model = BaselineVGGSNN(
            cls_num=args.num_classes,
            img_shape=args.img_shape
        ).to(device)
        model_desc = "BaselineVGGSNN (CNN-SNN, DVS-only)"
    elif args.model == 'spikformer':
        model = BaselineSpikformer(
            img_size_h=args.img_shape,
            img_size_w=args.img_shape,
            patch_size=args.patch_size,
            in_channels=2,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0
        ).to(device)
        model_desc = f"BaselineSpikformer (ViT-SNN, DVS-only)\n" \
                    f"  Patch size: {args.patch_size}x{args.patch_size}\n" \
                    f"  Embed dim: {args.embed_dim}\n" \
                    f"  Depth: {args.depth}\n" \
                    f"  Num heads: {args.num_heads}"
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型: {model_desc}")
    print(f"  总参数量: {total_params/1e6:.2f}M")
    print(f"  可训练参数: {trainable_params/1e6:.2f}M")
    
    # 优化器
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr,
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.scheduler_step, 
        gamma=args.scheduler_gamma
    )
    
    print(f"\n训练配置:")
    print(f"  优化器: {args.optim}")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率调度: StepLR (每{args.scheduler_step}轮 × {args.scheduler_gamma})")
    print(f"  损失函数: TET_loss")
    
    # 训练循环
    print(f"\n{'='*100}")
    print("开始训练...")
    print(f"{'='*100}\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, TET_loss, device, epoch, args.epochs
        )
        
        # 验证
        val_loss, val_acc, val_acc_top5 = validate(
            model, val_loader, TET_loss, device, phase='Val'
        )
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard日志
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/accuracy_top5', val_acc_top5, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        
        # 打印摘要
        print(f"\nEpoch [{epoch+1:3d}/{args.epochs}] Summary:")
        print(f"  {'='*96}")
        print(f"  Train │ Loss: {train_loss:.4f} │ Acc@1: {train_acc*100:6.2f}%")
        print(f"  Val   │ Loss: {val_loss:.4f} │ Acc@1: {val_acc*100:6.2f}% │ Acc@5: {val_acc_top5*100:6.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_baseline.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'args': args
            }, checkpoint_path)
            
            print(f"  ✓ 最佳模型已保存! Val Acc: {best_val_acc*100:.2f}%")
        
        print(f"  {'='*96}\n")
    
    # 测试最佳模型
    print(f"\n{'='*100}")
    print("测试最佳模型...")
    print(f"{'='*100}")
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_baseline.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 已加载最佳模型 (Epoch {best_epoch+1})")
    
    test_loss, test_acc, test_acc_top5 = validate(
        model, test_loader, TET_loss, device, phase='Test'
    )
    
    # 最终摘要
    print(f"\n{'='*100}")
    print("训练完成 - 最终结果".center(100))
    print(f"{'='*100}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}% (Epoch {best_epoch+1})")
    print(f"测试集准确率 @1: {test_acc*100:.2f}%")
    print(f"测试集准确率 @5: {test_acc_top5*100:.2f}%")
    print(f"{'='*100}\n")
    
    # 保存结果
    results_file = os.path.join(checkpoint_dir, 'results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"N-Caltech101 DVS 分类训练结果\n")
        f.write(f"{'='*100}\n\n")
        
        f.write(f"实验配置:\n")
        f.write(f"  实验名称: {exp_name}\n")
        f.write(f"  数据集: N-Caltech101 (DVS-only)\n")
        f.write(f"  DVS数据路径: {args.dvs_root}\n")
        f.write(f"  DVS数据格式: {args.dvs_format}\n")
        f.write(f"  类别数: {args.num_classes}\n")
        f.write(f"  图像大小: {args.img_shape}x{args.img_shape}\n")
        f.write(f"  时间步数: {args.T}\n")
        f.write(f"  数据划分: 训练={args.train_ratio:.0%}, 验证={args.val_ratio:.0%}\n")
        f.write(f"  数据增强: {'启用' if args.enable_augmentation else '禁用'}\n\n")
        
        f.write(f"训练配置:\n")
        f.write(f"  模型: {args.model.upper()}\n")
        if args.model == 'spikformer':
            f.write(f"  Patch size: {args.patch_size}\n")
            f.write(f"  Embed dim: {args.embed_dim}\n")
            f.write(f"  Depth: {args.depth}\n")
            f.write(f"  Num heads: {args.num_heads}\n")
        f.write(f"  优化器: {args.optim}\n")
        f.write(f"  学习率: {args.lr}\n")
        f.write(f"  权重衰减: {args.weight_decay}\n")
        f.write(f"  批次大小: {args.batch_size}\n")
        f.write(f"  训练轮数: {args.epochs}\n")
        f.write(f"  随机种子: {args.seed}\n\n")
        
        f.write(f"训练结果:\n")
        f.write(f"  最佳验证准确率: {best_val_acc*100:.2f}% (Epoch {best_epoch+1})\n")
        f.write(f"  测试集准确率 @1: {test_acc*100:.2f}%\n")
        f.write(f"  测试集准确率 @5: {test_acc_top5*100:.2f}%\n\n")
        
        f.write(f"模型保存路径:\n")
        f.write(f"  检查点: {checkpoint_dir}/best_baseline.pth\n")
        f.write(f"  日志: {log_dir}\n")
    
    print(f"✓ 结果已保存到: {results_file}\n")
    
    writer.close()
    print("训练完成!")


if __name__ == "__main__":
    main()

