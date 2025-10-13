# -*- coding: utf-8 -*-
"""
Training script for Baseline DVS Classification (DVS-only, no edge guidance)

Usage:
    python train_baseline.py --epochs 150 --lr 0.001 --batch_size 32 --weight_decay 0.001
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

from ESTL.models import BaselineVGGSNN
from ESTL.enhanced_dataloader import get_cifar10_DVS_enhanced
from tl_utils.loss_function import TET_loss
from tl_utils.common_utils import seed_all


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline DVS Classification (DVS-only)')

    # Data
    parser.add_argument('--data_set', type=str, default='CIFAR10',
                        help='Dataset name')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--dvs_sample_ratio', type=float, default=1.0,
                        help='Ratio of DVS training set')
    parser.add_argument('--enable_augmentation', action='store_true',
                        help='Enable data augmentation for DVS data')

    # Model
    parser.add_argument('--img_shape', default=32, type=int,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--T', default=10, type=int,
                        help='SNN simulation time steps')

    # Training
    parser.add_argument('--epochs', default=150, type=int,
                        help='Training epochs')
    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['SGD', 'Adam'],
                        help='Optimizer')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=1000,
                        help='Random seed')

    # Paths
    parser.add_argument('--log_dir', type=str, 
                        default='/home/user/kpm/kpm/results/ESTL/baseline/log_dir',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/user/kpm/kpm/results/ESTL/baseline/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--dvs_root', type=str,
                        default='/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat',
                        help='DVS CIFAR10 dataset root')
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU ID')

    args = parser.parse_args()
    return args


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader), 
                total=len(train_loader),
                desc=f'Epoch {epoch+1}',
                ncols=100)
    
    for batch_idx, (data, labels) in pbar:
        # Unpack data (DVS-only)
        if isinstance(data, (tuple, list)):
            # If data is (rgb, dvs), take only dvs
            _, dvs_data = data
            dvs_data = dvs_data.to(device).float()
        else:
            dvs_data = data.to(device).float()
        
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(dvs_data)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        outputs_mean = outputs.mean(dim=1)  # (B, T, C) -> (B, C) 在时间维度求平均
        _, predicted = outputs_mean.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    pbar.close()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for data, labels in val_loader:
            # Unpack data (DVS-only)
            if isinstance(data, (tuple, list)):
                _, dvs_data = data
                dvs_data = dvs_data.to(device).float()
            else:
                dvs_data = data.to(device).float()
            
            labels = labels.to(device)
            
            # Forward
            outputs = model(dvs_data)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            outputs_mean = outputs.mean(dim=1)  # (B, T, C) -> (B, C) 在时间维度求平均
            _, predicted = outputs_mean.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs_mean.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    accuracy_top5 = correct_top5 / total
    
    return avg_loss, accuracy, accuracy_top5


def main():
    args = parse_args()
    
    # Device
    device = torch.device(f"cuda:{args.GPU_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    seed_all(args.seed)
    
    # Experiment name
    exp_name = (
        f"Baseline_{args.data_set}_"
        f"T{args.T}_lr{args.lr}_wd{args.weight_decay}_"
        f"bs{args.batch_size}_ep{args.epochs}_seed{args.seed}"
    )
    
    # Paths
    log_dir = os.path.join(args.log_dir, exp_name)
    checkpoint_dir = os.path.join(args.checkpoint, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    print("\n" + "="*70)
    print("Baseline DVS Classification (DVS-only)".center(70))
    print("="*70)
    print(f"Experiment: {exp_name}")
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*70 + "\n")
    
    # Load data (DVS-only, no RGB)
    print("Loading data...")
    if args.data_set == 'CIFAR10':
        train_loader, val_loader, test_loader = get_cifar10_DVS_enhanced(
            args.batch_size,
            T=args.T,
            train_set_ratio=args.dvs_sample_ratio,
            encode_type='TET',
            augmentation=args.enable_augmentation,
            rgb_root=None,  # ⚠️ Baseline不使用RGB数据
            dvs_root=args.dvs_root,
            val_split=0.1  # 从训练集中划分10%作为验证集
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print(f"  Data augmentation: {'Enabled' if args.enable_augmentation else 'Disabled'}")
        print(f"  Mode: DVS-only (Baseline)")
        print(f"  DVS data: {args.dvs_root}")
    else:
        raise NotImplementedError(f"Dataset {args.data_set} not supported yet")
    
    # Create model
    print("\nCreating model...")
    model = BaselineVGGSNN(
        cls_num=args.num_classes,
        img_shape=args.img_shape
    ).to(device)
    
    print(f"  Baseline Model (DVS-only):")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"    Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    
    # Optimizer
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    print(f"\n训练配置:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}")
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "="*70)
    print("开始训练...")
    print("="*70)
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, TET_loss, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_acc_top5 = validate(
            model, val_loader, TET_loss, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # Log
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/accuracy_top5', val_acc_top5, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val Acc@5: {val_acc_top5*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_baseline.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': args
            }, checkpoint_path)
            
            print(f"  ✓ Best model saved! Val Acc: {best_val_acc*100:.2f}%")
    
    # Test on best model
    print("\n" + "="*70)
    print("Testing on best model...")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_baseline.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_acc_top5 = validate(model, test_loader, TET_loss, device)
    
    # Summary
    print("\n" + "="*70)
    print("Training Summary".center(70))
    print("="*70)
    print(f"Best Val Acc@1:   {best_val_acc*100:.2f}% (Epoch {best_epoch+1})")
    print(f"Test Acc@1:       {test_acc*100:.2f}%")
    print(f"Test Acc@5:       {test_acc_top5*100:.2f}%")
    print("="*70 + "\n")
    
    # Save results
    results_file = os.path.join(checkpoint_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"="*70 + "\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  Dataset: {args.data_set}\n")
        f.write(f"  Model: BaselineVGGSNN (DVS-only)\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Time steps (T): {args.T}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Best Val Acc@1: {best_val_acc*100:.2f}% (Epoch {best_epoch+1})\n")
        f.write(f"  Test Acc@1:     {test_acc*100:.2f}%\n")
        f.write(f"  Test Acc@5:     {test_acc_top5*100:.2f}%\n")
    
    print(f"Results saved to: {results_file}")
    
    writer.close()
    print("Done!")


if __name__ == "__main__":
    main()

