# -*- coding: utf-8 -*-
"""
Training script for Edge-Guided DVS Classification

Usage:
    python train.py --epochs 100 --lr 0.001 --batch_size 32
    python train.py --fusion_stages 2 3 4 --fusion_type cross_attention
    python train.py --train_baseline --epochs 100
"""

import argparse
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch.utils.tensorboard import SummaryWriter

from ESTL.models import EdgeGuidedVGGSNN
from ESTL.trainer import EdgeGuidedTrainer
from ESTL.enhanced_dataloader import get_cifar10_DVS_enhanced
from tl_utils.loss_function import TET_loss
from tl_utils.common_utils import seed_all


def parse_args():
    parser = argparse.ArgumentParser(description='Edge-Guided DVS Classification with D2CAF Fusion')

    # Data
    parser.add_argument('--data_set', type=str, default='CIFAR10',
                        help='Dataset name')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--dvs_sample_ratio', type=float, default=1.0,
                        help='Ratio of DVS training set')
    parser.add_argument('--enable_augmentation', action='store_true',
                        help='Enable data augmentation for DVS data')
    parser.add_argument('--train_with_rgb_size', action='store_true',
                        help='Use RGB dataset size for training (more batches but DVS repeats)')

    # Model
    parser.add_argument('--img_shape', default=32, type=int,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--T', default=10, type=int,
                        help='SNN simulation time steps')
    parser.add_argument('--fusion_stages', nargs='+', type=int, default=[4],
                        help='Stages to apply fusion (只在最后一层融合以提高效率)')
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['concat', 'cross_attention'],
                        help='Type of fusion')

    # Training
    parser.add_argument('--epochs', default=200, type=int,
                        help='Training epochs (DVS样本少,需要更多epoch)')
    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['SGD', 'Adam'],
                        help='Optimizer')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=1000,
                        help='Random seed')
    
    # Contrastive Learning
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Enable contrastive learning (default: disabled)')
    parser.add_argument('--contrastive_weight', default=0.3, type=float,
                        help='Weight for contrastive loss')

    # Paths
    parser.add_argument('--log_dir', type=str, 
                        default='/home/user/kpm/kpm/results/ESTL/fusion/log_dir',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/user/kpm/kpm/results/ESTL/fusion/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--rgb_root', type=str,
                        default='/home/user/kpm/kpm//Dataset/CIFAR10/cifar10',
                        help='RGB CIFAR10 dataset root for edge extraction')
    parser.add_argument('--dvs_root', type=str,
                        default='/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat',
                        help='DVS CIFAR10 dataset root')
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU ID')
    
    # Edge pretraining
    parser.add_argument('--pretrained_edge', type=str, default='/home/user/kpm/kpm/results/ESTL/edge-branch/checkpoints/EdgeBranch_Pretrain_lr0.001_seed1000/best_edge_branch.pth',
                        help='Path to pretrained edge branch weights')
    parser.add_argument('--freeze_edge', action='store_true',
                        help='Freeze edge branch parameters')
    
    # Experiment naming
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Custom experiment name prefix (default: auto-generate)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Device
    device = torch.device(f"cuda:{args.GPU_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    seed_all(args.seed)
    
    # Experiment name
    if args.exp_name is not None:
        # 使用自定义实验名前缀
        exp_prefix = args.exp_name
    else:
        # 默认前缀
        exp_prefix = "EdgeGuided"
    
    exp_name = (
        f"{exp_prefix}_{args.data_set}_"
        f"fusion{''.join(map(str, args.fusion_stages))}_{args.fusion_type}_"
        f"freeze{args.freeze_edge}_"
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
    print("Edge-Guided DVS Classification with D2CAF Fusion".center(70))
    print("="*70)
    print(f"Experiment: {exp_name}")
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    if args.data_set == 'CIFAR10':
        # EdgeGuided模式：使用RGB+DVS配对数据
        train_loader, val_loader, test_loader = get_cifar10_DVS_enhanced(
            args.batch_size,
            T=args.T,
            train_set_ratio=args.dvs_sample_ratio,
            encode_type='TET',
            augmentation=args.enable_augmentation,
            rgb_root=args.rgb_root,
            dvs_root=args.dvs_root,
            val_split=0.1,  # 从训练集中划分10%作为验证集
            use_rgb_size=args.train_with_rgb_size  # 新增参数
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print(f"  Data augmentation: {'Enabled' if args.enable_augmentation else 'Disabled'}")
        print(f"  Dataset size based on: {'RGB (DVS repeats)' if args.train_with_rgb_size else 'DVS (RGB cycles)'}")
        print(f"  Mode: RGB+DVS paired (Edge-Guided)")
        print(f"  RGB data: {args.rgb_root}")
        print(f"  DVS data: {args.dvs_root}")
    else:
        raise NotImplementedError(f"Dataset {args.data_set} not supported yet")
    
    # Create models
    print("\nCreating models...")
    
    # EdgeGuided模式：使用边缘引导模型
    model = EdgeGuidedVGGSNN(
        cls_num=args.num_classes,
        img_shape=args.img_shape,
        device=device,
        fusion_stages=args.fusion_stages,
        fusion_type=args.fusion_type
    ).to(device)
    
    # Load pretrained edge branch
    if args.pretrained_edge:
        print(f"\n加载预训练边缘分支权重...")
        print(f"  路径: {args.pretrained_edge}")
        
        checkpoint = torch.load(args.pretrained_edge, map_location=device)
        edge_state_dict = checkpoint['model_state_dict']
        
        # 加载边缘分支权重（去除edge_head，只加载特征提取部分）
        model_dict = model.edge_branch.state_dict()
        pretrained_dict = {}
        
        for k, v in edge_state_dict.items():
            # 跳过edge_head（仅用于预训练）
            if 'edge_head' not in k:
                if k in model_dict:
                    if v.size() == model_dict[k].size():
                        pretrained_dict[k] = v
                    else:
                        print(f"  ⚠ 跳过大小不匹配的参数: {k}")
                else:
                    print(f"  ⚠ 跳过不存在的参数: {k}")
        
        model_dict.update(pretrained_dict)
        model.edge_branch.load_state_dict(model_dict, strict=False)
        
        print(f"  ✓ 已加载 {len(pretrained_dict)} 个预训练参数")
        print(f"  ✓ 预训练epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['best_val_loss']:.4f}")
        
        # 冻结边缘分支
        if args.freeze_edge:
            for param in model.edge_branch.parameters():
                param.requires_grad = False
            print(f"  ✓ 边缘分支已冻结")
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  ✓ 可训练参数: {trainable_params/1e6:.2f}M")
    
    print(f"\n  Edge-Guided Model:")
    print(f"    Fusion stages: {args.fusion_stages}")
    print(f"    Fusion type: {args.fusion_type}")
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
    
    # Trainer (with contrastive learning)
    trainer = EdgeGuidedTrainer(
        args=args,
        device=device,
        writer=writer,
        model=model,
        optimizer=optimizer,
        criterion=TET_loss,
        scheduler=scheduler,
        model_path=checkpoint_dir,
        baseline_model=None,
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight
    )
    
    print(f"\n训练配置:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  对比学习: {'启用' if args.use_contrastive else '禁用'}")
    if args.use_contrastive:
        print(f"  对比学习权重: {args.contrastive_weight}")
    
    # Train
    best_train_acc, best_val_acc = trainer.train(train_loader, val_loader)
    
    # Test (使用独立的测试集)
    test_loss, test_acc, test_acc_top5 = trainer.test(test_loader)
    
    # Summary
    print("\n" + "="*70)
    print("Training Summary".center(70))
    print("="*70)
    print(f"Best Train Acc@1: {best_train_acc*100:.2f}%")
    print(f"Best Val Acc@1:   {best_val_acc*100:.2f}%")
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
        f.write(f"  Fusion stages: {args.fusion_stages}\n")
        f.write(f"  Fusion type: {args.fusion_type}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Time steps (T): {args.T}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Best Train Acc@1: {best_train_acc*100:.2f}%\n")
        f.write(f"  Best Val Acc@1:   {best_val_acc*100:.2f}%\n")
        f.write(f"  Test Acc@1:       {test_acc*100:.2f}%\n")
        f.write(f"  Test Acc@5:       {test_acc_top5*100:.2f}%\n")
    
    print(f"Results saved to: {results_file}")
    
    writer.close()
    print("Done!")


if __name__ == "__main__":
    main()

