# -*- coding: utf-8 -*-
"""
@file: train_caltech101_rgb2dvs.py
@description: Caltech101 RGB to DVS knowledge transfer training script
RGB到DVS的迁移学习训练脚本 - Caltech101数据集

功能：
- 使用Caltech101 RGB数据作为源域，N-Caltech101 DVS数据作为目标域进行迁移学习
- 支持灰度转换验证结构迁移假设
- 使用AlignmentTLTrainerWithProgressBar训练器，带进度条显示

使用方法：
1. 标准训练（使用全部数据）：
   python train_caltech101_rgb2dvs.py --batch_size 32 --lr 0.001 --epoch 100

2. 使用部分数据训练：
   python train_caltech101_rgb2dvs.py --RGB_sample_ratio 0.5 --dvs_sample_ratio 0.5

3. 启用灰度转换验证结构迁移：
   python train_caltech101_rgb2dvs.py --use_grayscale True

4. 自定义GPU和路径：
   python train_caltech101_rgb2dvs.py --GPU_id 1 --log_dir /path/to/logs --checkpoint /path/to/checkpoints
"""

import argparse
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

# 添加ESVAE根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
esvae_root = os.path.dirname(current_dir)  # 上一级目录
if esvae_root not in sys.path:
    sys.path.insert(0, esvae_root)

from dataloader.caltech101 import get_tl_caltech101
from tl_utils.caltech101_trainer import AlignmentTLTrainerWithDVSWeight
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP
from tl_utils import common_utils
from tl_utils.loss_function import TET_loss, TRT_loss

parser = argparse.ArgumentParser(description='Caltech101 RGB to DVS Transfer Learning')
parser.add_argument('--data_set', type=str, default='Caltech101',
                    help='Dataset name (fixed to Caltech101)')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=100, type=int, help='Training epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='SNN simulation time (default: 10)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='Encoder type for RGB data to SNN')
parser.add_argument('--seed', type=int, default=1000, help='Random seed for initialization')
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='Transfer loss type for encoder')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='Transfer loss type for features')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float,
                    help='Encoder transfer learning loss weight')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float,
                    help='Feature transfer learning loss weight')
parser.add_argument('--use_woap', default=False, type=bool,
                    help='Whether to use without Average Pooling version (stride=2 conv instead of AvgPool2d)')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/r2d_caltech101/log_dir',
                    help='Path to tensorboard log directory')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/r2d_caltech101/checkpoints',
                    help='Path to checkpoint directory')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID to use')
parser.add_argument('--num_classes', type=int, default=101, help='Number of classes (Caltech101 has 101 classes)')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, 
                    help='Ratio of RGB training set to use')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, 
                    help='Ratio of DVS training set to use')
parser.add_argument('--use_grayscale', default=False, type=bool, 
                    help='Whether to convert RGB to grayscale (keep 3 channels) for structure-based transfer validation')
# TRT (Temporal Regularization Training) 参数
parser.add_argument('--use_trt', action='store_true', default=False,
                    help='Whether to use TRT (Temporal Regularization Training) loss')
parser.add_argument('--trt_decay', type=float, default=0.5,
                    help='TRT decay factor δ (default: 0.5)')
parser.add_argument('--trt_lambda', type=float, default=1e-5,
                    help='TRT regularization coefficient λ (default: 1e-5)')
parser.add_argument('--trt_epsilon', type=float, default=1e-5,
                    help='TRT epsilon ε (default: 1e-5)')
parser.add_argument('--trt_eta', type=float, default=0.05,
                    help='TRT eta η (MSE loss weight, default: 0.05)')

args = parser.parse_args()

# 固定数据集为Caltech101
args.data_set = 'Caltech101'

# 设备配置
device = torch.device(f"cuda:{args.GPU_id}")

# 生成日志名称
log_name = (
    f"RGB2DVS_Caltech101_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"RGB{args.RGB_sample_ratio}_"
    f"DVS{args.dvs_sample_ratio}_"
    f"{'gray' if args.use_grayscale else 'color'}"
)

# 日志目录设置
log_dir = os.path.join(
    args.log_dir,
    f"RGB2DVS_Caltech101_{args.num_classes}",
    log_name
)

# 模型保存路径
checkpoint_dir = os.path.join(
    args.checkpoint,
    f"RGB2DVS_Caltech101_{args.num_classes}_{log_name}"
)

# 递归创建目录
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 设置模型保存路径 - 文件路径，不是目录路径
model_path = os.path.join(checkpoint_dir, "best_model.pth")
writer = SummaryWriter(log_dir=log_dir)

print(f"\n训练配置: {log_name}")
print(f"日志目录: {writer.log_dir}")
print(f"模型保存: {model_path}\n")

if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"Caltech101_{args.seed}_rgb2dvs_result.txt", "a")

    # 准备数据
    print("加载数据集...")
    train_loader, dvs_test_loader = get_tl_caltech101(
        args.batch_size,
        args.RGB_sample_ratio,
        args.dvs_sample_ratio
    )
    
    print(f"RGB训练集: {train_loader.dataset.get_len()[0]} 样本")
    print(f"DVS训练集: {train_loader.dataset.get_len()[1]} 样本")
    print(f"DVS测试集: {len(dvs_test_loader.dataset)} 样本\n")

    # 准备模型
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=48)
        print("模型: VGGSNNwoAP")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=48, device=device)
        print("模型: VGGSNN")

    if args.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 准备优化器
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=False)
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    print(f"优化器: {args.optim}, 学习率: {args.lr}, Epoch: {args.epoch}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 选择损失函数：TRT或TET
    if args.use_trt:
        print(f"使用TRT (Temporal Regularization Training) Loss")
        print(f"  - TRT decay (δ): {args.trt_decay}")
        print(f"  - TRT lambda (λ): {args.trt_lambda}")
        print(f"  - TRT epsilon (ε): {args.trt_epsilon}")
        print(f"  - TRT eta (η): {args.trt_eta}\n")
        # 创建TRT loss函数的wrapper
        criterion = lambda outputs, labels: TRT_loss(
            model, outputs, labels, 
            criterion=torch.nn.CrossEntropyLoss(),
            decay=args.trt_decay, 
            lamb=args.trt_lambda, 
            epsilon=args.trt_epsilon, 
            eta=args.trt_eta
        )
    else:
        print(f"使用TET (Temporal Efficient Training) Loss\n")
        criterion = TET_loss
    
    trainer = AlignmentTLTrainerWithDVSWeight(
        args, device, writer, model, optimizer, criterion, scheduler, model_path
    )
    
    print("开始训练...\n")
    best_train_acc, best_val_acc = trainer.train(train_loader, dvs_test_loader)

    # 测试
    print("\n最终测试...")
    test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader)

    print(f'test_acc1={test_acc1:.4f} ({test_acc1*100:.2f}%), test_acc5={test_acc5:.4f} ({test_acc5*100:.2f}%), test_loss={test_loss:.5f}')
    
    writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
    writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)

    # 保存结果到文件
    write_content = (
        f'=== Caltech101 RGB->DVS迁移学习 结果 ===\n'
        f'种子: {args.seed}\n'
        f'数据集: Caltech101 (RGB) -> N-Caltech101 (DVS)\n'
        f'类别数: {args.num_classes}\n'
        f'模型: {"VGGSNNwoAP" if args.use_woap else "VGGSNN"}\n'
        f'训练epochs: {args.epoch}, 学习率: {args.lr}\n'
        f'编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}\n'
        f'特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}\n'
        f'RGB样本比例: {args.RGB_sample_ratio}, DVS样本比例: {args.dvs_sample_ratio}\n'
        f'use_grayscale: {args.use_grayscale} (结构迁移验证)\n'
        f'best_train_acc: {best_train_acc}, best_val_acc: {best_val_acc}\n'
        f'test_acc1: {test_acc1:.4f} ({test_acc1*100:.2f}%), '
        f'test_acc5: {test_acc5:.4f} ({test_acc5*100:.2f}%), '
        f'test_loss: {test_loss:.5f}\n'
        f'模型保存路径: {model_path}\n'
        f'========================================\n\n'
    )
    f.write(write_content)
    f.close()
    
    writer.close()
    print(f"\n训练完成！")
    print(f"模型已保存到: {model_path}")
    print(f"结果已记录到: Caltech101_{args.seed}_rgb2dvs_result.txt")
    print(f"TensorBoard日志: {writer.log_dir}")

