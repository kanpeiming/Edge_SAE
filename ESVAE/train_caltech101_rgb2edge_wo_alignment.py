# -*- coding: utf-8 -*-
"""
Caltech101 RGB-only pretraining script (without alignment)
仅使用RGB数据预训练脚本（不包含对齐函数）

消融实验（1）：预训练仅用RGB，收敛后迁移参数至DVS训练阶段（不包含dvs_input层）
目的：验证对齐函数的有效性
预期：低于SDSTL的效果

使用方法：
1. 预训练RGB-only模型：
   python train_caltech101_rgb2edge_wo_alignment.py --epochs 50 --lr 0.001 --batch_size 32

2. 使用预训练参数进行DVS微调：
   python train_caltech101_baseline.py --pretrained_path /path/to/rgb_only_pretrained_best.pth
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

from dataloader.caltech101 import get_caltech101
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP  # 使用支持RGB和DVS双输入的模型
from pretrain.rgb_only_trainer import RGBOnlyTrainer  # RGB-only训练器
from tl_utils.loss_function import TET_loss
from tl_utils import common_utils
from tl_utils.common_utils import TimeEncoder

parser = argparse.ArgumentParser(description='Caltech101 RGB-only Pretraining (Ablation Study 1)')
parser.add_argument('--batch_size', default=32, type=int, help='Batchsize')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for RGB-only pretraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=50, type=int, help='RGB-only pretraining epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 10)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training.')
parser.add_argument('--use_woap', default=False, type=bool,
                    help='Whether to use without Average Pooling version')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/wo/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/wo/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set.')

args = parser.parse_args()

# 固定Caltech101参数
args.data_set = 'Caltech101'
args.num_classes = 101

device = torch.device(f"cuda:{args.GPU_id}")

# 生成日志名称
log_name = (
    f"Caltech101_RGB_Only_Pretrain_wo_Alignment_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"RGB{args.RGB_sample_ratio}"
)

# 日志目录设置
log_dir = os.path.join(
    args.log_dir,
    f"Caltech101_RGB_Only_{args.num_classes}",
    log_name
)

# 模型保存路径
checkpoint_dir = os.path.join(
    args.checkpoint,
    f"Caltech101_RGB_Only_{args.num_classes}_{log_name}"
)

# 递归创建目录
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 设置模型保存路径
model_path = os.path.join(checkpoint_dir, "rgb_only_pretrained.pth")
writer = SummaryWriter(log_dir=log_dir)

print(f"训练配置: {log_name}")
print(f"日志目录: {writer.log_dir}")
print(f"模型保存: {checkpoint_dir}")

if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"Caltech101_{args.seed}_rgb_only_pretrain_result.txt", "a")

    print("\n" + "="*80)
    print("RGB-only预训练 (消融实验1: 无对齐函数)")
    print("="*80)
    
    # 准备RGB数据 (用于RGB-only预训练)
    print("Loading Caltech101 RGB dataset for RGB-only pretraining...")
    rgb_train_loader, rgb_test_loader = get_caltech101(
        args.batch_size, 
        args.RGB_sample_ratio
    )
    
    print(f"\n=== RGB-only预训练数据集信息 ===")
    print(f"RGB训练集数量: {len(rgb_train_loader.dataset)}")
    print(f"RGB测试集数量: {len(rgb_test_loader.dataset)}")
    print(f"类别数量: {args.num_classes}")
    print(f"训练模式: 仅使用RGB数据，无对齐函数")
    print("===========================\n")

    # 准备模型 - 使用支持RGB/DVS双输入的预训练模型
    # 预训练时使用 rgb_input 层，迁移时只迁移 features/bottleneck/classifier 参数
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=48)  # Caltech101使用48x48
        print("使用VGGSNNwoAP模型 (without Average Pooling)")
        print("  架构: stride=2卷积替代AvgPool2d")
        print("  图像尺寸: 48×48")
        print("  输入通道: RGB=3通道 (使用rgb_input层)")
        print("  迁移策略: 预训练后只迁移features/bottleneck/classifier，不迁移rgb_input")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=48, device=device)  # Caltech101使用48x48
        print("使用标准VGGSNN模型 (with Average Pooling)")
        print("  架构: AvgPool2d下采样")
        print("  图像尺寸: 48×48")
        print("  输入通道: RGB=3通道 (使用rgb_input层)")
        print("  迁移策略: 预训练后只迁移features/bottleneck/classifier，不迁移rgb_input")

    print("✓ RGB-only模式: 使用rgb_input层训练，后续只迁移共享特征层")

    if args.parallel and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 准备RGB-only预训练优化器
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"RGB-only预训练使用Adam优化器，学习率: {args.lr}")
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                                   weight_decay=args.weight_decay, nesterov=False)
        print(f"RGB-only预训练使用SGD优化器，学习率: {args.lr}")
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")

    # RGB-only预训练学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch)

    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n训练配置:")
    print(f"  编码器类型: {args.encoder_type}")
    print(f"  训练模式: RGB-only (无对齐函数，使用rgb_input层)")
    print(f"  预期效果: 低于SDSTL，验证对齐函数的有效性")
    
    # 使用TET损失函数
    print(f"\n使用TET (Temporal Efficient Training) Loss")
    criterion = TET_loss
    
    # 准备编码器 (TimeEncoder将RGB复制T次)
    encoder = TimeEncoder(args.T, device)
    
    # 创建训练器并开始训练
    trainer = RGBOnlyTrainer(
        args=args,
        device=device,
        writer=writer,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        encoder=encoder,
        save_dir=checkpoint_dir
    )
    
    best_train_acc, best_test_acc = trainer.train(rgb_train_loader, rgb_test_loader)
    
    # 最佳模型已在训练循环中保存
    pretrained_path = os.path.join(checkpoint_dir, "rgb_only_pretrained_best.pth")
    print(f"\nRGB-only预训练模型已保存到: {pretrained_path}")
    
    # 记录最终结果到TensorBoard
    writer.add_scalar(tag="final/best_train_accuracy", scalar_value=best_train_acc, global_step=0)
    writer.add_scalar(tag="final/best_test_accuracy", scalar_value=best_test_acc, global_step=0)

    # 保存结果到文件
    write_content = (
        f'=== Caltech101 RGB-only预训练 结果 (消融实验1) ===\n'
        f'种子: {args.seed}\n'
        f'训练模式: RGB-only (无对齐函数，无边缘提取)\n'
        f'模型: {"VGGSNNwoAP" if args.use_woap else "VGGSNN"}\n'
        f'预训练epochs: {args.epoch}, 学习率: {args.lr}\n'
        f'RGB样本比例: {args.RGB_sample_ratio}\n'
        f'最佳训练准确率: {best_train_acc:.4f}\n'
        f'最佳测试准确率: {best_test_acc:.4f}\n'
        f'预训练模型保存路径: {pretrained_path}\n'
        f'训练方式: 使用rgb_input层(3通道)，迁移时只迁移共享层参数\n'
        f'实验目的: 验证对齐函数的有效性\n'
        f'预期: 低于SDSTL效果\n'
        f'=====================================\n\n'
    )
    f.write(write_content)
    f.close()
    
    writer.close()
    print(f"\n预训练完成！模型已保存到: {pretrained_path}")
    print(f"结果已记录到: Caltech101_{args.seed}_rgb_only_pretrain_result.txt")
    print(f"\n使用预训练参数进行DVS微调:")
    print(f"python train_caltech101_baseline.py --pretrained_path {pretrained_path}")

