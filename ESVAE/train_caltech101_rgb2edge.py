# -*- coding: utf-8 -*-
"""
Caltech101 RGB to Edge knowledge transfer pretraining script
RGB到边缘信息的迁移学习预训练脚本

功能：
- 使用RGB数据作为源域，RGB边缘信息作为目标域进行预训练
- 使用Sobel+Canny双边缘提取器生成2通道边缘图
- 保存预训练参数供后续DVS微调使用

使用方法：
1. 预训练RGB->Edge模型：
   python train_caltech101_rgb2edge.py --epochs 50 --lr 0.001 --batch_size 32

2. 使用预训练参数进行DVS微调：
   python train_caltech101_baseline.py --pretrained_path /path/to/rgb_edge_pretrained_best.pth
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
from pretrain.pretrainer import AlignmentTLTrainer_Edge_1
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP
from pretrain.Edge import SobelEdgeExtractionModule, CannyEdgeDetectionModule
from tl_utils.loss_function import TET_loss, TRT_loss
from tl_utils import common_utils

parser = argparse.ArgumentParser(description='Caltech101 RGB->Edge Pretraining')
parser.add_argument('--batch_size', default=32, type=int, help='Batchsize')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for RGB->Edge pretraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=30, type=int, help='RGB->Edge pretraining epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 10)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training.')
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='the transfer loss for encoder.')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='the transfer loss for features.')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float,
                    help='encoder transfer learning loss ratio')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float,
                    help='feature transfer learning loss ratio')
parser.add_argument('--use_woap', default=False, type=bool,
                    help='Whether to use without Average Pooling version')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/pretrain/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set.')
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

# 固定Caltech101参数
args.data_set = 'Caltech101'
args.num_classes = 101

device = torch.device(f"cuda:{args.GPU_id}")

# 生成日志名称
log_name = (
    f"Caltech101_RGB2Edge_Pretrain_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"RGB{args.RGB_sample_ratio}_"
    f"TWoSobelEdge_"  # 标记使用了Sobel+Canny双边缘提取器
    f"trt{args.use_trt}"
)

# 日志目录设置
log_dir = os.path.join(
    args.log_dir,
    f"Caltech101_EdgePretrain_{args.num_classes}",
    log_name
)

# 模型保存路径
checkpoint_dir = os.path.join(
    args.checkpoint,
    f"Caltech101_EdgePretrain_{args.num_classes}_{log_name}"
)

# 递归创建目录
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 设置模型保存路径
model_path = checkpoint_dir
writer = SummaryWriter(log_dir=log_dir)

print(f"训练配置: {log_name}")
print(f"日志目录: {writer.log_dir}")
print(f"模型保存: {model_path}")

if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"Caltech101_{args.seed}_rgb2edge_pretrain_result.txt", "a")

    print("\n" + "="*80)
    print("RGB->Edge预训练 (使用Sobel/Canny边缘检测)")
    print("="*80)
    
    # 准备RGB数据 (用于RGB->Edge预训练)
    print("Loading Caltech101 RGB dataset for RGB->Edge pretraining...")
    rgb_train_loader, rgb_test_loader = get_caltech101(
        args.batch_size, 
        args.RGB_sample_ratio
    )
    
    print(f"\n=== RGB->Edge预训练数据集信息 ===")
    print(f"RGB训练集数量: {len(rgb_train_loader.dataset)}")
    print(f"RGB测试集数量: {len(rgb_test_loader.dataset)}")
    print(f"类别数量: {args.num_classes}")
    print(f"训练模式: RGB作为源域 -> RGB边缘信息作为目标域")
    print("===========================\n")

    # 准备模型 - 选择标准VGGSNN模型
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=48)  # Caltech101使用48x48
        print("使用VGGSNNwoAP模型 (without Average Pooling)")
        print("  架构: stride=2卷积替代AvgPool2d")
        print("  图像尺寸: 48×48")
        print("  输入通道: RGB=3通道, Edge=2通道(Sobel+Canny)")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=48, device=device)  # Caltech101使用48x48
        print("使用标准VGGSNN模型 (with Average Pooling)")
        print("  架构: AvgPool2d下采样")
        print("  图像尺寸: 48×48")
        print("  输入通道: RGB=3通道, Edge=2通道(Sobel+Canny)")

    # 为模型添加边缘提取器
    model.edge_extractor1 = SobelEdgeExtractionModule(device=device, in_channels=3)
    model.edge_extractor2 = CannyEdgeDetectionModule(device=device, in_channels=3)
    
    print("✓ 已添加双边缘提取器:")
    print("  - edge_extractor1: Sobel边缘检测 (输出1通道)")
    print("  - edge_extractor2: Canny边缘检测 (输出1通道)")
    print("  - 叠加后: 2通道边缘图，近似DVS双通道特性")

    if args.parallel and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 准备RGB->Edge预训练优化器
    if args.optim == 'Adam':
        # 分层学习率：输入层使用更高的学习率
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10},
            {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr}
        ])
        print(f"RGB->Edge预训练使用Adam优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10,
             'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False},
            {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr * 1,
             'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False}
        ])
        print(f"RGB->Edge预训练使用SGD优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")

    # RGB->Edge预训练学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n迁移学习配置:")
    print(f"  编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}")
    print(f"  特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}")
    print(f"  编码器类型: {args.encoder_type}")
    print(f"  边缘提取: Sobel + Canny双算法叠加")
    
    # 选择损失函数：TRT或TET
    if args.use_trt:
        print(f"\n使用TRT (Temporal Regularization Training) Loss")
        print(f"  - TRT decay (δ): {args.trt_decay}")
        print(f"  - TRT lambda (λ): {args.trt_lambda}")
        print(f"  - TRT epsilon (ε): {args.trt_epsilon}")
        print(f"  - TRT eta (η): {args.trt_eta}")
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
        print(f"\n使用TET (Temporal Efficient Training) Loss")
        criterion = TET_loss
    
    # RGB->Edge预训练 (使用AlignmentTLTrainer_Edge_1)
    print("\n开始RGB->Edge预训练...")
    trainer = AlignmentTLTrainer_Edge_1(
        args, device, writer, model, optimizer, criterion, scheduler, 
        os.path.join(model_path, "rgb_edge_pretrained.pth")
    )
    
    best_train_acc, best_train_loss = trainer.train(rgb_train_loader)
    
    # RGB->Edge预训练测试
    test_loss, test_acc1, test_acc5 = trainer.test(rgb_test_loader)
    print(f'\nRGB->Edge预训练结果: test_loss={test_loss:.5f} test_acc1={test_acc1:.4f} test_acc5={test_acc5:.4f}')
    
    # 保存RGB->Edge预训练模型
    pretrained_path = os.path.join(model_path, "rgb_edge_pretrained_best.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc1': test_acc1,
        'test_acc5': test_acc5,
        'test_loss': test_loss,
        'args': args
    }, pretrained_path)
    print(f"RGB->Edge预训练模型已保存到: {pretrained_path}")

    # 记录测试结果到TensorBoard
    writer.add_scalar(tag="final/rgb_edge_accuracy", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="final/rgb_edge_loss", scalar_value=test_loss, global_step=0)

    # 保存结果到文件
    write_content = (
        f'=== Caltech101 RGB->Edge预训练 结果 ===\n'
        f'种子: {args.seed}\n'
        f'边缘提取器: Sobel + Canny双算法 (2通道输出)\n'
        f'模型: {"VGGSNNwoAP" if args.use_woap else "VGGSNN"}\n'
        f'预训练epochs: {args.epochs}, 学习率: {args.lr}\n'
        f'编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}\n'
        f'特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}\n'
        f'RGB样本比例: {args.RGB_sample_ratio}\n'
        f'RGB->Edge预训练准确率: {test_acc1:.4f}%\n'
        f'预训练模型保存路径: {pretrained_path}\n'
        f'=====================================\n\n'
    )
    f.write(write_content)
    f.close()
    
    writer.close()
    print(f"\n预训练完成！模型已保存到: {pretrained_path}")
    print(f"结果已记录到: Caltech101_{args.seed}_rgb2edge_pretrain_result.txt")
    print(f"\n使用预训练参数进行DVS微调:")
    print(f"python train_caltech101_baseline.py --pretrained_path {pretrained_path}")
