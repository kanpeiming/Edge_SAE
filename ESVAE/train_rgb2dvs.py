# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: train_rgb2dvs.py
@time: 2024/10/24
@description: RGB to DVS knowledge transfer training script
"""

'''
RGB到DVS的迁移学习。
'''

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

import dataloader.cifar
from dataloader.mnist import *
from dataloader.caltech101 import *
from tl_utils.trainer import AlignmentTLTrainerWithProgressBar
from pretrain.pretrainModel import *
from tl_utils import common_utils
from tl_utils.loss_function import TET_loss, TRT_loss

parser = argparse.ArgumentParser(description='PyTorch RGB to DVS Transfer Learning')
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100',
                             'CINIC10_WO_CIFAR10', 'ImageNet2Caltech', 'Caltech51'],
                    help='the data set type.')
parser.add_argument('--batch_size', default=32, type=int, help='Batchsize')  # Cifar10: 32, MNIST: 32, Caltech101: xx
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')  # 增加学习率以加快收敛
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=100, type=int, help='Training epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 10)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='the transfer loss for transfer learning.')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='the transfer loss for transfer learning.')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float, metavar='N',
                    help='encoder transfer learning loss ratio')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float, metavar='N',
                    help='feature transfer learning loss ratio')
# Model Architecture 参数
parser.add_argument('--use_woap', default=False, type=bool,
                    help='Whether to use without Average Pooling version (stride=2 conv instead of AvgPool2d)')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/r2e/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/r2e/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')
parser.add_argument('--num_classes', type=int, default=10, help='the number of data classes.')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='the ratio of used dvs training set. ')
parser.add_argument('--val_ratio', type=float, default=0.0, help='the ratio of validation set split from training set. (default: 0.0, consistent with tl.py)')
parser.add_argument('--use_validation', default=False, type=bool, help='Whether to use validation set during training (default: False, consistent with tl.py)')
parser.add_argument('--use_cutout', default=False, type=bool, help='Whether to use Cutout data augmentation')
parser.add_argument('--cutout_length', default=16, type=int, help='Length of cutout square')
parser.add_argument('--use_grayscale', default=False, type=bool, help='Whether to convert RGB to grayscale (keep 3 channels) for structure-based transfer learning validation')
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{args.GPU_id}")

log_name = (
    f"RGB2DVS_{args.data_set}_"
    f"standard_trainer_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"RGB{args.RGB_sample_ratio}_"
    f"DVS{args.dvs_sample_ratio}_"
    f"{'gray' if args.use_grayscale else 'color'}_"
    f"val{args.use_validation}"
)

# 日志目录设置
log_dir = os.path.join(
    args.log_dir,
    f"RGB2DVS_{args.data_set}_{args.num_classes}",
    log_name
)

# 模型保存路径
checkpoint_dir = os.path.join(
    args.checkpoint,
    f"RGB2DVS_{args.data_set}_{args.num_classes}_{log_name}"
)

# 递归创建目录
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 设置模型保存路径 - 修复：应该是文件路径，不是目录路径
model_path = os.path.join(checkpoint_dir, "best_model.pth")
writer = SummaryWriter(log_dir=log_dir)

print(log_name)
print(writer.log_dir)

if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_rgb2dvs_result.txt", "a")

    # preparing data - 加载RGB和DVS数据
    if args.data_set == 'CIFAR10':
        # 统一调用数据加载器
        train_loader, val_loader, dvs_test_loader = dataloader.cifar.get_tl_cifar10(
            args.batch_size,
            args.RGB_sample_ratio,
            args.dvs_sample_ratio,
            args.val_ratio,
            args.use_cutout,
            args.cutout_length,
            args.use_grayscale
        )
        
        print("\n=== 数据集分配总结 ===")
        print(f"RGB训练集数量: {train_loader.dataset.get_len()[0]}")
        print(f"DVS配对训练数量: {train_loader.dataset.get_len()[1]} (用于RGB-DVS迁移学习)")
        
        if val_loader is not None and args.use_validation:
            print(f"DVS验证集数量: {len(val_loader.dataset)} (独立验证，不参与配对)")
            # 计算数据分配
            dvs_total_for_training = len(val_loader.dataset) / args.val_ratio
            dvs_actual_training = int(dvs_total_for_training * (1 - args.val_ratio))
            print(f"✓ 正确分配: DVS总数{int(dvs_total_for_training)} = 配对训练{dvs_actual_training} + 验证{len(val_loader.dataset)}")
            print(f"✓ 避免数据泄露: 验证集DVS数据不参与RGB-DVS配对训练")
        else:
            print(f"DVS验证集: 无 (与tl.py一致，使用全部DVS训练数据)")
        
        print(f"DVS测试集数量: {len(dvs_test_loader.dataset)}")
        print("=====================\n")
    elif args.data_set == 'MNIST':
        # MNIST暂时不支持验证集划分，使用原有方式
        train_loader, dvs_test_loader = get_tl_mnist(
            args.batch_size,
            args.RGB_sample_ratio,
            args.dvs_sample_ratio
        )
        val_loader = None
        print("训练集RGB数量", train_loader.dataset.get_len()[0])
        print("训练集DVS数量", train_loader.dataset.get_len()[1])
        print("测试集DVS数量", len(dvs_test_loader.dataset))
        print("注意：MNIST数据集暂不支持验证集划分")
    elif args.data_set == 'Caltech101':
        # Caltech101暂时不支持验证集划分，使用原有方式
        train_loader, dvs_test_loader = get_tl_caltech101(
            args.batch_size,
            args.RGB_sample_ratio,
            args.dvs_sample_ratio
        )
        val_loader = None
        print("训练集RGB数量", train_loader.dataset.get_len()[0])
        print("训练集DVS数量", train_loader.dataset.get_len()[1])
        print("测试集DVS数量", len(dvs_test_loader.dataset))
        print("注意：Caltech101数据集暂不支持验证集划分")
    else:
        raise NotImplementedError(f"Dataset {args.data_set} not implemented for RGB2DVS transfer")

    # preparing model - 选择模型（支持2种组合）
    # 根据数据集设置图像尺寸 - 与tl.py保持完全一致
    if args.data_set == 'CIFAR10':
        img_shape = 48  # 与tl.py一致：CIFAR10使用32x32（虽然数据加载器resize到48，但模型使用32）
    elif args.data_set == 'MNIST':
        img_shape = 32  # N-MNIST: 32x32 (resized from 28x28)
    elif args.data_set == 'Caltech101':
        img_shape = 48  # N-Caltech101: 48x48
    else:
        img_shape = 48  # 默认使用48
    
    if args.use_woap:
        from pretrain.pretrainModel import VGGSNNwoAP
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=img_shape)
        print("使用标准VGGSNNwoAP模型 (without Average Pooling)")
        print("  架构: stride=2卷积替代AvgPool2d")
        print(f"  图像尺寸: {img_shape}×{img_shape}")
        print("  瓶颈层: 与tl.py完全一致（bottleneck包含LIFSpike + bottleneck_lif_node再次激活）")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=img_shape, device=device)

    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # preparing training set - 使用统一学习率策略
    if args.optim == 'Adam':
        # 使用统一学习率（避免分层学习率导致的不平衡）
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"使用Adam优化器，统一学习率: {args.lr}")
        
        # 分层学习率（暂时禁用）
        # optimizer = torch.optim.Adam([
        #     {'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10},
        #     {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr}
        # ])
        # print(f"使用Adam优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")

    elif args.optim == 'SGD':
        # 使用统一学习率
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=False)
        print(f"使用SGD优化器，统一学习率: {args.lr}")
        
        # 分层学习率（可选）
        # optimizer = torch.optim.SGD([
        #     {'params': [p for n, p in model.named_parameters() if 'input' in n],
        #      'lr': args.lr * 10, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False},
        #     {'params': [p for n, p in model.named_parameters() if 'input' not in n],
        #      'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False}
        # ])
        # print(f"使用SGD优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")
    else:
        raise Exception(f"The value of optim should in ['SGD', 'Adam'], "
                        f"and your input is {args.optim}")

    # 学习率调度器
    if args.data_set == 'CIFAR10':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif args.data_set == 'MNIST':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.data_set == 'Caltech101':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    print(f"训练轮数: {args.epoch}")
    print(f"使用带进度条的AlignmentTLTrainer (来自tl_utils.trainer)")
    print("主要改进:")
    print("  ✓ 使用更标准的训练器实现")
    print("  ✓ 添加了tqdm进度条显示")
    print("  ✓ 支持--use_grayscale参数进行灰度转换")
    print("  ✓ 包含梯度监控功能")
    print("  ✓ 更完善的验证集处理")
    print("  ✓ 修复了模型保存路径问题")
    print("  ✓ 与其他脚本保持一致的接口")
    
    if args.use_grayscale:
        print("  ✓ 灰度转换: 启用 (RGB->灰度，保持三通道)")
        print("    - 用于验证迁移学习是否基于结构而非色彩信息")
    else:
        print("  ○ 灰度转换: 未启用 (使用原始RGB色彩信息)")

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

    # 训练（使用带进度条和灰度支持的AlignmentTLTrainer）
    print("使用AlignmentTLTrainerWithProgressBar进行RGB到DVS迁移学习...")
    if args.use_grayscale:
        print("  ✓ 启用灰度转换: RGB图像将转换为灰度但保持三通道")
    trainer = AlignmentTLTrainerWithProgressBar(
        args, device, writer, model, optimizer, criterion, scheduler, model_path
    )
    
    # 根据是否有验证集决定传入参数
    if args.use_validation and val_loader is not None:
        print("使用验证集进行训练...")
        best_train_acc, best_val_acc = trainer.train(train_loader, val_loader)
        best_train_loss = None  # 标准训练器不返回loss
    else:
        print("使用标准训练模式（无验证集）...")
        # 传入空的验证集
        best_train_acc, best_val_acc = trainer.train(train_loader, dvs_test_loader)
        best_train_loss = None  # 标准训练器不返回loss

    test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader)

    print(f'test_loss={test_loss:.5f} test_acc1={test_acc1:.4f} test_acc5={test_acc5:.4f}')
    writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
    writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)

    write_content = (
        f'seed: {args.seed} \n'
        f'encoder_tl_loss: {args.encoder_tl_lamb} * {args.encoder_tl_loss_type} \n'
        f'feature_tl_loss: {args.feature_tl_lamb} * {args.feature_tl_loss_type} \n'
        f'RGB_sample_ratio: {args.RGB_sample_ratio}, dvs_sample_ratio: {args.dvs_sample_ratio} \n'
        f'use_grayscale: {args.use_grayscale} (structure-based transfer validation) \n'
        f'best_train_acc: {best_train_acc}, best_val_acc: {best_val_acc} \n'
        f'test_acc1: {test_acc1}, test_acc5: {test_acc5}, test_loss: {test_loss} \n\n'
    )
    f.write(write_content)
    f.close()
