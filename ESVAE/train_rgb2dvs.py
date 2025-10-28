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
from pretrain.pretrainer import *
from pretrain.validation_trainer import AlignmentTLTrainer_RGB2DVS_WithValidation
from pretrain.pretrainModel import *
from tl_utils import common_utils
from tl_utils.loss_function import TET_loss

parser = argparse.ArgumentParser(description='PyTorch RGB to DVS Transfer Learning')
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100',
                             'CINIC10_WO_CIFAR10', 'ImageNet2Caltech', 'Caltech51'],
                    help='the data set type.')
parser.add_argument('--batch_size', default=64, type=int, help='Batchsize')  # Cifar10: 32, MNIST: 32, Caltech101: xx
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')  # CIFAR10: 0.0002, Caltech101: 0.0002, MNIST: 0.0001
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=80, type=int, help='Training epochs')
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
# Regional Focus 参数 - 多层级区域关注
parser.add_argument('--use_regional_focus', default=False, type=bool,
                    help='Whether to use multi-layer regional focus module')
parser.add_argument('--regional_similarity', default='cosine', type=str, choices=['cosine', 'l2', 'dot'],
                    help='Similarity type for regional focus')
parser.add_argument('--regional_alpha', default=0.5, type=float,
                    help='Weight for multi-layer regional focus loss (default: 0.5)')
parser.add_argument('--regional_beta', default=0.01, type=float,
                    help='Beta for regional focus regularization (deprecated, kept for compatibility)')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/r2e/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/r2e/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')
parser.add_argument('--num_classes', type=int, default=10, help='the number of data classes.')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='the ratio of used dvs training set. ')
parser.add_argument('--val_ratio', type=float, default=0.1, help='the ratio of validation set split from training set.')
parser.add_argument('--use_validation', default=True, type=bool, help='Whether to use validation set during training')
parser.add_argument('--use_cutout', default=False, type=bool, help='Whether to use Cutout data augmentation')
parser.add_argument('--cutout_length', default=16, type=int, help='Length of cutout square')

args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{args.GPU_id}")

log_name = (
    f"RGB2DVS_{args.data_set}_"
    f"{'MLR-' + args.regional_similarity + f'-{args.regional_alpha}' if args.use_regional_focus else 'baseline'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"RGB{args.RGB_sample_ratio}_"
    f"DVS{args.dvs_sample_ratio}"
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

# 设置模型保存路径
model_path = checkpoint_dir
writer = SummaryWriter(log_dir=log_dir)

print(log_name)
print(writer.log_dir)

# 打印改进信息
print("\n=== 训练改进策略 ===")
print(f"✓ 图像分辨率: 32×32 (保持原始尺寸，避免瓶颈层维度问题)")
print(f"✓ AutoAugment: CIFAR10Policy (25种子策略)")
print(f"✓ 分层学习率: 输入层×10, 其他层×1")
if args.use_cutout:
    print(f"✓ Cutout增强: 启用 (长度={args.cutout_length})")
else:
    print("○ Cutout增强: 未启用")
print("===================\n")

if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_rgb2dvs_result.txt", "a")

    # preparing data - 加载RGB和DVS数据
    if args.data_set == 'CIFAR10':
        if args.use_validation:
            train_loader, val_loader, dvs_test_loader = dataloader.cifar.get_tl_cifar10(
                args.batch_size, 
                args.RGB_sample_ratio, 
                args.dvs_sample_ratio,
                args.val_ratio,
                args.use_cutout,
                args.cutout_length
            )
            print("训练集RGB数量", train_loader.dataset.get_len()[0])
            print("训练集DVS数量", train_loader.dataset.get_len()[1])
            print("验证集DVS数量", len(val_loader.dataset))
            print("测试集DVS数量", len(dvs_test_loader.dataset))
        else:
            # 兼容原有的无验证集模式
            train_loader, _, dvs_test_loader = dataloader.cifar.get_tl_cifar10(
                args.batch_size, 
                args.RGB_sample_ratio, 
                args.dvs_sample_ratio,
                0.0,  # 不划分验证集
                args.use_cutout,
                args.cutout_length
            )
            val_loader = None
            print("训练集RGB数量", train_loader.dataset.get_len()[0])
            print("训练集DVS数量", train_loader.dataset.get_len()[1])
            print("测试集DVS数量", len(dvs_test_loader.dataset))
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

    # preparing model - 选择模型
    if args.use_regional_focus:
        import sys
        import os
        # 添加Regional-Focus目录到路径
        regional_focus_path = os.path.join(os.path.dirname(__file__), 'Regional-Focus')
        if regional_focus_path not in sys.path:
            sys.path.insert(0, regional_focus_path)
        from enhanced_model import VGGSNN_RegionalFocus
        
        # 多层级区域关注配置
        regional_focus_config = {
            'similarity_type': args.regional_similarity,
            'weight_constraint': 'softmax',
            'alpha': args.regional_alpha,  # 区域关注损失权重
        }
        
        model = VGGSNN_RegionalFocus(
            cls_num=args.num_classes, 
            img_shape=32, 
            device=device,
            use_regional_focus=True,
            regional_focus_config=regional_focus_config
        )
        print(f"使用多层级区域关注模块:")
        print(f"  相似度类型: {args.regional_similarity}")
        print(f"  区域关注权重: {args.regional_alpha}")
        print(f"  DVS输入: 2通道→3通道自动转换")
        print(f"  约束层级: 6层 (输入层→Pool1→Pool2→Pool3→Pool4→瓶颈层)")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=32, device=device)
        print("使用标准VGGSNN模型")

    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # preparing training set - 使用分层学习率策略
    if args.optim == 'Adam':
        # Adam优化器使用分层学习率
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10},
            {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr}
        ])
        print(f"使用Adam优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")
    elif args.optim == 'SGD':
        # SGD优化器使用分层学习率
        optimizer = torch.optim.SGD([
            {'params': [p for n, p in model.named_parameters() if 'input' in n], 
             'lr': args.lr * 10, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False},
            {'params': [p for n, p in model.named_parameters() if 'input' not in n], 
             'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False}
        ])
        print(f"使用SGD优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")
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

    # 训练（使用RGB到DVS的迁移学习训练器）
    if args.use_validation and val_loader is not None:
        # 使用带验证集的训练器
        print("使用验证集进行训练...")
        trainer = AlignmentTLTrainer_RGB2DVS_WithValidation(
            args, device, writer, model, optimizer, TET_loss, scheduler, model_path
        )
        best_train_acc, best_train_loss = trainer.train_with_validation(train_loader, val_loader)
    else:
        # 使用标准训练器
        print("使用标准训练模式（无验证集）...")
        trainer = AlignmentTLTrainer_RGB2DVS(
            args, device, writer, model, optimizer, TET_loss, scheduler, model_path
        )
        best_train_acc, best_train_loss = trainer.train(train_loader)
    
    test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader)

    print(f'test_loss={test_loss:.5f} test_acc1={test_acc1:.3f} test_acc5={test_acc5:.4f}')
    writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
    writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)

    write_content = (
        f'seed: {args.seed} \n'
        f'encoder_tl_loss: {args.encoder_tl_lamb} * {args.encoder_tl_loss_type} \n'
        f'feature_tl_loss: {args.feature_tl_lamb} * {args.feature_tl_loss_type} \n'
        f'RGB_sample_ratio: {args.RGB_sample_ratio}, dvs_sample_ratio: {args.dvs_sample_ratio} \n'
        f'best_train_acc: {best_train_acc}, best_train_loss: {best_train_loss} \n'
        f'test_acc1: {test_acc1}, test_acc5: {test_acc5}, test_loss: {test_loss} \n\n'
    )
    f.write(write_content)
    f.close()

