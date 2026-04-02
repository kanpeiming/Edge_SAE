# -*- coding: utf-8 -*-
"""
Edge to DVS knowledge transfer training script
边缘图到DVS的迁移学习训练脚本

功能：
- 加载RGB->Edge预训练参数
- 使用预处理的Edge数据（2通道）作为源域
- 使用DVS数据（2通道）作为目标域进行迁移学习
- 支持Caltech101、CIFAR10和CEP-DVS数据集

使用方法：
python train_edge2dvs.py --data_set Caltech101 --pretrained_path /path/to/rgb_edge_pretrained_best.pth --epochs 100
python train_edge2dvs.py --data_set CEP-DVS --pretrained_path /path/to/rgb_edge_pretrained_best.pth --epochs 100
"""

import argparse
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

# 添加ESVAE根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
esvae_root = os.path.dirname(current_dir)
if esvae_root not in sys.path:
    sys.path.insert(0, esvae_root)

from dataloader.caltech101 import get_edge2dvs_caltech101
from dataloader.cifar import get_edge2dvs_cifar10
from dataloader.cepdvs import get_edge2dvs_cepdvs
from pretrain.edge2dvs_trainer import AlignmentTLTrainer_Edge2DVS
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP
from tl_utils import common_utils
from tl_utils.loss_function import TET_loss

parser = argparse.ArgumentParser(description='Edge to DVS Transfer Learning')
parser.add_argument('--data_set', type=str, default='Caltech101', 
                    choices=['Caltech101', 'CIFAR10', 'CEP-DVS'],
                    help='Dataset name (Caltech101, CIFAR10, or CEP-DVS)')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=100, type=int, help='Training epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='SNN simulation time')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='Encoder type for edge data')
parser.add_argument('--seed', type=int, default=1000, help='Random seed')
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='Transfer loss for encoder')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='Transfer loss for features')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float, help='Encoder transfer loss ratio')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float, help='Feature transfer loss ratio')
parser.add_argument('--use_woap', default=False, type=bool, help='Use VGGSNNwoAP')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/edge2dvs/log_dir',
                    help='Tensorboard log directory')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/edge2dvs/checkpoints',
                    help='Checkpoint directory')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_classes', type=int, default=None, help='Number of classes (default: auto-detect from dataset)')
parser.add_argument('--edge_sample_ratio', type=float, default=1.0, help='Edge training set ratio')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='DVS training set ratio')
parser.add_argument('--img_size', type=int, default=48, help='Image size')
parser.add_argument('--pretrained_path', type=str, default='', help='Path to RGB->Edge pretrained model')
parser.add_argument('--edge_root', type=str, 
                    default='',
                    help='Edge data root directory')
parser.add_argument('--dvs_root', type=str,
                    default='',
                    help='DVS data root directory')

args = parser.parse_args()

# 根据数据集自动设置默认路径和类别数
if args.data_set == 'Caltech101':
    if args.num_classes is None:
        args.num_classes = 101
    if not args.edge_root:
        args.edge_root = '/home/user/kpm/kpm/Dataset/Caltech101/caltech101_edge'
    if not args.dvs_root:
        args.dvs_root = '/home/user/kpm/kpm/Dataset/Caltech101/NCALTECH101/NCALTECH101/Caltech101'
    if not args.pretrained_path:
        args.pretrained_path = '/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/Caltech101_EdgePretrain_101_Caltech101_RGB2Edge_Pretrain_AP_enc-time_encoder_opt-Adam_lr0.001_T10_seed1000_RGB1.0_TWoSobelEdge_img_shape48/rgb_edge_pretrained_best.pth'
elif args.data_set == 'CIFAR10':
    if args.num_classes is None:
        args.num_classes = 10
    if not args.edge_root:
        args.edge_root = '/home/user/kpm/kpm/Dataset/CIFAR10/cifar10_edge'
    if not args.dvs_root:
        args.dvs_root = '/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat'
    if not args.pretrained_path:
        args.pretrained_path = '/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/CIFAR10_10_Feature-Alignment_CIFAR10_enc-time_encoder_opt-Adam_lr0.001_T2_seed1000_TwoChannelBaseOnlySobel/best_model.pth'  # CIFAR10预训练路径需要用户指定
elif args.data_set == 'CEP-DVS':
    if args.num_classes is None:
        args.num_classes = 20
    if not args.edge_root:
        from dataloader.cepdvs import CEPDVS_EDGE_ROOT
        args.edge_root = CEPDVS_EDGE_ROOT
    if not args.dvs_root:
        # 使用预处理的DVS数据（.pt文件）而不是原始.mat文件
        from dataloader.cepdvs import CEPDVS_DVS_PROCESSED_ROOT
        args.dvs_root = CEPDVS_DVS_PROCESSED_ROOT
    if not args.pretrained_path:
        args.pretrained_path = '/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/CEP-DVS_EdgePretrain_20_CEP-DVS_RGB2Edge_Pretrain_AP_enc-time_encoder_opt-Adam_lr0.001_T10_seed1000_RGB1.0_TWoSobelEdge_img_shape48/rgb_edge_pretrained_best.pth'  # CEP-DVS预训练路径需要用户指定
else:
    raise ValueError(f"Unsupported dataset: {args.data_set}")

device = torch.device(f"cuda:{args.GPU_id}")

log_name = (
    f"Edge2DVS_{args.data_set}_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"Edge{args.edge_sample_ratio}_"
    f"DVS{args.dvs_sample_ratio}_"
    f"img{args.img_size}"
)

log_dir = os.path.join(args.log_dir, f"Edge2DVS_{args.data_set}_{args.num_classes}", log_name)
checkpoint_dir = os.path.join(args.checkpoint, f"Edge2DVS_{args.data_set}_{args.num_classes}_{log_name}")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

model_path = os.path.join(checkpoint_dir, "best_model.pth")
writer = SummaryWriter(log_dir=log_dir)

print(f"训练配置: {log_name}")
print(f"日志目录: {writer.log_dir}")


if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_edge2dvs_result.txt", "a")

    print("\n" + "="*80)
    print(f"Edge->DVS迁移学习 ({args.data_set})")
    print("="*80)
    
    # 加载数据
    print(f"加载Edge和DVS数据集...")
    print(f"数据集: {args.data_set}")
    print(f"Edge数据路径: {args.edge_root}")
    print(f"DVS数据路径: {args.dvs_root}")
    
    if args.data_set == 'Caltech101':
        train_loader, test_loader = get_edge2dvs_caltech101(
            batch_size=args.batch_size, 
            edge_root=args.edge_root, 
            dvs_root=args.dvs_root,
            edge_ratio=args.edge_sample_ratio, 
            dvs_ratio=args.dvs_sample_ratio,
            num_workers=8,
            img_size=args.img_size
            # split_ratio不再需要，因为DVS数据已预先划分为train/test
        )
    elif args.data_set == 'CIFAR10':
        train_loader, test_loader = get_edge2dvs_cifar10(
            batch_size=args.batch_size, 
            edge_root=args.edge_root, 
            dvs_root=args.dvs_root,
            edge_ratio=args.edge_sample_ratio, 
            dvs_ratio=args.dvs_sample_ratio,
            num_workers=8,
            img_size=args.img_size
            # split_ratio不再需要，因为DVS数据已预先划分为train/test
        )
    elif args.data_set == 'CEP-DVS':
        train_loader, test_loader = get_edge2dvs_cepdvs(
            batch_size=args.batch_size, 
            edge_root=args.edge_root, 
            dvs_root=args.dvs_root,
            edge_ratio=args.edge_sample_ratio, 
            dvs_ratio=args.dvs_sample_ratio,
            num_workers=8,
            img_size=args.img_size,
            split_ratio=0.9,
            time_bins=args.T
        )
    else:
        raise ValueError(f"不支持的数据集: {args.data_set}")
    
    # 检查标签范围（调试用）
    print("\n检查数据集标签范围...")
    try:
        sample_batch = next(iter(train_loader))
        (edge_data, dvs_data), (edge_labels, dvs_labels) = sample_batch
        print(f"  Edge数据形状: {edge_data.shape}")
        print(f"  DVS数据形状: {dvs_data.shape}")
        print(f"  Edge标签范围: [{edge_labels.min().item()}, {edge_labels.max().item()}]")
        print(f"  DVS标签范围: [{dvs_labels.min().item()}, {dvs_labels.max().item()}]")
        print(f"  模型类别数: {args.num_classes}")
        
        # 检查标签是否超出范围
        if edge_labels.max().item() >= args.num_classes:
            print(f"  ⚠️  警告: Edge标签最大值 {edge_labels.max().item()} >= 类别数 {args.num_classes}")
        if dvs_labels.max().item() >= args.num_classes:
            print(f"  ⚠️  警告: DVS标签最大值 {dvs_labels.max().item()} >= 类别数 {args.num_classes}")
    except Exception as e:
        print(f"  无法检查标签范围: {e}")
    
    # 准备模型
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=args.img_size)
        print("使用VGGSNNwoAP模型")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=args.img_size, device=device)
        print("使用标准VGGSNN模型")
    
    # 加载预训练参数（参考baseline.py的方式）
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"\n加载RGB->Edge预训练参数: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # 检查checkpoint的键
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            print(f"加载epoch {checkpoint.get('epoch', 'unknown')}的预训练模型")
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型的state_dict
        model_dict = model.state_dict()
        
        # 过滤掉不匹配的键（如edge_extractor）和形状不匹配的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape
                          and 'edge_extractor' not in k}
        
        # 更新模型参数
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个预训练参数")
        skipped_params = set(model_dict.keys()) - set(pretrained_dict.keys())
        if skipped_params:
            print(f"跳过的参数数量: {len(skipped_params)}")
            edge_extractor_params = [k for k in skipped_params if 'edge_extractor' in k]
            if edge_extractor_params:
                print(f"  - edge_extractor相关参数（未加载）: {len(edge_extractor_params)} 个")
    else:
        print("警告: 未提供预训练参数，从头开始训练")
    
    if args.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # 优化器
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"使用Adam优化器，学习率: {args.lr}")
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                   weight_decay=args.weight_decay, nesterov=False)
        print(f"使用SGD优化器，学习率: {args.lr}")
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    print(f"\n迁移学习配置:")
    print(f"  编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}")
    print(f"  特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}")
    print(f"  训练轮数: {args.epoch}")
    
    criterion = TET_loss
    
    # 训练
    print("\n开始Edge->DVS迁移学习...")
    trainer = AlignmentTLTrainer_Edge2DVS(
        args, device, writer, model, optimizer, criterion, scheduler, model_path
    )
    
    best_train_acc, best_train_loss = trainer.train(train_loader)
    test_loss, test_acc1, test_acc5 = trainer.test(test_loader)
    
    print(f'\ntest_loss={test_loss:.5f} test_acc1={test_acc1:.4f} test_acc5={test_acc5:.4f}')
    
    writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
    writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)
    
    write_content = (
        f'=== Edge->DVS迁移学习结果 ({args.data_set}) ===\n'
        f'种子: {args.seed}\n'
        f'数据集: {args.data_set}\n'
        f'类别数: {args.num_classes}\n'
        f'预训练模型: {args.pretrained_path}\n'
        f'编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}\n'
        f'特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}\n'
        f'Edge样本比例: {args.edge_sample_ratio}, DVS样本比例: {args.dvs_sample_ratio}\n'
        f'best_train_acc: {best_train_acc}\n'
        f'test_acc1: {test_acc1}, test_acc5: {test_acc5}, test_loss: {test_loss}\n'
        f'==============================\n\n'
    )
    f.write(write_content)
    f.close()
    writer.close()
    
    print(f"\n训练完成！模型已保存到: {model_path}")
