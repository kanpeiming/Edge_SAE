# -*- coding: utf-8 -*-
"""
RGB to DVS knowledge transfer training script
RGB到DVS的迁移学习训练脚本

功能：
- 加载RGB->Edge预训练参数（可选）
- 使用RGB数据（3通道）作为源域
- 使用DVS数据（2通道）作为目标域进行迁移学习
- 支持Caltech101、CIFAR10和CEP-DVS数据集
- Caltech101: 自动移除Faces类，保持101类（包含BACKGROUND_Google）

使用方法：
1. 使用预训练参数训练:
   python train_rgb2dvs.py --data_set Caltech101 --pretrained_path /path/to/rgb_edge_pretrained_best.pth --epoch 100

2. 从头开始训练:
   python train_rgb2dvs.py --data_set Caltech101 --epoch 100

3. CIFAR10训练:
   python train_rgb2dvs.py --data_set CIFAR10 --pretrained_path /path/to/pretrained.pth --epoch 100
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

from dataloader.caltech101 import get_tl_caltech101
from dataloader.cifar import get_tl_cifar10
from pretrain.edge2dvs_trainer import AlignmentTLTrainer_Edge2DVS
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP
from tl_utils import common_utils
from tl_utils.loss_function import TET_loss

parser = argparse.ArgumentParser(description='RGB to DVS Transfer Learning')
parser.add_argument('--data_set', type=str, default='Caltech101', 
                    choices=['Caltech101', 'CIFAR10'],
                    help='Dataset name (Caltech101 or CIFAR10)')
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
                    help='Encoder type for RGB data')
parser.add_argument('--seed', type=int, default=1000, help='Random seed')
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='Transfer loss for encoder')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='Transfer loss for features')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float, help='Encoder transfer loss ratio')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float, help='Feature transfer loss ratio')
parser.add_argument('--use_woap', default=False, type=bool, help='Use VGGSNNwoAP')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/rgb2dvs/log_dir',
                    help='Tensorboard log directory')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/rgb2dvs/checkpoints',
                    help='Checkpoint directory')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_classes', type=int, default=None, help='Number of classes (default: auto-detect from dataset)')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='RGB training set ratio')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='DVS training set ratio')
parser.add_argument('--img_size', type=int, default=48, help='Image size')
parser.add_argument('--pretrained_path', type=str, default='', help='Path to RGB->Edge pretrained model (optional)')
parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
parser.add_argument('--split_ratio', type=float, default=0.9, help='Train/test split ratio for DVS data (if not pre-split)')
# 事件注意力参数
parser.add_argument('--use_event_attention', action='store_true', default=False,
                    help='Enable event mid-frame guided attention for DVS branch')
parser.add_argument('--event_attention_reduction', type=int, default=8,
                    help='Channel reduction ratio for event attention (default: 8)')

args = parser.parse_args()

# 修复：AlignmentTLTrainer_RGB2DVS使用args.epochs，但参数定义为args.epoch
# 创建别名以保持兼容性
args.epochs = args.epoch

# 根据数据集自动设置类别数
if args.data_set == 'Caltech101':
    if args.num_classes is None:
        args.num_classes = 101  # Caltech101: 100个物体类 + 1个BACKGROUND_Google (移除Faces)
elif args.data_set == 'CIFAR10':
    if args.num_classes is None:
        args.num_classes = 10
else:
    raise ValueError(f"Unsupported dataset: {args.data_set}")

device = torch.device(f"cuda:{args.GPU_id}")

log_name = (
    f"RGB2DVS_{args.data_set}_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"RGB{args.RGB_sample_ratio}_"
    f"DVS{args.dvs_sample_ratio}_"
    f"img{args.img_size}_"
    f"{'EventAttn' if args.use_event_attention else 'NoAttn'}"
)

log_dir = os.path.join(args.log_dir, f"RGB2DVS_{args.data_set}_{args.num_classes}", log_name)
checkpoint_dir = os.path.join(args.checkpoint, f"RGB2DVS_{args.data_set}_{args.num_classes}_{log_name}")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 注意：AlignmentTLTrainer_RGB2DVS期望model_path是目录，它会自动添加"best_model.pth"
model_path = checkpoint_dir
writer = SummaryWriter(log_dir=log_dir)

print(f"训练配置: {log_name}")
print(f"日志目录: {writer.log_dir}")


if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_rgb2dvs_result.txt", "a")

    print("\n" + "="*80)
    print(f"RGB->DVS迁移学习 ({args.data_set})")
    print("="*80)
    
    # 加载数据
    print(f"加载RGB和DVS数据集...")
    print(f"数据集: {args.data_set}")
    
    if args.data_set == 'Caltech101':
        print(f"注意: Caltech101 RGB数据将自动移除Faces类，保持101类（含BACKGROUND_Google）")
        train_loader, test_loader = get_tl_caltech101(
            batch_size=args.batch_size,
            train_set_ratio=args.RGB_sample_ratio,
            dvs_train_set_ratio=args.dvs_sample_ratio,
            num_workers=args.num_workers,
            img_size=args.img_size,
            split_ratio=args.split_ratio
        )
    elif args.data_set == 'CIFAR10':
        train_loader, test_loader = get_tl_cifar10(
            batch_size=args.batch_size,
            train_set_ratio=args.RGB_sample_ratio,
            dvs_train_set_ratio=args.dvs_sample_ratio,
            num_workers=args.num_workers,
            img_size=args.img_size
        )
    else:
        raise ValueError(f"不支持的数据集: {args.data_set}")
    
    # 检查数据集信息
    print("\n检查数据集信息...")
    try:
        sample_batch = next(iter(train_loader))
        (rgb_data, dvs_data), labels = sample_batch
        print(f"  RGB数据形状: {rgb_data.shape}")
        print(f"  DVS数据形状: {dvs_data.shape}")
        print(f"  标签范围: [{labels.min().item()}, {labels.max().item()}]")
        print(f"  模型类别数: {args.num_classes}")
        
        # 检查标签是否超出范围
        if labels.max().item() >= args.num_classes:
            print(f"  ⚠️  警告: 标签最大值 {labels.max().item()} >= 类别数 {args.num_classes}")
    except Exception as e:
        print(f"  无法检查数据集信息: {e}")
    
    # 准备模型
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=args.img_size,
                          use_event_attention=args.use_event_attention,
                          event_attention_reduction=args.event_attention_reduction)
        print("\n使用VGGSNNwoAP模型 (without Average Pooling)")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=args.img_size, device=device,
                      use_event_attention=args.use_event_attention,
                      event_attention_reduction=args.event_attention_reduction)
        print("\n使用标准VGGSNN模型 (with Average Pooling)")
    
    # 打印事件注意力配置
    if args.use_event_attention:
        print(f"✓ 事件注意力已启用:")
        print(f"  - 中间稳定帧引导的时序注意力")
        print(f"  - 插入位置: dvs_input后 + features[0]后")
        print(f"  - 通道压缩比: {args.event_attention_reduction}")
    else:
        print("✗ 事件注意力未启用（使用标准训练）")
    
    # 加载预训练参数（如果提供）
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"\n加载预训练参数: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # 检查checkpoint的键
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            print(f"  加载epoch {checkpoint.get('epoch', 'unknown')}的预训练模型")
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型的state_dict
        model_dict = model.state_dict()
        
        # 构建需要排除的模块列表
        exclude_modules = ['edge_extractor', 'rgb_to_gray']  # 排除边缘提取器和灰度转换模块
        
        # 过滤掉不匹配的键和需要排除的模块
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape
                          and not any(exclude_module in k for exclude_module in exclude_modules)}
        
        # 更新模型参数
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"  成功加载 {len(pretrained_dict)}/{len(model_dict)} 个预训练参数")
        skipped_params = set(model_dict.keys()) - set(pretrained_dict.keys())
        if skipped_params:
            print(f"  跳过的参数数量: {len(skipped_params)}")
            edge_extractor_params = [k for k in skipped_params if 'edge_extractor' in k or 'rgb_to_gray' in k]
            if edge_extractor_params:
                print(f"    - edge_extractor/rgb_to_gray相关参数（未加载）: {len(edge_extractor_params)} 个")
    else:
        if args.pretrained_path:
            print(f"\n警告: 预训练模型路径不存在: {args.pretrained_path}")
        print("  从头开始训练（未使用预训练参数）")
    
    if args.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # 优化器
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"\n使用Adam优化器，学习率: {args.lr}")
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                   weight_decay=args.weight_decay, nesterov=False)
        print(f"\n使用SGD优化器，学习率: {args.lr}")
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")
    
    # 学习率调度器
    if args.data_set == 'Caltech101':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'CIFAR10':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n迁移学习配置:")
    print(f"  编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}")
    print(f"  特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}")
    print(f"  训练轮数: {args.epoch}")
    print(f"  编码器类型: {args.encoder_type}")
    
    criterion = TET_loss
    print(f"\n使用TET (Temporal Efficient Training) Loss")
    
    # 训练
    print("\n开始RGB->DVS迁移学习...")
    print("使用Edge2DVS训练器（RGB作为3通道输入）")
    trainer = AlignmentTLTrainer_Edge2DVS(
        args, device, writer, model, optimizer, criterion, scheduler, model_path
    )
    
    best_train_acc, best_train_loss = trainer.train(train_loader)
    test_loss, test_acc1, test_acc5 = trainer.test(test_loader)
    
    print(f'\n最终测试结果:')
    print(f'  test_loss={test_loss:.5f}')
    print(f'  test_acc1={test_acc1:.4f} ({test_acc1*100:.2f}%)')
    print(f'  test_acc5={test_acc5:.4f} ({test_acc5*100:.2f}%)')
    
    writer.add_scalar(tag="final_test/accuracy1", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="final_test/accuracy5", scalar_value=test_acc5, global_step=0)
    writer.add_scalar(tag="final_test/loss", scalar_value=test_loss, global_step=0)
    
    write_content = (
        f'=== RGB->DVS迁移学习结果 ({args.data_set}) ===\n'
        f'种子: {args.seed}\n'
        f'数据集: {args.data_set}\n'
        f'类别数: {args.num_classes}\n'
    )
    
    if args.data_set == 'Caltech101':
        write_content += f'注意: RGB数据已移除Faces类，保持101类（含BACKGROUND_Google）\n'
    
    write_content += (
        f'预训练模型: {args.pretrained_path if args.pretrained_path else "无（从头训练）"}\n'
        f'编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}\n'
        f'特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}\n'
        f'RGB样本比例: {args.RGB_sample_ratio}, DVS样本比例: {args.dvs_sample_ratio}\n'
        f'best_train_acc: {best_train_acc:.4f}\n'
        f'test_acc1: {test_acc1:.4f}, test_acc5: {test_acc5:.4f}, test_loss: {test_loss:.5f}\n'
        f'==============================\n\n'
    )
    f.write(write_content)
    f.close()
    writer.close()
    
    print(f"\n训练完成！模型已保存到: {os.path.join(model_path, 'best_model.pth')}")
    print(f"结果已记录到: {args.data_set}_{args.seed}_rgb2dvs_result.txt")
