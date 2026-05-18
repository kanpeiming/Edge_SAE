"""
N-Caltech101 数据集基线训练脚本
Training script for N-Caltech101 dataset baseline model

数据集要求 (Dataset Requirements):
支持两种数据格式：

1. .pt格式（推荐，加载速度快）:
dataset_path/
├── train/
│   ├── 0_np.pt
│   ├── 1_np.pt
│   └── ...
└── test/
    ├── 0_np.pt
    ├── 1_np.pt
    └── ...

2. .bin格式（原始事件数据）:
dataset_path/
├── accordion/
│   ├── image_0001.bin
│   └── ...
├── wild_cat/
│   ├── image_0016.bin
│   └── ...
└── ... (共101个类别目录)

注意：.bin格式需要先使用预处理脚本转换为.pt格式以提高训练速度：
python preprocess_bin_to_pt.py --bin_root /path/to/bin/data --pt_output /path/to/pt/data

使用示例 (Usage Examples):
1. 使用自定义数据集路径训练:
   python train_caltech101_baseline.py --caltech101_dvs_path /home/user/kpm/kpm/Dataset/Caltech101/n-caltech101 --batch_size 32

2. 使用默认数据路径训练:
   python train_caltech101_baseline.py --batch_size 32 --lr 0.001 --epoch 100

3. 使用预训练模型:
   python train_caltech101_baseline.py --pretrained_path /path/to/pretrained.pth --lr 0.0001 --load_features --load_bottleneck --load_classifier

参数说明:
- caltech101_dvs_path: N-Caltech101数据集路径 (包含train和test文件夹)
- batch_size: 批次大小 (默认64)
- lr: 学习率 (默认0.001)
- T: SNN时间步数 (默认10)
- size: 输入图像尺寸 (默认48)
- dvs_sample_ratio: 训练集使用比例 (默认1.0)
- val_split: 验证集划分比例 (默认0.0，即不从训练集中拆分，直接使用测试集作为验证集)
- pretrained_path: 预训练模型路径
- load_features: 是否加载features模块 (Conv层)
- load_bottleneck: 是否加载bottleneck层
- load_classifier: 是否加载classifier层

特性:
- 模块化数据加载器（位于dataloader.caltech101模块）
- 自动识别.pt文件格式（支持xxx.pt和xxx_np.pt）
- 传统数据增强（随机翻转、平移）
- 101类别分类头
- 支持预训练模型加载
- 使用测试集作为每轮验证集选择最佳模型，训练集使用全量数据
- 简洁的训练脚本结构
"""

import os
import torch
import argparse
from tqdm import tqdm
from tl_utils.common_utils import seed_all
from tl_utils.trainer import BaselineTrainer
from tl_utils.loss_function import TET_loss
from dataloader.caltech101 import create_caltech101_dataloaders
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training for N-Caltech101')
parser.add_argument('--batch_size', default=16, type=int, help='Batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=50, type=int, help='Training epochs')
parser.add_argument('--id', default='caltech101_baseline', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 10)')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training.')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0,
                    help='the ratio of used dvs training set.')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--pretrained_path', type=str, default=None,
                    help='the path of pretrained model parameters')
parser.add_argument('--size', type=int, default=48,
                    help='Input image size for N-Caltech101')
parser.add_argument('--caltech101_dvs_path', type=str, default='/home/user/kpm/kpm/Dataset/Caltech101/NCALTECH101/NCALTECH101/Caltech101',
                    help='Path to N-Caltech101 DVS dataset (if not provided, will use default path in dataloader)')
# Fine-tuning and pretrained model loading parameters
parser.add_argument('--fine_tuning', default='no', type=str, help='Fine-tuning mode identifier')
parser.add_argument('--load_dvs_input', action='store_true', default=False,
                    help='Whether to load dvs_input related parameters from pretrained model (default: False)')
parser.add_argument('--load_features', action='store_true', default=False,
                    help='Whether to load features related parameters from pretrained model (default: False)')
parser.add_argument('--load_bottleneck', action='store_true', default=False,
                    help='Whether to load bottleneck related parameters from pretrained model (default: False)')
parser.add_argument('--load_classifier', action='store_true', default=False,
                    help='Whether to load classifier related parameters from pretrained model (default: False)')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of data loading workers (default: 4, use 0 for Windows if encountering issues)')
# 事件注意力参数
parser.add_argument('--use_event_attention', action='store_true', default=False,
                    help='Enable event mid-frame guided attention for DVS data')
parser.add_argument('--event_attention_reduction', type=int, default=8,
                    help='Channel reduction ratio for event attention (default: 8)')
# 验证集划分参数
parser.add_argument('--val_split', type=float, default=0.0,
                    help='Validation set split ratio from training set (default: 0.0, i.e., use test set as validation)')
args = parser.parse_args()

# 添加缺失的data_set属性（trainer需要用到）
args.data_set = 'Caltech101'

# 参数预设值
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 日志名称
attn_tag = "_EventAttn" if args.use_event_attention else ""
log_name = f"FT_{args.fine_tuning}_NCaltech101_baseline_lr{args.lr}_T{args.T}_bs{args.batch_size}_seed{args.seed}_imageSize{args.size}{attn_tag}"

# 创建日志和检查点目录
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.checkpoint, exist_ok=True)

writer = SummaryWriter(log_dir=os.path.join(args.log_dir, log_name))
print(f"Training log: {log_name}")

model_path = os.path.join(args.checkpoint, f'{log_name}.pth')


if __name__ == "__main__":
    # 设置随机数种子
    seed_all(args.seed)

    # 准备数据
    print("Loading N-Caltech101 dataset...")
    print(f"Dataset path: {args.caltech101_dvs_path}")
    
    # 打印数据增强信息
    print("Using traditional augmentation:")
    print("  - Horizontal flip: 50% probability")
    print("  - Random translation")
    print("Validation strategy: using test set as validation (no split from training set)")
    
    train_loader, _, test_loader = create_caltech101_dataloaders(
        data_path=args.caltech101_dvs_path,
        batch_size=args.batch_size,
        train_ratio=args.dvs_sample_ratio,
        num_workers=args.num_workers,
        img_size=args.size,
        use_nda=False,
        use_eventrpg=False,
        eventrpg_mix_prob=0.5,
        val_split=0.0
    )
    # 用测试集充当每轮 epoch 的验证集，训练集使用全量数据
    val_loader = test_loader

    # 准备模型 - N-Caltech101有101个类别
    print("Initializing VGGSNN model for N-Caltech101...")
    model = VGGSNN(2, 101, args.size, 
                   use_event_attention=args.use_event_attention,
                   event_attention_reduction=args.event_attention_reduction)
    
    # 打印事件注意力配置
    if args.use_event_attention:
        print(f"✓ 事件注意力已启用:")
        print(f"  - 中间稳定帧引导的时序注意力")
        print(f"  - 插入位置: dvs_input后 + features[0]后")
        print(f"  - 通道压缩比: {args.event_attention_reduction}")
        print(f"  - 仅作用于DVS数据（Baseline训练）")
    else:
        print("✗ 事件注意力未启用（使用标准Baseline训练）")
    
    if args.parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 加载预训练模型参数（如果提供）
    if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
        print(f"正在加载预训练模型参数: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)

        # 检查checkpoint的键
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            print(f"加载epoch {checkpoint.get('epoch', 'unknown')}的预训练模型")
        else:
            pretrained_dict = checkpoint

        # 获取当前模型的state_dict
        model_dict = model.state_dict()

        # 构建需要排除的模块列表
        exclude_modules = ['edge_extractor']  # 始终排除edge_extractor
        if not args.load_dvs_input:
            exclude_modules.append('dvs_input')
        if not args.load_features:
            exclude_modules.append('features')
        if not args.load_bottleneck:
            exclude_modules.append('bottleneck')
        if not args.load_classifier:
            exclude_modules.append('classifier')

        # 过滤掉不匹配的键和需要排除的模块
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape
                           and not any(exclude_module in k for exclude_module in exclude_modules)}

        # 更新模型参数
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个预训练参数")
        skipped_params = set(model_dict.keys()) - set(pretrained_dict.keys())
        print(f"跳过的参数数量: {len(skipped_params)}")
        
        # 详细显示各个模块的跳过情况
        if not args.load_dvs_input:
            dvs_input_params = [k for k in skipped_params if 'dvs_input' in k]
            if dvs_input_params:
                print(f"  - dvs_input相关参数（未加载）: {len(dvs_input_params)} 个")
        
        if not args.load_features:
            features_params = [k for k in skipped_params if 'features' in k]
            if features_params:
                print(f"  - features相关参数（未加载）: {len(features_params)} 个")
        
        if not args.load_bottleneck:
            bottleneck_params = [k for k in skipped_params if 'bottleneck' in k]
            if bottleneck_params:
                print(f"  - bottleneck相关参数（未加载）: {len(bottleneck_params)} 个")
        
        if not args.load_classifier:
            classifier_params = [k for k in skipped_params if 'classifier' in k]
            if classifier_params:
                print(f"  - classifier相关参数（未加载）: {len(classifier_params)} 个")
        
        # 显示其他跳过的参数
        other_params = [k for k in skipped_params 
                        if not any(module in k for module in ['dvs_input', 'features', 'bottleneck', 'classifier', 'edge_extractor'])]
        if other_params:
            print(f"  - 其他跳过的参数: {other_params}")
    else:
        print("训练从头开始（未提供预训练模型）")

    # 准备训练组件
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch)

    print("Starting training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 使用TET损失函数
    print(f"\n使用TET (Temporal Efficient Training) Loss")
    criterion = TET_loss

    # 使用BaselineTrainer（使用验证集选择最佳模型）
    trainer = BaselineTrainer(args, device, writer, model, optimizer, criterion, scheduler, model_path)
    
    print("\n注意：使用BaselineTrainer - 训练过程中使用【测试集】作为验证集选择最佳模型")
    print("=" * 80)
    
    best_val_acc = trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("训练完成！开始在测试集上进行最终评估...")
    print("=" * 80)
    
    # 加载最佳模型
    if os.path.exists(model_path):
        print(f"加载最佳训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)
    
    # 在测试集上进行最终评估
    test_loss, test_acc = trainer.test(test_loader)
    print(f'\n最终测试结果:')
    print(f'  Loss: {test_loss:.5f}')
    print(f'  Accuracy: {test_acc:.5f} ({test_acc*100:.2f}%)')
    print(f'  最佳训练准确率: {trainer.best_train_acc:.5f} ({trainer.best_train_acc*100:.2f}%)')
    print(f'  最佳验证准确率: {best_val_acc:.5f} ({best_val_acc*100:.2f}%)')
    
    # 记录最终测试结果
    writer.add_scalar(tag="final_test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="final_test/loss", scalar_value=test_loss, global_step=0)
    writer.add_scalar(tag="final_test/train_accuracy", scalar_value=trainer.best_train_acc, global_step=0)
    writer.add_scalar(tag="final_test/val_accuracy", scalar_value=best_val_acc, global_step=0)
    
    writer.close()
    print(f"Training completed. Model saved to: {model_path}")
