"""
N-Caltech101 数据集基线训练脚本
Training script for N-Caltech101 dataset baseline model

数据集要求 (Dataset Requirements):
数据集路径应包含以下结构:
dataset_path/
├── train/
│   ├── 0_np.pt
│   ├── 1_np.pt
│   └── ...
└── test/
    ├── 0_np.pt
    ├── 1_np.pt
    └── ...

使用示例 (Usage Examples):
1. 使用自定义数据集路径训练:
   python train_caltech101_baseline.py --caltech101_dvs_path /home/user/kpm/kpm/Dataset/Caltech101/n-caltech101 --batch_size 32

2. 使用默认数据路径训练:
   python train_caltech101_baseline.py --batch_size 32 --lr 0.001 --epoch 100

3. 使用NDA数据增强训练:
   python train_caltech101_baseline.py --use_nda --batch_size 32 --lr 0.001 --epoch 100

4. 使用EventRPG数据增强训练:
   python train_caltech101_baseline.py --use_eventrpg --eventrpg_mix_prob 0.5 --batch_size 32

5. 使用预训练模型:
   python train_caltech101_baseline.py --pretrained_path /path/to/pretrained.pth --lr 0.0001

参数说明:
- caltech101_dvs_path: N-Caltech101数据集路径 (包含train和test文件夹)
- batch_size: 批次大小 (默认32)
- lr: 学习率 (默认0.001)
- T: SNN时间步数 (默认10)
- size: 输入图像尺寸 (默认48)
- dvs_sample_ratio: 训练集使用比例 (默认1.0)
- use_nda: 是否使用NDA数据增强 (默认False，包含水平翻转0.5+roll/rotate/shear随机选择)
- use_eventrpg: 是否使用EventRPG数据增强 (默认False，包含几何增强+RPGMix)
- eventrpg_mix_prob: EventRPG的RPGMix概率 (默认0.5)

特性:
- 模块化数据加载器（位于dataloader.caltech101模块）
- 自动识别.pt文件格式（支持xxx.pt和xxx_np.pt）
- 智能数据增强（随机翻转、平移）
- 101类别分类头
- 简洁的训练脚本结构
"""

import os
import torch
import argparse
from tqdm import tqdm
from tl_utils.common_utils import seed_all
from tl_utils.trainer import Trainer
from tl_utils.loss_function import TET_loss, TRT_loss
from dataloader.caltech101 import create_caltech101_dataloaders
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training for N-Caltech101')
parser.add_argument('--batch_size', default=64, type=int, help='Batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=60, type=int, help='Training epochs')
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
parser.add_argument('--caltech101_dvs_path', type=str, default='/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101',
                    help='Path to N-Caltech101 DVS dataset (if not provided, will use default path in dataloader)')
# NDA (Neuromorphic Data Augmentation) 参数
parser.add_argument('--use_nda', action='store_true', default=False,
                    help='Whether to use NDA (Neuromorphic Data Augmentation) including flip(0.5) + roll/rotate/shear')
# EventRPG 数据增强参数
parser.add_argument('--use_eventrpg', action='store_true', default=False,
                    help='Whether to use EventRPG data augmentation (Geometric + RPGDrop + RPGMix)')
parser.add_argument('--eventrpg_mix_prob', type=float, default=0.5,
                    help='EventRPG RPGMix probability (default: 0.5)')
# TRT (Temporal Regularization Training) 参数
parser.add_argument('--use_trt', action='store_true', default=False,
                    help='Whether to use TRT (Temporal Regularization Training) loss')
parser.add_argument('--trt_decay', type=float, default=0.5,
                    help='TRT decay factor δ (controls regularization decay over time, default: 0.5)')
parser.add_argument('--trt_lambda', type=float, default=1e-5,
                    help='TRT regularization coefficient λ (default: 1e-5)')
parser.add_argument('--trt_epsilon', type=float, default=1e-5,
                    help='TRT epsilon ε (safeguard value to prevent division by zero, default: 1e-5)')
parser.add_argument('--trt_eta', type=float, default=0.05,
                    help='TRT eta η (MSE loss weight, default: 0.05)')
parser.add_argument('--fine_tuning', default= 'no', type = str, help='Fine-tuning or no')
args = parser.parse_args()

# 添加缺失的data_set属性（trainer需要用到）
args.data_set = 'Caltech101'

# 参数预设值
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 日志名称 加载预训练参数，改变一下命名为微调
eventrpg_tag = f"_EventRPG-mix{args.eventrpg_mix_prob}" if args.use_eventrpg else ""
log_name = f"FT_{args.fine_tuning}_NCaltech101_baseline_lr{args.lr}_T{args.T}_bs{args.batch_size}_seed{args.seed}_nda{args.use_nda}_trt{args.use_trt}_{eventrpg_tag}_imageSize{args.size}"

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
    if args.use_eventrpg:
        print(f"Using EventRPG augmentation:")
        print(f"  - Geometric augmentation: Identity/Flip/Rotate/Scale/Translation/Shear (randomly selected)")
        print(f"  - RPGMix probability: {args.eventrpg_mix_prob}")
    elif args.use_nda:
        print(f"Using NDA augmentation:")
        print("  - Horizontal flip: 50% probability")
        print("  - Random augmentation: Roll/Rotate/Shear (one selected randomly)")
    else:
        print("Using traditional augmentation:")
        print("  - Horizontal flip: 50% probability")
        print("  - Random translation")
    
    train_loader, test_loader = create_caltech101_dataloaders(
        data_path=args.caltech101_dvs_path,
        batch_size=args.batch_size,
        train_ratio=args.dvs_sample_ratio,
        num_workers=8,
        img_size=args.size,
        use_nda=args.use_nda,  # 传递NDA参数
        use_eventrpg=args.use_eventrpg,  # 传递EventRPG参数
        eventrpg_mix_prob=args.eventrpg_mix_prob
    )

    # 准备模型 - N-Caltech101有101个类别
    print("Initializing VGGSNN model for N-Caltech101...")
    model = VGGSNN(2, 101, args.size)  # N-Caltech101: (2, 101, size)
    
    if args.parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 加载预训练模型参数（如果提供）
    if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained model from {args.pretrained_path}")
        try:
            checkpoint = torch.load(args.pretrained_path, map_location=device)
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()

            # 筛选出可以迁移的参数（从RGB->Edge预训练模型迁移到DVS模型）
            pretrained_dict_filtered = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    # 从RGB分支迁移到DVS分支的参数
                    if 'rgb_input' in k:
                        # 将rgb_input的参数迁移到dvs_input
                        dvs_key = k.replace('rgb_input', 'dvs_input')
                        if dvs_key in model_dict:
                            if 'weight' in k and v.shape[1] == 3:  # RGB输入层权重
                                # RGB预训练时学到了从3通道RGB到2通道边缘的映射
                                # 现在DVS也是2通道，取RGB权重的前2个通道作为初始化
                                pretrained_dict_filtered[dvs_key] = v[:, :2, :, :].clone()
                                # print(f"  迁移 {k} -> {dvs_key} (RGB 3通道 -> DVS 2通道)")
                            else:
                                # 偏置等参数直接迁移
                                pretrained_dict_filtered[dvs_key] = v.clone()
                                print(f"  迁移 {k} -> {dvs_key}")
                    elif 'features' in k or 'classifier' in k or 'bottleneck' in k:
                        # 特征提取层和分类器直接迁移（这些层学到了边缘特征的高层表示）
                        pretrained_dict_filtered[k] = v.clone()
                        # print(f"  迁移 {k}")
                    elif 'edge_extractor' in k:
                        # 跳过边缘提取器参数，DVS不需要
                        print(f"  跳过边缘提取器参数: {k}")

            if pretrained_dict_filtered:
                model_dict.update(pretrained_dict_filtered)
                model.load_state_dict(model_dict)
                print(f"Successfully loaded {len(pretrained_dict_filtered)} pretrained parameters")
            else:
                print("Warning: No matching parameters found in pretrained model")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Training from scratch...")

    # 准备训练组件
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch)

    print("Starting training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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

    # 训练
    trainer = Trainer(args, device, writer, model, optimizer, criterion, scheduler, model_path)
    trainer.train(train_loader, test_loader)

    # 测试
    print("Final testing...")
    test_loss, test_acc = trainer.test(test_loader)
    print(f'Final test results - Loss: {test_loss:.5f}, Accuracy: {test_acc:.5f} ({test_acc*100:.1f}%)')
    
    # 记录最终测试结果
    writer.add_scalar(tag="final_test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="final_test/loss", scalar_value=test_loss, global_step=0)
    
    writer.close()
    print(f"Training completed. Model saved to: {model_path}")
