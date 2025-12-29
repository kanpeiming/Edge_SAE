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

3. 使用预训练模型:
   python train_caltech101_baseline.py --pretrained_path /path/to/pretrained.pth --lr 0.0001 --load_features --load_bottleneck --load_classifier

参数说明:
- caltech101_dvs_path: N-Caltech101数据集路径 (包含train和test文件夹)
- batch_size: 批次大小 (默认32)
- lr: 学习率 (默认0.001)
- T: SNN时间步数 (默认10)
- size: 输入图像尺寸 (默认48)
- dvs_sample_ratio: 训练集使用比例 (默认1.0)
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
- 简洁的训练脚本结构
"""

import os
import torch
import argparse
from tqdm import tqdm
from tl_utils.common_utils import seed_all
from tl_utils.trainer import Trainer
from tl_utils.loss_function import TET_loss
from dataloader.caltech101 import create_caltech101_dataloaders
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training for N-Caltech101')
parser.add_argument('--batch_size', default=64, type=int, help='Batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=100, type=int, help='Training epochs')
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
args = parser.parse_args()

# 添加缺失的data_set属性（trainer需要用到）
args.data_set = 'Caltech101'

# 参数预设值
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 日志名称
log_name = f"FT_{args.fine_tuning}_NCaltech101_baseline_lr{args.lr}_T{args.T}_bs{args.batch_size}_seed{args.seed}_imageSize{args.size}"

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
    
    train_loader, test_loader = create_caltech101_dataloaders(
        data_path=args.caltech101_dvs_path,
        batch_size=args.batch_size,
        train_ratio=args.dvs_sample_ratio,
        num_workers=8,
        img_size=args.size,
        use_nda=False,
        use_eventrpg=False,
        eventrpg_mix_prob=0.5
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
