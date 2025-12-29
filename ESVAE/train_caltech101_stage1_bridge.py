# -*- coding: utf-8 -*-
"""
Stage 1: Event-aware Bridge Pretraining Script for Caltech101

功能：
- 加载Stage 0预训练的模型
- 进行Stage 1 Bridge训练：RGB-edge ↔ DVS事件感知对齐
- 使用category-level匹配的数据
- 主要目标：引入真实DVS数据，使backbone具备事件感知能力

使用方法：
1. 从Stage 0预训练模型开始：
   python train_caltech101_stage1_bridge.py --stage0_path /path/to/rgb_edge_pretrained_best.pth --epochs 30 --lr 0.0001

2. 自定义Bridge和分类损失权重：
   python train_caltech101_stage1_bridge.py --bridge_loss_weight 1.0 --cls_loss_weight 0.1

3. 使用部分数据训练：
   python train_caltech101_stage1_bridge.py --train_set_ratio 0.8
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

from dataloader.caltech101 import get_stage1_bridge_caltech101, get_n_caltech101
from pretrain.stage1_trainer import Stage1BridgeTrainer
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP
from pretrain.Edge import EventBridgeHead
from tl_utils.loss_function import TET_loss
from tl_utils import common_utils

parser = argparse.ArgumentParser(description='Caltech101 Stage 1: Event-aware Bridge Pretraining')
parser.add_argument('--batch_size', default=8, type=int, help='Batchsize')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for Stage 1 (小学习率)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=30, type=int, help='Stage 1 training epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 10)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training.')

# Stage 1 特定参数
parser.add_argument('--stage0_path', type=str, required=True,
                    help='Path to Stage 0 pretrained model')
parser.add_argument('--bridge_loss_weight', default=1.0, type=float,
                    help='Weight for bridge loss (主导损失)')
parser.add_argument('--cls_loss_weight', default=0.1, type=float,
                    help='Weight for classification loss (稳定正则项，小权重)')
parser.add_argument('--edge_data_path', type=str, default=None,
                    help='Path to preprocessed edge data (if None, use default path)')

# 模型参数
parser.add_argument('--use_woap', default=False, type=bool,
                    help='Whether to use without Average Pooling version')
parser.add_argument('--img_shape', type=int, default=48, help='Image shape for Caltech101')

# 数据参数
parser.add_argument('--train_set_ratio', type=float, default=1.0, 
                    help='Ratio of training set to use')
parser.add_argument('--dvs_train_set_ratio', type=float, default=1.0,
                    help='Ratio of DVS training set to use')

# 路径参数
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/stage1/log_dir',
                    help='Path to tensorboard log directory')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/stage1/checkpoints',
                    help='Path to checkpoint directory')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID to use')

args = parser.parse_args()

# 固定Caltech101参数
args.data_set = 'Caltech101'
args.num_classes = 101

device = torch.device(f"cuda:{args.GPU_id}")

# 生成日志名称
log_name = (
    f"Caltech101_Stage1Bridge_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"bridge{args.bridge_loss_weight}_"
    f"cls{args.cls_loss_weight}_"
    f"img{args.img_shape}"
)

# 日志目录设置
log_dir = os.path.join(
    args.log_dir,
    f"Caltech101_Stage1_{args.num_classes}",
    log_name
)

# 模型保存路径
checkpoint_dir = os.path.join(
    args.checkpoint,
    f"Caltech101_Stage1_{args.num_classes}_{log_name}"
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
    f = open(f"Caltech101_{args.seed}_stage1_bridge_result.txt", "a")

    print("\n" + "="*80)
    print("Stage 1: Event-aware Bridge Pretraining (RGB-edge ↔ DVS)")
    print("="*80)
    
    # 准备数据
    print("Loading Caltech101 Stage 1 Bridge dataset...")
    train_loader, test_loader = get_stage1_bridge_caltech101(
        args.batch_size, 
        args.train_set_ratio,
        args.dvs_train_set_ratio,
        edge_data_path=args.edge_data_path
    )
    
    print(f"\n=== Stage 1 Bridge训练数据集信息 ===")
    print(f"训练集数量: {len(train_loader.dataset)}")
    print(f"测试集数量: {len(test_loader.dataset)}")
    print(f"类别数量: {args.num_classes}")
    print(f"训练模式: RGB-edge和DVS category-level匹配")
    print(f"目标: 引入真实DVS数据，使backbone具备事件感知能力")
    print("===========================\n")

    # 准备模型
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=args.img_shape)
        print("使用VGGSNNwoAP模型 (without Average Pooling)")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=args.img_shape, device=device)
        print("使用标准VGGSNN模型 (with Average Pooling)")
    
    print(f"  架构: VGGSNN")
    print(f"  图像尺寸: {args.img_shape}×{args.img_shape}")
    print(f"  输入通道: 统一使用dvs_input (2通道)")
    print(f"  Edge数据: 使用预处理数据（Sobel 2通道）")

    # 添加EventBridgeHead
    model.event_bridge_head = EventBridgeHead(
        input_dim=256,  # bottleneck维度
        output_size=args.img_shape,
        prediction_type='density'
    )
    
    print("✓ 已添加EventBridgeHead:")
    print("  - 输入: bottleneck特征 (256维)")
    print("  - 输出: 事件密度图 (H×W)")
    print("  - 功能: 从Edge预测DVS事件统计")
    print("  - 优势: Edge数据已预处理，节省显存和计算时间")

    # 加载Stage 0预训练模型
    if os.path.exists(args.stage0_path):
        print(f"\n正在加载Stage 0预训练模型: {args.stage0_path}")
        checkpoint = torch.load(args.stage0_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            print(f"✓ 加载Stage 0模型参数")
        else:
            pretrained_dict = checkpoint
        
        # 获取当前模型的state_dict
        model_dict = model.state_dict()
        
        # 过滤：加载所有兼容的参数，跳过event_bridge_head
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items()
                                    if k in model_dict and v.shape == model_dict[k].shape
                                    and 'event_bridge_head' not in k}
        
        # 更新模型参数
        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)
        
        print(f"✓ 成功加载 {len(pretrained_dict_filtered)}/{len(model_dict)} 个预训练参数")
        print(f"  跳过的参数: {set(model_dict.keys()) - set(pretrained_dict_filtered.keys())}")
    else:
        raise FileNotFoundError(f"Stage 0预训练模型不存在: {args.stage0_path}")

    if args.parallel and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 准备优化器：分层学习率
    # - dvs_input: 中等学习率（继续适应）
    # - features/bottleneck: 小学习率（保持Stage 0结构）
    # - classifier: 小学习率（稳定决策边界）
    # - event_bridge_head: 大学习率（从头训练）
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'event_bridge_head' in n], 
             'lr': args.lr * 10},  # Bridge head: 高学习率
            {'params': [p for n, p in model.named_parameters() if 'dvs_input' in n], 
             'lr': args.lr * 2},   # DVS input: 中等学习率
            {'params': [p for n, p in model.named_parameters() 
                       if 'event_bridge_head' not in n and 'dvs_input' not in n and 'edge_extractor' not in n], 
             'lr': args.lr}        # Backbone: 小学习率
        ])
        print(f"\nStage 1优化器: Adam")
        print(f"  EventBridgeHead学习率: {args.lr * 10}")
        print(f"  DVS Input学习率: {args.lr * 2}")
        print(f"  Backbone学习率: {args.lr}")
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': [p for n, p in model.named_parameters() if 'event_bridge_head' in n], 
             'lr': args.lr * 10, 'momentum': 0.9, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if 'dvs_input' in n], 
             'lr': args.lr * 2, 'momentum': 0.9, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() 
                       if 'event_bridge_head' not in n and 'dvs_input' not in n and 'edge_extractor' not in n], 
             'lr': args.lr, 'momentum': 0.9, 'weight_decay': args.weight_decay}
        ])
        print(f"\nStage 1优化器: SGD")
        print(f"  EventBridgeHead学习率: {args.lr * 10}")
        print(f"  DVS Input学习率: {args.lr * 2}")
        print(f"  Backbone学习率: {args.lr}")
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\nStage 1配置:")
    print(f"  Bridge损失权重: {args.bridge_loss_weight} (主导)")
    print(f"  分类损失权重: {args.cls_loss_weight} (稳定正则)")
    print(f"  训练epochs: {args.epochs}")
    print(f"  编码器类型: {args.encoder_type}")
    print(f"  目标: RGB-edge ↔ DVS事件感知桥接")
    
    # 使用TET损失函数
    print(f"\n使用TET (Temporal Efficient Training) Loss")
    criterion = TET_loss
    
    # Stage 1训练
    print("\n开始Stage 1 Bridge训练...")
    trainer = Stage1BridgeTrainer(
        args, device, writer, model, optimizer, criterion, scheduler, 
        os.path.join(model_path, "stage1_bridge.pth")
    )
    
    best_edge_acc, best_dvs_acc = trainer.train(train_loader)
    
    # Stage 1测试
    test_loss, test_acc1, test_acc5 = trainer.test(test_loader)
    print(f'\nStage 1结果: test_loss={test_loss:.5f} test_acc1={test_acc1:.4f} test_acc5={test_acc5:.4f}')
    
    # 保存Stage 1模型
    stage1_path = os.path.join(model_path, "stage1_bridge_best.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc1': test_acc1,
        'test_acc5': test_acc5,
        'test_loss': test_loss,
        'args': args
    }, stage1_path)
    print(f"Stage 1模型已保存到: {stage1_path}")

    # 记录测试结果到TensorBoard
    writer.add_scalar(tag="final/stage1_accuracy", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="final/stage1_loss", scalar_value=test_loss, global_step=0)

    # 保存结果到文件
    write_content = (
        f'=== Caltech101 Stage 1: Event-aware Bridge Pretraining 结果 ===\n'
        f'种子: {args.seed}\n'
        f'Stage 0模型: {args.stage0_path}\n'
        f'模型: {"VGGSNNwoAP" if args.use_woap else "VGGSNN"}\n'
        f'训练epochs: {args.epochs}, 学习率: {args.lr}\n'
        f'Bridge损失权重: {args.bridge_loss_weight}\n'
        f'分类损失权重: {args.cls_loss_weight}\n'
        f'训练集比例: {args.train_set_ratio}\n'
        f'Stage 1测试准确率: {test_acc1:.4f}%\n'
        f'模型保存路径: {stage1_path}\n'
        f'目标: RGB-edge ↔ DVS事件感知桥接\n'
        f'下一步: Stage 2 DVS-only微调\n'
        f'=====================================\n\n'
    )
    f.write(write_content)
    f.close()
    
    writer.close()
    print(f"\nStage 1训练完成！模型已保存到: {stage1_path}")
    print(f"结果已记录到: Caltech101_{args.seed}_stage1_bridge_result.txt")
    print(f"\n使用Stage 1参数进行Stage 2 DVS微调:")
    print(f"python train_caltech101_baseline.py --pretrained_path {stage1_path}")

