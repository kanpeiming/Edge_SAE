import os
import torch
import argparse
from tqdm import tqdm
from tl_utils.common_utils import seed_all
from tl_utils.trainer import BaselineTrainer
from tl_utils.loss_function import TET_loss
from dataloader.mnist import get_n_mnist
from dataloader.cifar import get_cifar10_DVS
from dataloader.caltech101 import get_n_caltech101
from dataloader.cepdvs import get_cepdvs_dvs
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
# parser.add_argument('action', default='train_ann', type=str,
#                     choices=['train_ann', 'train_snn', 'train_snn_from_zero', 'test'],
#                     help='Action: train or test.')
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'CEP-DVS'],
                    help='the data set type.')
parser.add_argument('--batch_size', default=64, type=int, help='Batchsize')  # TODO: 观察是否可以增大
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')  # TODO: 0.001，0.0006都试一下
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=80, type=int, help='Training epochs')
# parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--id', default='test', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 16)')
parser.add_argument('--encoder_type', type=str, default='lap_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=1,
                    help='the ratio of used dvs training set. ')  # TODO: 注意观察该处数值
parser.add_argument('--dvs_encoding_type', type=str, default='TET', choices=['TET', 'spikingjelly'])
parser.add_argument('--model', type=str, default='vgg16')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--img_shape', type=int, default=None,
                    help='Image shape for Caltech101 (default: 48 for CIFAR10, 34 for MNIST, 48 for Caltech101)')
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--pretrained_path', type=str,
                    default=None,
                    help='the path of pretrained model parameters')
parser.add_argument('--data_dir', type=str, default='/data/zhan/Event_Camera_Datasets',
                    help='Root directory for all datasets')
# EventRPG数据增强参数
parser.add_argument('--use_eventrpg', action='store_true', default=False,
                    help='Whether to use EventRPG data augmentation (Geometric + RPGDrop + RPGMix)')
parser.add_argument('--eventrpg_mix_prob', type=float, default=0.5,
                    help='EventRPG RPGMix probability (default: 0.5)')
parser.add_argument('--experiment_name', type=str, default='baseline', )
args = parser.parse_args()

# 参数预设值
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

# 注意：log_name、writer和model_path在main函数中生成
# 因为img_shape需要先设置默认值
writer = None
model_path = None

if __name__ == "__main__":
    # 设置随机数种子
    seed_all(args.seed)

    # 设置图像尺寸默认值（如果未指定）
    if args.img_shape is None:
        if args.data_set == 'CIFAR10':
            args.img_shape = 48  # CIFAR10-DVS默认48x48
        elif args.data_set == 'MNIST':
            args.img_shape = 34  # N-MNIST默认34x34
        elif args.data_set == 'Caltech101':
            args.img_shape = 48  # N-Caltech101默认48x48
        elif args.data_set == 'CEP-DVS':
            args.img_shape = 48  # CEP-DVS默认48x48
        else:
            args.img_shape = 48  # 通用默认值

    print(f"\n{'=' * 60}")
    print(f"Baseline实验配置 (直接在DVS数据上训练SNN)")
    print('=' * 60)
    print(f"数据集: {args.data_set}")
    print(f"图像尺寸: {args.img_shape}×{args.img_shape}")
    print(f"时间步数: {args.T}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epoch}")
    print(f"学习率: {args.lr}")
    print(f"数据增强: {'EventRPG (mix_prob=' + str(args.eventrpg_mix_prob) + ')' if args.use_eventrpg else '传统增强'}")
    print(f"训练集使用比例: {args.dvs_sample_ratio}")
    print('=' * 60 + '\n')

    # 生成日志名称（包含img_shape信息）
    eventrpg_tag = f"_EventRPG-mix{args.eventrpg_mix_prob}" if args.use_eventrpg else ""
    log_name = (f"Baseline_{args.data_set}_"
                f"img{args.img_shape}_"
                f"T{args.T}_"
                f"seed{args.seed}_"
                f"ratio{args.dvs_sample_ratio}_"
                f"lr{args.lr}_"
                f"epoch{args.epoch}"
                f"{eventrpg_tag}_"
                f"{args.experiment_name}")

    print(f"实验日志名称: {log_name}\n")

    # 创建TensorBoard writer和模型保存路径
    log_dir = os.path.join(args.log_dir + '_' + args.data_set, log_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    checkpoint_dir = os.path.join(args.checkpoint + '_' + args.data_set)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f'{log_name}.pth')

    # ============================================================================
    # 准备数据集
    # ============================================================================
    # 注意：Baseline方法不使用验证集
    # - Baseline是直接在DVS数据上训练SNN，作为性能基准
    # - 不需要模型选择或超参数调优，因此不需要验证集
    # - 只使用训练集(train)和测试集(test)进行训练和评估
    # ============================================================================
    print("加载数据集...")
    if args.data_set == 'CIFAR10':
        train_loader, test_loader = get_cifar10_DVS(
            args.batch_size, args.T,
            train_set_ratio=args.dvs_sample_ratio,
            encode_type=args.dvs_encoding_type,
            use_eventrpg=args.use_eventrpg,
            eventrpg_mix_prob=args.eventrpg_mix_prob
        )
    elif args.data_set == 'Caltech101':
        train_loader, test_loader = get_n_caltech101(
            args.batch_size, args.T,
            train_set_ratio=args.dvs_sample_ratio,
            encode_type=args.dvs_encoding_type,
            size=args.img_shape,
            use_eventrpg=args.use_eventrpg,
            eventrpg_mix_prob=args.eventrpg_mix_prob
        )
    elif args.data_set == 'MNIST':
        train_loader, test_loader = get_n_mnist(
            args.batch_size, args.T,
            train_set_ratio=args.dvs_sample_ratio,
            encode_type=args.dvs_encoding_type,
            use_eventrpg=args.use_eventrpg,
            eventrpg_mix_prob=args.eventrpg_mix_prob
        )
    elif args.data_set == 'CEP-DVS':
        train_loader, test_loader = get_cepdvs_dvs(
            args.batch_size,
            train_set_ratio=args.dvs_sample_ratio,
            img_size=args.img_shape,
            time_bins=args.T,
            split_ratio=0.9
        )
    else:
        raise ValueError(f"不支持的数据集: {args.data_set}")

    print(f"✓ 训练集样本数: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"✓ 测试集样本数: {len(test_loader.dataset)} ({len(test_loader)} batches)")
    print()

    # ============================================================================
    # 准备模型
    # ============================================================================
    # 根据数据集选择类别数
    dataset_config = {
        'CIFAR10': 10,
        'MNIST': 10,
        'Caltech101': 101,
        'CEP-DVS': 20
    }
    
    if args.data_set not in dataset_config:
        raise ValueError(f"不支持的数据集: {args.data_set}")
    
    cls_num = dataset_config[args.data_set]
    img_shape = args.img_shape

    print(f"初始化VGGSNN模型...")
    print(f"  类别数: {cls_num}")
    print(f"  输入尺寸: {img_shape}×{img_shape}")
    
    model = VGGSNN(cls_num=cls_num, img_shape=img_shape)
    
    if args.parallel and torch.cuda.device_count() > 1:
        print(f"  使用 {torch.cuda.device_count()} 个GPU进行并行训练")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print()

    # ============================================================================
    # 加载预训练模型参数（可选）
    # ============================================================================
    if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
        print(f"加载预训练模型参数: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)

        # 检查checkpoint的键
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            epoch_info = checkpoint.get('epoch', 'unknown')
            print(f"  来自epoch: {epoch_info}")
        else:
            pretrained_dict = checkpoint

        # 获取当前模型的state_dict
        model_dict = model.state_dict()

        # 过滤掉不匹配的键（如edge_extractor等迁移学习相关模块）
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict 
                           and v.shape == model_dict[k].shape
                           and 'edge_extractor' not in k}

        # 更新模型参数
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"  成功加载: {len(pretrained_dict)}/{len(model_dict)} 个参数")
        skipped = set(model_dict.keys()) - set(pretrained_dict.keys())
        if skipped:
            print(f"  跳过参数数量: {len(skipped)}")
        print()
    else:
        print("从头开始训练（未使用预训练模型）\n")

    # ============================================================================
    # 准备训练组件
    # ============================================================================
    print("配置训练组件...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch)
    criterion = TET_loss  # Temporal Efficient Training Loss
    
    print(f"  优化器: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  学习率调度: CosineAnnealingLR (T_max={args.epoch})")
    print(f"  损失函数: TET_loss")
    print()

    # ============================================================================
    # 开始训练
    # ============================================================================
    # 使用BaselineTrainer，它不会在训练过程中使用测试集
    print("开始训练...\n")
    trainer = BaselineTrainer(args, device, writer, model, optimizer, criterion, scheduler, model_path)
    trainer.train(train_loader)

    # ============================================================================
    # 最终测试
    # ============================================================================
    print("\n" + "="*60)
    print("最终测试评估")
    print("="*60)
    test_loss, test_acc = trainer.test(test_loader)
    print(f'测试损失: {test_loss:.5f}')
    print(f'测试精度: {test_acc:.3f} ({test_acc*100:.2f}%)')
    print("="*60)
    
    # 记录最终测试结果
    writer.add_scalar(tag="final_test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="final_test/loss", scalar_value=test_loss, global_step=0)
    
    writer.close()
    print(f"\n训练完成！模型已保存至: {model_path}")
