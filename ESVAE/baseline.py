import os
import torch
import argparse
from tqdm import tqdm
from tl_utils.common_utils import seed_all
from tl_utils.trainer import Trainer
from tl_utils.loss_function import TET_loss
# from dataloader import get_cifar10_DVS, get_n_mnist, get_n_caltech101  暂时不使用n_caltech101数据
from dataloader.mnist import get_n_mnist
from dataloader.cifar import get_cifar10_DVS
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
# parser.add_argument('action', default='train_ann', type=str,
#                     choices=['train_ann', 'train_snn', 'train_snn_from_zero', 'test'],
#                     help='Action: train or test.')
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST'],
                    help='the data set type.')
parser.add_argument('--batch_size', default=64, type=int, help='Batchsize')  # TODO: 观察是否可以增大
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')  # TODO: 0.001，0.0006都试一下
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=100, type=int, help='Training epochs')
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
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/log_dir', help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--pretrained_path', type=str,
                    default='/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/CIFAR10_10_Feature-Alignment_CIFAR10_enc-time_encoder_opt-Adam_lr0.001_T10_seed1000_2Edge/best_model.pth',
                    help='the path of pretrained model parameters')
parser.add_argument('--data_dir', type=str, default='/data/zhan/Event_Camera_Datasets',
                    help='Root directory for all datasets')
args = parser.parse_args()

# 参数预设值
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

# log_name = f"Temporal_Efficient_Training_with_{args.dvs_sample_ratio}_dvs_data"
log_name = f"woAP_SDSTL_CIFAR10_TET_{args.data_set}-data_set_{args.seed}-seed_{args.dvs_sample_ratio}-dvs_data_{args.dvs_encoding_type}-dvs_encoder_{args.lr}-lr_VGGSNN"

writer = SummaryWriter(log_dir=os.path.join(args.log_dir + '_' + args.data_set, log_name))
print(log_name)

model_path = os.path.join(args.checkpoint + '_' + args.data_set, f'{log_name}.pth')

if __name__ == "__main__":
    # 设置随机数种子
    seed_all(args.seed)

    # preparing data
    if args.data_set == 'CIFAR10':
        train_loader, test_loader = get_cifar10_DVS(args.batch_size, args.T,
                                                    train_set_ratio=args.dvs_sample_ratio,
                                                    encode_type=args.dvs_encoding_type)
    elif args.data_set == 'Caltech101':
        train_loader, test_loader = get_n_caltech101(args.batch_size, args.T,
                                                     train_set_ratio=args.dvs_sample_ratio,
                                                     encode_type=args.dvs_encoding_type)
    elif args.data_set == 'MNIST':
        train_loader, test_loader = get_n_mnist(args.batch_size, args.T,
                                                train_set_ratio=args.dvs_sample_ratio,
                                                encode_type=args.dvs_encoding_type)

    print("训练集DVS数量", len(train_loader) * args.batch_size)

    print("测试集DVS数量", len(test_loader) * args.batch_size)

    # preparing model
    # 根据数据集选择正确的图像尺寸
    if args.data_set == 'CIFAR10':
        img_shape = 48  # CIFAR10-DVS使用48x48（更大的特征图，保留更多细节）
        cls_num = 10
    elif args.data_set == 'MNIST':
        img_shape = 34  # N-MNIST使用34x34
        cls_num = 10
    elif args.data_set == 'Caltech101':
        img_shape = 224  # N-Caltech101使用224x224
        cls_num = 101
    else:
        raise ValueError(f"Unsupported dataset: {args.data_set}")
    
    model = VGGSNNwoAP(cls_num=cls_num, img_shape=img_shape)
    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # 加载预训练模型参数（train.py预训练的VGGSNNwoAP模型）
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
        
        # 过滤掉不匹配的键（如edge_extractor）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape 
                          and 'edge_extractor' not in k}
        
        # 更新模型参数
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个预训练参数")
        print(f"跳过的参数: {set(model_dict.keys()) - set(pretrained_dict.keys())}")
    else:
        print("未找到预训练模型，从头开始训练VGGSNNwoAP模型（Baseline实验）")

    # preparing training set
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch)

    # train
    trainer = Trainer(args, device, writer, model, optimizer, TET_loss, scheduler, model_path)
    trainer.train(train_loader, test_loader)

    # test
    test_loss, test_acc = trainer.test(test_loader)
    print('test_loss={:.5f}\t test_acc={:.3f}'.format(test_loss, test_acc))
    writer.add_scalar(tag="test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)
