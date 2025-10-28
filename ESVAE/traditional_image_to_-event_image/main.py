# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: main.py
@time: 2022/4/19 11:11
"""

import os
import torch
import argparse
from utils import seed_all, Trainer
from utils.loss_function import TET_loss
from dataloader import get_cifar10_DVS, get_n_mnist, get_n_caltech101
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
# parser.add_argument('action', default='train_ann', type=str,
#                     choices=['train_ann', 'train_snn', 'train_snn_from_zero', 'test'],
#                     help='Action: train or test.')
parser.add_argument('--data_set',type=str, default='MNIST',
                    choices=['CIFAR10','Caltech101','MNIST'],
                    help='the data set type.')
parser.add_argument('--batch_size', default=32, type=int, help='Batchsize')   # TODO: 观察是否可以增大
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')  # TODO: 0.001，0.0006都试一下
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=150, type=int, help='Training epochs')
# parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--id', default='test', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 16)')
parser.add_argument('--encoder_type', type=str, default='lap_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=0.1, help='the ratio of used dvs training set. ')  # TODO: 注意观察该处数值
parser.add_argument('--dvs_encoding_type', type=str, default='TET', choices=['TET', 'spikingjelly'])
parser.add_argument('--model', type=str, default='vgg16')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N', help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--log_dir', type=str, default='./log_dir/main', help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/main', help='the path of checkpoint dir.')

args = parser.parse_args()

# 参数预设值
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

# log_name = f"Temporal_Efficient_Training_with_{args.dvs_sample_ratio}_dvs_data"
log_name = f"Temporal_Efficient_Training_with_{args.data_set}-data_set_{args.seed}-seed_{args.dvs_sample_ratio}-dvs_data_{args.dvs_encoding_type}-dvs_encoder_{args.lr}-lr"

writer = SummaryWriter(log_dir=os.path.join(args.log_dir + '_' +args.data_set, log_name))
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
    
    print("训练集DVS数量",len(train_loader)*args.batch_size)

    print("测试集DVS数量",len(test_loader)*args.batch_size)

    # preparing model
    model = VGGSNNwoAP(2, 101, 48)  # Caltech101: (2, 101, 224); MNIST、Cifar10: (2, 10, 48)
    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

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
