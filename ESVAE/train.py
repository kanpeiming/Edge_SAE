# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: tl.py
@time: 2022/4/19 11:11
"""

import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter

import dataloader.cifar
from dataloader.mnist import *
from pretrain.pretrainer import *
from pretrain.pretrainModel import *
from tl_utils import common_utils
from tl_utils.loss_function import TET_loss

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100',
                             'CINIC10_WO_CIFAR10', 'ImageNet2Caltech', 'Caltech51'],
                    help='the data set type.')
parser.add_argument('--batch_size', default=32, type=int, help='Batchsize')  # Cifar10: 32, MNIST: 32, Caltech101: xx
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')  # CIFAR10: 0.0002, Caltech101: 0.0002, MNIST: 0.0001
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
# parser.add_argument('--id', default='test', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 10)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
# parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set. ')
# parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='the ratio of used dvs training set. ')
# parser.add_argument('--dvs_encoding_type', type=str, default='TET', choices=['TET', 'spikingjelly'])
# parser.add_argument('--model', type=str, default='vgg16')
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='the transfer loss for transfer learning.')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='the transfer loss for transfer learning.')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float, metavar='N',
                    help='encoder transfer learning loss ratio')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float, metavar='N',
                    help='feature transfer learning loss ratio')
parser.add_argument('--log_dir', type=str, default='/data/kpm/pretrain/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/data/kpm/pretrain/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')
parser.add_argument('--num_classes', type=int, default=10, help='the number of data classes.')

args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{args.GPU_id}")

log_name = (
    f"Feature-Alignment_{args.data_set}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"2Edge"  # 加上这个标记，表示使用了两个 edge_extractor
)

# 对应的路径设置
# 正确拼接路径
log_dir = os.path.join(
    args.log_dir,
    f"FeatureAlign_{args.data_set}_{args.num_classes}",
    log_name
)
# 模型保存路径也加上标记
checkpoint_dir = os.path.join(
    args.checkpoint,
    f"{args.data_set}_{args.num_classes}_{log_name}"  # 增加唯一性
)

# 递归创建目录（无需检查是否存在）
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 设置模型保存路径
model_path = checkpoint_dir
writer = SummaryWriter(log_dir=log_dir)

print(log_name)
print(writer.log_dir)

if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_grid_result.txt", "a")

    # preparing data
    if args.data_set == 'CIFAR10':
        train_loader = dataloader.cifar.get_cifar10(32, 1.0)
        print("训练集RGB数量", train_loader.get_len())
        # print("RGB数量转换为边缘图的数量", target_train_loader.get_len())
        # print("训练集DVS数量", train_loader.get_len()[1])
        # print("测试集DVS数量", dvs_test_loader.get_len()[1])
    # elif args.data_set == 'CINIC10_WO_CIFAR10':
    #     train_loader, dvs_test_loader = get_tl_cinic10_wo_cifar10(args.batch_size, args.RGB_sample_ratio,
    #                                                               args.dvs_sample_ratio)
    #     print("训练集RGB数量", train_loader.get_len()[0])
    #     print("训练集DVS数量", train_loader.get_len()[1])
    #     print("测试集DVS数量", dvs_test_loader.get_len()[1])
    # elif args.data_set == 'ImageNet2Caltech':
    #     train_loader, dvs_test_loader = get_tl_imagenet2caltech(args.batch_size, args.RGB_sample_ratio,
    #                                                             args.dvs_sample_ratio)
    #     print("训练集RGB数量", train_loader.get_len()[0])
    #     print("训练集DVS数量", train_loader.get_len()[1])
    #     print("测试集DVS数量", dvs_test_loader.get_len()[1])
    # elif args.data_set == 'Caltech51':
    #     train_loader, dvs_test_loader = get_tl_imagenet2caltech(args.batch_size, args.RGB_sample_ratio,
    #                                                             args.dvs_sample_ratio)
    #     print("训练集RGB数量", train_loader.get_len()[0])
    #     print("训练集DVS数量", train_loader.get_len()[1])
    #     print("测试集DVS数量", dvs_test_loader.get_len()[1])
    # elif args.data_set == 'Caltech101':
    #     train_loader, dvs_test_loader = get_tl_caltech101(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
    #     print("训练集RGB数量", train_loader.get_len()[0])
    #     print("训练集DVS数量", train_loader.get_len()[1])
    #     print("测试集DVS数量", dvs_test_loader.get_len()[1])
    elif args.data_set == 'MNIST':
        train_loader = get_mnist(args.batch_size, 1.0)
        print("训练集RGB数量", len(train_loader.dataset))
    # elif args.data_set == 'ImageNet100':
    #     train_loader, dvs_val_loader, dvs_test_loader_list = get_tl_imagenet100(args.batch_size, args.RGB_sample_ratio,
    #                                                                             args.dvs_sample_ratio, args.seed,
    #                                                                             args.num_classes)
    #     print("训练集RGB数量", train_loader.get_len()[0])
    #     print("训练集DVS数量", train_loader.get_len()[1])
    #     print("验证集DVS数量", dvs_val_loader.get_len()[1])
    #     print("测试集DVS数量", dvs_test_loader_list[0].get_len()[1] * len(dvs_test_loader_list))

    # preparing model  选择模型
    model = VGGSNN(cls_num=10, img_shape=32, device='cuda')

    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # preparing training set
    if args.optim == 'Adam':
        # optimizer = torch.optim.Adam(
        #     [{'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10},
        #      {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr * 1}]
        # )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            [{'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10,
              'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False},
             {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr * 1,
              'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False}]
        )
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=False)
    else:
        raise Exception(f"The value of optim should in ['SGD', 'Adam'], "
                        f"and your input is {args.optim}")
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=10)

    if args.data_set == 'CIFAR10':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif args.data_set == 'CINIC10_WO_CIFAR10':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'ImageNet2Caltech':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'Caltech101':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'MNIST':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.data_set == 'ImageNet100':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        raise Exception(
            f"The value of data_set should in ['CIFAR10', 'CINIC10_WO_CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100'], "
            f"and your input is {args.data_set}")

    # 训练（使用修改后的训练器）
    trainer = AlignmentTLTrainer_Edge_1(args, device, writer, model, optimizer, TET_loss, scheduler, model_path)

    if args.data_set == 'ImageNet100':
        best_train_acc, best_train_loss = trainer.train(train_loader)
        # test using a list of test loaders
        test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader_list)
    else:
        best_train_acc, best_train_loss = trainer.train(train_loader)
        # test using a single test loader
        test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader)

    if type(test_loss) is list:
        all_test_loss = test_loss
        test_loss = sum(test_loss) / len(test_loss)
        all_test_acc1 = test_acc1
        test_acc1 = sum(test_acc1) / len(test_acc1)
        all_test_acc5 = test_acc5
        test_acc5 = sum(test_acc5) / len(test_acc5)
        print(f'test_loss={test_loss:.5f} test_acc1={test_acc1:.3f} test_acc5={test_acc5:.4f}')
        writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
        writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
        writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)
        for test_id in range(len(all_test_loss)):
            writer.add_scalar(tag=f"test{test_id + 1}/accuracy1", scalar_value=all_test_acc1[test_id], global_step=0)
            writer.add_scalar(tag=f"test{test_id + 1}/accuracy5", scalar_value=all_test_acc5[test_id], global_step=0)
            writer.add_scalar(tag=f"test{test_id + 1}/loss", scalar_value=all_test_loss[test_id], global_step=0)
    else:
        print(f'test_loss={test_loss:.5f} test_acc1={test_acc1:.3f} test_acc5={test_acc5:.4f}')
        writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
        writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
        writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)

    write_content = (
        f'seed: {args.seed} \n'
        f'encoder_tl_loss: {args.encoder_tl_lamb} * {args.encoder_tl_loss_type} \n'
        f'feature_tl_loss: {args.feature_tl_lamb} * {args.feature_tl_loss_type} \n'
        f'best_train_acc: {best_train_acc}, best_train_loss: {best_train_loss} \n\n'
    )
    f.write(write_content)
    f.close()
