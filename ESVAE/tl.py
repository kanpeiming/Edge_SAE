# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: tl.py
@time: 2022/4/19 11:11
"""

'''
加载预训练参数，利用纯DVS数据进行分类。
'''

import os
import torch
import argparse

from tl_utils.common_utils import seed_all
from tl_utils.trainer import *  # 迁移训练trainer
from tl_utils.loss_function import TET_loss
from dataloader.cifar import *
from dataloader.mnist import *
from torch.utils.tensorboard import SummaryWriter
# from models.tl_models.tl_MNIST import *
from pretrain.pretrainModel import *
# from pretrain.pretrainer import *    # 结构解耦迁移trainer


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100',
                             'CINIC10_WO_CIFAR10', 'ImageNet2Caltech', 'Caltech51'],
                    help='the data set type.')
parser.add_argument('--batch_size', default=32, type=int, help='Batchsize')  # Cifar10: 32, MNIST: 32, Caltech101: xx
parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')  # CIFAR10: 0.0002, Caltech101: 0.0002, MNIST: 0.0001
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=120, type=int, help='Training epochs')
parser.add_argument('--id', default='test', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 10)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='the ratio of used dvs training set. ')
parser.add_argument('--dvs_encoding_type', type=str, default='TET', choices=['TET', 'spikingjelly'])
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
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/tl/checkpoints',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/tl/log_dir',
                    help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')
parser.add_argument('--num_classes', type=int, default=10, help='the number of data classes.')

args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{args.GPU_id}")

log_name = f"RGBToDvs-transfer_learning_with_{args.data_set}-data_set_{args.encoder_tl_loss_type}-en_loss_type_{args.feature_tl_loss_type}-fea_loss_type_{args.encoder_tl_lamb}-en_tl_lamb{args.feature_tl_lamb}-fea_tl_lamb_{args.seed}-seed_{args.RGB_sample_ratio}-RGB_data_{args.dvs_sample_ratio}-dvs_data_{args.encoder_type}-source_encoder_{args.dvs_encoding_type}-dvs_encoder_{args.optim}-optim_{args.lr}-lr"
# log_name = 'test'
writer = SummaryWriter(log_dir=os.path.join(f'{args.log_dir}_{args.data_set}', log_name))
model_path = os.path.join(f'{args.checkpoint}', f'{log_name}.pth')

if not os.path.exists(writer.log_dir):
    os.mkdir(writer.log_dir)
if not os.path.exists(f'{args.checkpoint}'):
    os.mkdir(f'{args.checkpoint}')

print(log_name)
print(writer.log_dir)

if __name__ == "__main__":
    seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_{args.RGB_sample_ratio}_grid_result.txt", "a")

    # preparing data
    if args.data_set == 'CIFAR10':
        train_loader, dvs_test_loader = get_tl_cifar10(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
        print("训练集RGB数量", train_loader.get_len()[0])
        print("训练集DVS数量", train_loader.get_len()[1])
        print("测试集DVS数量", dvs_test_loader.get_len()[1])
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
    # elif args.data_set == 'MNIST':
    #     train_loader, dvs_test_loader = get_tl_mnist(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
    #     print("训练集RGB数量", train_loader.get_len()[0])
    #     print("训练集DVS数量", train_loader.get_len()[1])
    #     print("测试集DVS数量", dvs_test_loader.get_len()[1])
    # elif args.data_set == 'ImageNet100':
    #     train_loader, dvs_val_loader, dvs_test_loader_list = get_tl_imagenet100(args.batch_size, args.RGB_sample_ratio,
    #                                                                             args.dvs_sample_ratio, args.seed,
    #                                                                             args.num_classes)
    #     print("训练集RGB数量", train_loader.get_len()[0])
    #     print("训练集DVS数量", train_loader.get_len()[1])
    #     print("验证集DVS数量", dvs_val_loader.get_len()[1])
    #     print("测试集DVS数量", dvs_test_loader_list[0].get_len()[1] * len(dvs_test_loader_list))

    # preparing model   从这里导入预训练的模型
    init_device = torch.device("cuda:0")
    model = VGGSNN(cls_num=10, img_shape=32, device='cuda')

    # 加载预训练参数
    pretrained_model_path = '/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/CIFAR10_10_Feature-Alignment_CIFAR10_enc-time_encoder_opt-Adam_lr0.001_T10_seed1000_2Edge/best_model.pth'  # 替换为你的模型文件路径
    # model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    # pretrained_model = torch.load(pretrained_model_path, weights_only=True)
    model_dict = model.state_dict()
    pretrained_model_dict = {key: value for key, value in model_dict.items() if (key in model_dict and 'bottleneck' not in key and 'classifier' not in key and 'rgb_input' not in key and 'dvs_input' not in key )}
    model_dict.update(pretrained_model_dict)
    model.load_state_dict(model_dict)
    model.load_state_dict(torch.load(pretrained_model_path, weights_only=True), strict=False)

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

    # train
    trainer = AlignmentTLTrainer_Edge_1(args, device, writer, model, optimizer, TET_loss, scheduler, model_path)
    if args.data_set == 'ImageNet100':
        best_train_acc, best_val_acc = trainer.train(train_loader, dvs_val_loader)
        # test
        test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader_list)
    else:
        best_train_acc, best_val_acc = trainer.train(train_loader, dvs_test_loader)
        # test
        test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader)

    if type(test_loss) is list:
        all_test_loss = test_loss
        test_loss = sum(test_loss) / len(test_loss)
        all_test_acc1 = test_acc1
        test_acc1 = sum(test_acc1) / len(test_acc1)
        all_test_acc5 = test_acc5
        print(test_acc5)
        test_acc5 = sum(test_acc5) / len(test_acc5)
        print(f'test_loss={test_loss:.5f} test_acc1={test_acc1:.3f} all_test_acc1={all_test_acc1} '
              f'val_acc5={test_acc5:.4f} all_val_acc5={all_test_acc5}')
        writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
        writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
        writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)
        for test_id in range(len(all_test_loss)):
            writer.add_scalar(tag=f"test{test_id + 1}/accuracy1", scalar_value=all_test_acc1[test_id],
                              global_step=0)
            writer.add_scalar(tag=f"test{test_id + 1}/accuracy5", scalar_value=all_test_acc5[test_id],
                              global_step=0)
            writer.add_scalar(tag=f"test{test_id + 1}/loss", scalar_value=all_test_loss[test_id],
                              global_step=0)
    else:
        print(f'test_loss={test_loss:.5f} test_acc1={test_acc1:.3f} val_acc5={test_acc5:.4f}')
        writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
        writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
        writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)

    write_content = f'seed: {args.seed} \n' \
                    f'encoder_tl_loss: {args.encoder_tl_lamb} * {args.encoder_tl_loss_type} \n' \
                    f'feature_tl_loss: {args.feature_tl_lamb} * {args.feature_tl_loss_type} \n' \
                    f'best_train_acc: {best_train_acc}, best_val_acc: {best_val_acc} \n\n'
    f.write(write_content)
    f.close()
