# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: tl.py
@time: 2022/4/19 11:11
"""

import os
import torch
import argparse
from utils import seed_all, TLTrainer, AlignmentTLTrainerOnImageNet, set_optimizer, set_lr_scheduler
from utils.loss_function import TET_loss
from dataloader import *
from models.tl_models import ResNet
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import surrogate, neuron, functional


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('--data_set', type=str, default='ImageNet100',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100'],
                    help='the data set type.')
parser.add_argument('--num_classes', type=int, default=100,
                    help='the number of data classes.')
parser.add_argument('--batch_size', default=16, type=int, help='Batchsize')  # Cifar10: 32, MNIST: 32, Caltech101: xx
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam', 'AdamX', 'RMSprop'], help='Optimizer')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='Learning rate')  # CIFAR10: 0.0002, Caltech101: 0.0002, MNIST: 0.0001, ImageNet: 0.1
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
    dest="weight_decay",
)
parser.add_argument(
    "--norm-weight-decay",
    default=None,
    type=float,
    help="weight decay for Normalization layers (default: None, same value as --wd)",
)
parser.add_argument(
    "--label-smoothing", default=0.1, type=float, help="label smoothing (default: 0.1)", dest="label_smoothing"
)
parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.2)")
parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix alpha (default: 1.0)")
parser.add_argument("--lr-scheduler", default="cosa", type=str, help="the lr scheduler (default: cosa)")
parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="the number of epochs to warmup (default: 5)")
parser.add_argument(
    "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)"
)
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    help="Use pre-trained models from the modelzoo",
    action="store_true",
)
parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
parser.add_argument("--amp", default=False, type=bool, help="whether use automatic mixed precision training")
parser.add_argument('--cupy', default=True, type=bool, help="set the neurons to use cupy backend")
parser.add_argument("--auto-augment", default='ta_wide', type=str, help="auto augment policy (default: ta_wide)")
parser.add_argument("--random-erase", default=0.1, type=float, help="random erasing probability (default: 0.1)")
parser.add_argument('--epoch', default=200, type=int, help='Training epochs')
parser.add_argument('--id', default='test', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 16)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=2000, help='seed for initializing training. ')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='the ratio of used dvs training set. ')
parser.add_argument('--dvs_encoding_type', type=str, default='TET', choices=['TET', 'spikingjelly'])
parser.add_argument('--model', type=str, default='spiking_resnet18')
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='the transfer loss for transfer learning.')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='the transfer loss for transfer learning.')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float, metavar='N',
                    help='encoder transfer learning loss ratio')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float, metavar='N',
                    help='feature transfer learning loss ratio')
parser.add_argument('--log_dir', type=str, default='./log_dir/tl', help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/tl', help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')

args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{args.GPU_id}")

log_name = f"transfer_learning_with_{args.data_set}-data_set_{args.encoder_tl_loss_type}-en_loss_type_{args.feature_tl_loss_type}-fea_loss_type_{args.encoder_tl_lamb}-en_tl_lamb{args.feature_tl_lamb}-fea_tl_lamb_{args.seed}-seed_{args.RGB_sample_ratio:.2f}-RGB_data_{args.dvs_sample_ratio}-dvs_data_{args.encoder_type}-source_encoder_{args.dvs_encoding_type}-dvs_encoder_{args.optim}-optim_{args.lr}-lr"
writer = SummaryWriter(log_dir=os.path.join(args.log_dir + '_' + args.data_set, log_name))
print(log_name)

model_path = os.path.join(args.checkpoint + '_' + args.data_set, f'{log_name}.pth')

if __name__ == "__main__":
    seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_{args.RGB_sample_ratio}_grid_result.txt", "a")

    # preparing data
    if args.data_set == 'CIFAR10':
        train_loader, dvs_test_loader = get_tl_cifar10(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
        # train_loader, test_loader = get_cifar10(args.batch_size,args.RGB_sample_ratio)
        # dvs_train_loader, dvs_test_loader = get_cifar10_DVS(args.batch_size, args.T,
        #                                                 train_set_ratio=args.dvs_sample_ratio,
        #                                                 encode_type=args.dvs_encoding_type)
    elif args.data_set == 'Caltech101':
        train_loader, dvs_test_loader = get_tl_caltech101(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
        # train_loader, test_loader = get_caltech101(args.batch_size,args.RGB_sample_ratio)
        # dvs_train_loader, dvs_test_loader = get_n_caltech101(args.batch_size, args.T,
        #                                                 train_set_ratio=args.dvs_sample_ratio,
        #                                                 encode_type=args.dvs_encoding_type)
    elif args.data_set == 'MNIST':
        train_loader, dvs_test_loader = get_tl_mnist(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
        # train_loader, test_loader = get_mnist(args.batch_size,args.RGB_sample_ratio)
        # dvs_train_loader, dvs_test_loader = get_n_mnist(args.batch_size, args.T,
        #                                                 train_set_ratio=args.dvs_sample_ratio,
        #                                                 encode_type=args.dvs_encoding_type)
    elif args.data_set == 'ImageNet100':
        train_loader, dvs_val_loader, dvs_test_loader_list = get_tl_imagenet100(args.batch_size, args.RGB_sample_ratio,
                                                                                args.dvs_sample_ratio, args.seed, args.num_classes)
        # train_loader, test_loader = get_mnist(args.batch_size,args.RGB_sample_ratio)
        # dvs_train_loader, dvs_test_loader = get_n_mnist(args.batch_size, args.T,
        #                                                 train_set_ratio=args.dvs_sample_ratio,
        #                                                 encode_type=args.dvs_encoding_type)

    print("训练集RGB数量", train_loader.get_len()[0])
    print("训练集DVS数量", train_loader.get_len()[1])

    print("验证集DVS数量", dvs_val_loader.get_len()[1])

    # print("测试集RGB数量", dvs_test_loader_list[0].get_len()[0] * len(dvs_test_loader_list))
    print("测试集DVS数量", dvs_test_loader_list[0].get_len()[1] * len(dvs_test_loader_list))

    # preparing model
    if args.model in ResNet.__all__:
        model = ResNet.__dict__[args.model](pretrained=args.pretrained, spiking_neuron=neuron.IFNode,
                                            surrogate_function=surrogate.ATan(), detach_reset=True,
                                            num_classes=args.num_classes, tl=True)
        functional.set_step_mode(model, step_mode='m')
        if args.cupy:
            functional.set_backend(model, 'cupy', neuron.IFNode)
    else:
        raise ValueError(f"args.model should be one of {ResNet.__all__}")

    model.to(device)

    # preparing training set
    if args.norm_weight_decay is None:
        parameters = model.parameters()
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
    optimizer = set_optimizer(args.optim, args.lr, args.momentum, args.weight_decay, parameters)

    scheduler = set_lr_scheduler(args.lr_scheduler, args.epoch, optimizer,
                                 lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma)

    # train
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    trainer = AlignmentTLTrainerOnImageNet(args, device, writer, model, optimizer, criterion, scheduler, model_path)
    best_train_acc, best_val_acc = trainer.train(train_loader, dvs_val_loader)

    # test
    test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader_list)
    if type(test_loss) is list:
        all_test_loss = test_loss
        test_loss = sum(test_loss) / len(test_loss)
        all_test_acc1 = test_acc1
        test_acc1 = sum(test_acc1) / len(test_acc1)
        all_test_acc5 = test_acc5
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
