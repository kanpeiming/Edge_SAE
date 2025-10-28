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
from utils import seed_all, ANN2SNNTrainer
from utils.loss_function import TET_loss
from utils.ann2snn_utils import replace_activation_by_floor, replace_activation_by_neuron, regular_set
from dataloader import get_cifar10_DVS, get_cifar10
from models.ann_models.VGG import tet_vgg, tet_vgg_woap
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MultiheadAttention


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
# parser.add_argument('action', default='train_ann', type=str,
#                     choices=['train_ann', 'train_snn', 'train_snn_from_zero', 'test'],
#                     help='Action: train or test.')
parser.add_argument('--batch_size', default=100, type=int, help='Batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=150, type=int, help='Training epochs')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--id', default='test', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=True, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 16)')
parser.add_argument('--encoder_type', type=str, default='poison_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=0.2, help='the ratio of used dvs training set. ')
parser.add_argument('--dvs_encoding_type', type=str, default='TET', choices=['TET', 'spikingjelly'])
parser.add_argument('--model', type=str, default='vgg16')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N', help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--log_dir', type=str, default='./log_dir', help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='the path of checkpoint dir.')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_name = f"ann2snn_Training_with_{args.dvs_sample_ratio}_dvs_data_train_on_ann"
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, log_name))
print(log_name)

model_path = os.path.join(args.checkpoint, f'{log_name}.pth')


if __name__ == "__main__":
    seed_all(args.seed)

    # preparing data
    train_loader, test_loader = get_cifar10(args.batch_size)
    dvs_train_loader, dvs_test_loader = get_cifar10_DVS(args.batch_size, args.T,
                                                        train_set_ratio=args.dvs_sample_ratio,
                                                        encode_type=args.dvs_encoding_type)

    # preparing model
    model = tet_vgg_woap()
    model = replace_activation_by_floor(model, t=args.T)

    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # preparing training set
    criterion = torch.nn.CrossEntropyLoss().to(device)
    para1, para2, para3 = regular_set(model)
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': args.weight_decay},
                                {'params': para2, 'weight_decay': args.weight_decay},
                                {'params': para3, 'weight_decay': args.weight_decay}
                                ],
                                lr=args.lr,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch*2)

    # train on rgb
    trainer = ANN2SNNTrainer(args, device, writer, model, optimizer, criterion, scheduler, model_path)
    trainer.train_ann(train_loader, test_loader)

    # train on dvs
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer.network = replace_activation_by_neuron(trainer.network, device)
    trainer.train_snn(dvs_train_loader, dvs_test_loader)

    # test on rgb and dvs
    # trainer.network = replace_activation_by_neuron(trainer.network, device)
    test_loss, test_acc, test_tot = trainer.test_snn(dvs_test_loader)
    print('test_loss={:.5f} test_acc={:.3f} test_tot={}'.format(test_loss, test_acc, test_tot))
    writer.add_scalar(tag="test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)
