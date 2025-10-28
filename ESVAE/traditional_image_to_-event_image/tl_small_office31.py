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
from utils import seed_all, TLTrainer
from utils.loss_function import TET_loss
from dataloader import get_small_office31
from models.tl_models.CifarNet import CifarNetSNN
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('--batch_size', default=16, type=int, help='Batchsize')
parser.add_argument('--lr', default=0.0006, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=150, type=int, help='Training epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=True, type=bool, help='Whether to use multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='snn simulation time (default: 16)')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='the encoder type of rgb data for snn.')
parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
parser.add_argument('--model', type=str, default='vgg16')
parser.add_argument('--source_name', type=str, default='amazon')
parser.add_argument('--target_name', type=str, default='webcam')
parser.add_argument('--CKA_type', type=str, default='mem', choices=['mem', 'spike'])
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N', help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--tl_lamb', default=0.2, type=float, metavar='N', help='transfer learning loss ratio')
parser.add_argument('--log_dir', type=str, default='./log_dir', help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='the path of checkpoint dir.')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_name = f"transfer_learning_with_on_small_office31_{args.source_name}2{args.target_name}_{args.encoder_type}-source_encoder_{args.lr}-lr-tl_lamb-{args.tl_lamb}-CKA_type-{args.CKA_type}"
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, log_name))
print(log_name)

model_path = os.path.join(args.checkpoint, f'{log_name}.pth')


if __name__ == "__main__":
    seed_all(args.seed)

    # preparing data
    source_train_dataloader, target_train_dataloader, target_test_dataloader = get_small_office31(args.batch_size,
                                                                                                  args.source_name,
                                                                                                  args.target_name)

    # preparing model
    model = CifarNetSNN()
    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # preparing training set
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch)

    # train
    trainer = TLTrainer(args, device, writer, model, optimizer, TET_loss, scheduler, model_path)
    trainer.train(source_train_dataloader, target_train_dataloader, target_test_dataloader)

    # test
    test_loss, test_acc = trainer.test(target_test_dataloader)
    print('test_loss={:.5f}\t test_acc={:.3f}'.format(test_loss, test_acc))
    writer.add_scalar(tag="test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)
