# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: trainer.py
@time: 2022/4/19 17:08
"""
import os
import gc
import time
import torch
import itertools
from tqdm import tqdm
from tl_utils.common_utils import LapPoissonEncoder, MyPoissonEncoder, TimeEncoder, accuracy

try:
    from spikingjelly.activation_based.functional import reset_net
except:
    from utils.common_utils import reset_net
import psutil
from torch import nn
from pretrain.Edge import *


class Trainer(object):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        self.args = args
        self.device = device
        self.writer = writer
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_path = model_path

        self.best_train_acc = 0
        self.best_val_acc = 0

        self.encoder_dict = {'poison_encoder': MyPoissonEncoder(self.args.T, self.device),
                             'lap_encoder': LapPoissonEncoder(self.args.T, self.device),
                             'time_encoder': TimeEncoder(self.args.T, self.device)}

        self.state = self.prepare_state()

    def prepare_state(self):
        """
        save the args parameters to state dict, prepare for model saving.
        :return:
        """
        state = {}
        arg_names = [a for a in dir(self.args) if not a.startswith('__') and not callable(getattr(self.args, a))]
        for name in arg_names:
            state[name] = eval('self.args.' + name)
        return state

    def train(self, train_loader, val_loader, second_loader=None):
        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_loss = 0
            train_num = 0
            train_correct = 0

            # 添加tqdm进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f'Epoch [{epoch+1}/{self.args.epoch}]', 
                       leave=True)
            
            for i, (data, labels) in pbar:
                self.optimizer.zero_grad()

                # 若输入为rgb图片，则转换为2通道的脉冲编码，(N, 3, H, W) -> (N, T, 2, H, W)
                if len(data.shape) == 4:
                    data = self.encoder_dict[self.args.encoder_type](data, out_channel=2)

                data, labels = data.to(self.device), labels.to(self.device)

                outputs = self.network(data.float())
                mean_out = outputs.mean(1)
                loss = self.criterion(outputs, labels)

                train_loss += loss.item()
                loss.mean().backward()
                self.optimizer.step()
                # self.scheduler.step()

                train_num += float(labels.size(0))
                _, predicted = mean_out.cpu().max(1)
                train_correct += float(predicted.eq(labels.cpu()).sum().item())
                
                # 更新进度条信息
                current_loss = train_loss / (i + 1)
                current_acc = train_correct / train_num
                pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.3f}'})
            
            pbar.close()
            
            self.scheduler.step()
            train_acc = train_correct / train_num
            train_loss = train_loss / train_num
            print('Epoch:[{}/{}]\t time cost: {:.2f}min\t train_loss={:.5f}\t train_acc={:.3f}'.format(epoch,
                                                                                                       self.args.epoch,
                                                                                                       (
                                                                                                               time.time() - start) / 60,
                                                                                                       train_loss,
                                                                                                       train_acc))

            val_loss, val_acc = self.test(val_loader)
            print('Epoch:[{}/{}]\t val_loss={:.5f}\t val_acc={:.3f}'.format(epoch, self.args.epoch,
                                                                            val_loss, val_acc))
            if second_loader:
                second_loss, second_acc = self.test(second_loader)
                print('Epoch:[{}/{}]\t second_loss={:.5f}\t second_acc={:.3f}'.format(epoch, self.args.epoch,
                                                                                      second_loss, second_acc))

            self.writer.add_scalar(tag="train/accuracy", scalar_value=train_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=train_loss, global_step=epoch)
            self.writer.add_scalar(tag="val/accuracy", scalar_value=val_acc, global_step=epoch)
            self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < train_acc:
                self.best_train_acc = train_acc
                self.save_model(epoch)

            if self.best_val_acc < val_acc:
                self.best_val_acc = val_acc

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
        return self.best_train_acc

    def test(self, test_loader):
        self.network.eval()
        test_loss = 0
        test_num = 0
        test_correct = 0

        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                if len(data.shape) == 4:
                    data = self.encoder_dict[self.args.encoder_type](data)

                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.network(data)
                mean_out = outputs.mean(1)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                test_num += float(labels.size(0))
                _, predicted = mean_out.cpu().max(1)
                test_correct += float(predicted.eq(labels.cpu()).sum().item())
        return test_loss / test_num, test_correct / test_num

    def save_model(self, epoch):
        # model saving
        print('Saving..')
        self.state['net'] = self.network.state_dict()
        self.state['best_test_acc'] = self.best_train_acc
        self.state['epoch'] = epoch

        # if not os.path.isdir('checkpoints'+ '_' + self.args.data_set):
        #     os.mkdir('checkpoints'+ '_' + self.args.data_set)
        if not os.path.isdir(self.args.checkpoint + '_' + self.args.data_set):
            os.mkdir(self.args.checkpoint + '_' + self.args.data_set)
        torch.save(self.state, self.model_path, _use_new_zipfile_serialization=False)

    def load_model(self):
        print(f"load saved trained model: {self.model_path}")
        saved_data = torch.load(self.model_path)
        net = saved_data['net']
        self.network.load_state_dict(net)


class TLTrainer(Trainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(TLTrainer, self).__init__(args, device, writer, network, optimizer, criterion, scheduler, model_path)

    def train(self, source_train_loader, target_train_loader, target_val_loader):

        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_num = 1
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            target_train_loss = 0
            target_train_correct = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            # 获取最大dataloader的长度，并以此长度获取数据，较小的dataloader从头重新加入循环遍历

            max_dataloader_len = max(len(target_train_loader), len(source_train_loader))
            source_data_iter = iter(source_train_loader)
            target_data_iter = iter(target_train_loader)

            for i in range(max_dataloader_len):
                # 匹配源域训练集
                try:
                    source_data, source_labels = source_data_iter.next()
                except StopIteration:
                    source_data_iter = iter(source_train_loader)
                    source_data, source_labels = source_data_iter.next()
                # 匹配目标域训练集
                try:
                    target_data, target_labels = target_data_iter.next()
                except StopIteration:
                    target_data_iter = iter(target_train_loader)
                    target_data, target_labels = target_data_iter.next()

                # target_train_iter = itertools.cycle(iter(target_train_loader))

                # for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_train_loader, target_train_iter)):
                self.optimizer.zero_grad()

                if source_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if target_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    target_data = self.encoder_dict[self.args.encoder_type](target_data)
                source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)

                source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(source_data.float(),
                                                                                                target_data.float(),
                                                                                                self.args.encoder_tl_loss_type,
                                                                                                self.args.feature_tl_loss_type)
                source_mean_out = source_outputs.mean(1)
                source_clf_loss = self.criterion(source_outputs, source_labels)

                target_mean_out = target_outputs.mean(1)
                target_clf_loss = self.criterion(target_outputs, target_labels)

                loss = source_clf_loss + target_clf_loss
                if self.args.encoder_tl_lamb > 0.0:
                    loss += self.args.encoder_tl_lamb * encoder_tl_loss
                if self.args.feature_tl_lamb > 0.0:
                    loss += self.args.feature_tl_lamb * feature_tl_loss

                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_encoder_tl_loss += encoder_tl_loss.item()
                total_feature_tl_loss += feature_tl_loss.item()
                total_loss += loss.item()

                loss.mean().backward()
                self.optimizer.step()
                self.scheduler.step()

                train_num += float(source_labels.size(0))

                _, source_predicted = source_mean_out.cpu().max(1)
                source_train_correct += float(source_predicted.eq(source_labels.cpu()).sum().item())

                _, target_predicted = target_mean_out.cpu().max(1)
                target_train_correct += float(target_predicted.eq(target_labels.cpu()).sum().item())

                reset_net(self.network)

            # self.scheduler.step()
            source_train_acc = source_train_correct / train_num
            target_train_acc = target_train_correct / train_num
            total_acc = (source_train_acc + target_train_acc) / 2
            source_train_loss = source_train_loss / train_num
            target_train_loss = target_train_loss / train_num
            total_encoder_tl_loss = total_encoder_tl_loss / train_num
            total_feature_tl_loss = total_feature_tl_loss / train_num
            total_loss = total_loss / train_num
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  'source_clf_loss={:.5f} source_train_acc={:.3f} '
                  'target_clf_loss={:.5f} target_train_acc={:.3f} '
                  'total_loss={:.5f} train_acc={:.3f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(epoch, self.args.epoch,
                                                                         (time.time() - start) / 60,
                                                                         source_train_loss, source_train_acc,
                                                                         target_train_loss, target_train_acc,
                                                                         total_loss, total_acc,
                                                                         total_feature_tl_loss, total_feature_tl_loss))

            val_loss, val_acc = self.test(target_val_loader)
            print('Epoch:[{}/{}]\t val_loss={:.5f}\t val_acc={:.3f}'.format(epoch, self.args.epoch,
                                                                            val_loss, val_acc))

            self.writer.add_scalar(tag="train/source_accuracy", scalar_value=source_train_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/target_accuracy", scalar_value=target_train_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/accuracy", scalar_value=total_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)
            self.writer.add_scalar(tag="val/accuracy", scalar_value=val_acc, global_step=epoch)
            self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < target_train_acc:
                self.best_train_acc = target_train_acc

            if self.best_val_acc < val_acc:
                self.best_val_acc = val_acc
                self.save_model(epoch)

            # 较小的dataloader参与重新遍历后大概率无法恰好完成所有数据的读取，需要继续完成迭代以释放内存
            if len(target_train_loader) > len(source_train_loader):
                while 1:
                    try:
                        source_data_iter.next()
                    except StopIteration:
                        break
            elif len(target_train_loader) < len(source_train_loader):
                while 1:
                    try:
                        target_data_iter.next()
                    except StopIteration:
                        break
            else:
                pass

            # print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
            # torch.cuda.empty_cache()
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

            info = psutil.virtual_memory()
            print(info)

            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")
        return self.best_train_acc

    def test(self, test_loader):
        self.network.eval()
        if type(test_loader) is list:
            test_loss_list = [0] * len(test_loader)
            test_num_list = [0] * len(test_loader)
            test_correct_list = [0] * len(test_loader)
            test_correct5_list = [0] * len(test_loader)

            with torch.no_grad():
                for i, loader in enumerate(test_loader):
                    for _, (data, labels) in enumerate(loader):
                        if data.shape[1] == 3:
                            data = self.encoder_dict[self.args.encoder_type](data)
                        data, labels = data.to(self.device), labels.to(self.device)
                        outputs = self.network(data, data)
                        mean_out = outputs.mean(1)
                        loss = self.criterion(outputs, labels)
                        test_loss_list[i] += loss.item()
                        test_num_list[i] += float(labels.size(0))
                        test_acc1, test_acc5 = accuracy(mean_out, labels, topk=(1, 5))
                        test_correct_list[i] += test_acc1.item()
                        test_correct5_list[i] += test_acc5.item()
                        reset_net(self.network)
                    test_loss_list[i] /= test_num_list[i]
                    test_correct_list[i] /= test_num_list[i]
                    test_correct5_list[i] /= test_num_list[i]

            return test_loss_list, test_correct_list, test_correct5_list
        else:
            test_loss = 0
            test_num = 0
            test_correct = 0
            test_correct5 = 0

            with torch.no_grad():
                for i, (data, labels) in enumerate(test_loader):
                    if data.shape[1] == 3:
                        data = self.encoder_dict[self.args.encoder_type](data)
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.network(data, data)
                    mean_out = outputs.mean(1)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()
                    test_num += float(labels.size(0))
                    test_acc1, test_acc5 = accuracy(mean_out, labels, topk=(1, 5))
                    test_correct += test_acc1.item()
                    test_correct5 += test_acc5.item()
                    reset_net(self.network)
            return test_loss / test_num, test_correct / test_num, test_correct5 / test_num


class AlignmentTLTrainer(TLTrainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(AlignmentTLTrainer, self).__init__(args, device, writer, network, optimizer, criterion, scheduler,
                                                 model_path)

    def train(self, train_loader, dvs_val_loader):

        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_num = 1
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            source_train_correct5 = 0
            target_train_loss = 0
            target_train_correct = 0
            target_train_correct5 = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            for i, ((source_data, target_data), labels) in enumerate(train_loader):
                self.optimizer.zero_grad()

                if source_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if target_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    target_data = self.encoder_dict[self.args.encoder_type](target_data)

                source_data, labels = source_data.to(self.device), labels.to(self.device)
                target_data, labels = target_data.to(self.device), labels.to(self.device)

                source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(source_data.float(),
                                                                                                target_data.float(),
                                                                                                self.args.encoder_tl_loss_type,
                                                                                                self.args.feature_tl_loss_type)
                source_mean_out = source_outputs.mean(1)  # (N, num_classes)
                source_clf_loss = self.criterion(source_outputs, labels)

                target_mean_out = target_outputs.mean(1)  # (N, num_classes)
                target_clf_loss = self.criterion(target_outputs, labels)

                loss = source_clf_loss + target_clf_loss

                if self.args.encoder_tl_lamb > 0.0:
                    loss = loss + self.args.encoder_tl_lamb * encoder_tl_loss
                if self.args.feature_tl_lamb > 0.0:
                    loss = loss + self.args.feature_tl_lamb * feature_tl_loss

                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_encoder_tl_loss += encoder_tl_loss.item()
                total_feature_tl_loss += feature_tl_loss.item()
                total_loss += loss.item()

                loss.mean().backward()
                self.optimizer.step()
                # self.scheduler.step()

                train_num += float(labels.size(0))

                # _, source_predicted = source_mean_out.cpu().max(1)
                # source_train_correct += float(source_predicted.eq(labels.cpu()).sum().item())
                #
                # _, target_predicted = target_mean_out.cpu().max(1)
                # target_train_correct += float(target_predicted.eq(labels.cpu()).sum().item())

                source_acc1, source_acc5 = accuracy(source_mean_out, labels, topk=(1, 5))
                target_acc1, target_acc5 = accuracy(target_mean_out, labels, topk=(1, 5))
                source_train_correct += source_acc1
                source_train_correct5 += source_acc5
                target_train_correct += target_acc1
                target_train_correct5 += target_acc5

                reset_net(self.network)

            self.scheduler.step()

            source_train_acc1 = source_train_correct / train_num
            target_train_acc1 = target_train_correct / train_num
            total_acc = (source_train_acc1 + target_train_acc1) / 2
            source_train_acc5 = source_train_correct5 / train_num
            target_train_acc5 = target_train_correct5 / train_num
            total_acc5 = (source_train_acc5 + target_train_acc5) / 2
            source_train_loss = source_train_loss / train_num
            target_train_loss = target_train_loss / train_num
            total_encoder_tl_loss = total_encoder_tl_loss / train_num
            total_feature_tl_loss = total_feature_tl_loss / train_num
            total_loss = total_loss / train_num
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  'source_clf_loss={:.5f} source_train_acc={:.4f} source_train_acc5={:.4f} '
                  'target_clf_loss={:.5f} target_train_acc={:.4f} target_train_acc5={:.4f} '
                  'total_loss={:.5f} train_acc={:.4f} train_acc5={:.4f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(epoch, self.args.epoch,
                                                                         (time.time() - start) / 60,
                                                                         source_train_loss, source_train_acc1,
                                                                         source_train_acc5,
                                                                         target_train_loss, target_train_acc1,
                                                                         target_train_acc5,
                                                                         total_loss, total_acc, total_acc5,
                                                                         total_encoder_tl_loss, total_feature_tl_loss))

            grad_data_name = self.network.dvs_input.fwd.module._modules['0']
            grad_data = self.network.dvs_input.fwd.module._modules['0'].weight.grad
            print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
            grad_data_name = self.network.rgb_input.fwd.module._modules['0']
            grad_data = self.network.rgb_input.fwd.module._modules['0'].weight.grad
            print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
            # for l in range(7):
            #     grad_data_name = self.network.features._modules[str(l)].module._modules['0']
            #     grad_data = self.network.features._modules[str(l)].module._modules['0'].weight.grad
            #     print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
            # grad_data_name = self.network.bottleneck.module._modules['0']
            # grad_data = self.network.bottleneck.module._modules['0'].weight.grad
            # print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())

            # 新增验证集评估部分
            val_loss, val_acc1, val_acc5 = self.test(dvs_val_loader)
            if type(val_loss) is list:
                all_val_loss = val_loss
                val_loss = sum(val_loss) / len(val_loss)
                all_val_acc1 = val_acc1
                val_acc1 = sum(val_acc1) / len(val_acc1)
                all_val_acc5 = val_acc5
                val_acc5 = sum(val_acc5) / len(val_acc5)

                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} val_acc1={val_acc1:.4f} ' \
                      f'all_val_acc1={all_val_acc1} val_acc5={val_acc5:.4f} all_val_acc5={all_val_acc5}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)
                for val_id in range(len(all_val_loss)):
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy1", scalar_value=all_val_acc1[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy5", scalar_value=all_val_acc5[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/loss", scalar_value=all_val_loss[val_id],
                                           global_step=epoch)
            else:
                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} '
                      f'val_acc1={val_acc1:.4f} val_acc5={val_acc5:.4f}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < target_train_acc1:
                self.best_train_acc = target_train_acc1

            if self.best_val_acc < val_acc1:
                self.best_val_acc = val_acc1
                self.save_model(epoch)

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
            # torch.cuda.empty_cache()
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

            info = psutil.virtual_memory()
            print(info)

            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")
        return self.best_train_acc, self.best_val_acc



class AlignmentTLTrainerOnImageNet(TLTrainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(AlignmentTLTrainerOnImageNet, self).__init__(args, device, writer, network, optimizer, criterion,
                                                           scheduler,
                                                           model_path)

    def train(self, train_loader, dvs_val_loader):
        if self.args.amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_num = 1
            total_loss = 0
            source_train_loss = 0
            source_train_correct1 = 0
            source_train_correct5 = 0
            target_train_loss = 0
            target_train_correct1 = 0
            target_train_correct5 = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            for i, ((source_data, target_data), labels) in enumerate(train_loader):
                if source_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if target_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    target_data = self.encoder_dict[self.args.encoder_type](target_data)
                source_data, labels = source_data.to(self.device), labels.to(self.device)
                target_data, labels = target_data.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    source_data = source_data.permute(1, 0, 2, 3, 4)  # (N, T, 3, H, W) -> (T, N, 3, H, W)
                    target_data = target_data.permute(1, 0, 2, 3, 4)  # (N, T, 3, H, W) -> (T, N, 3, H, W)

                    source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(source_data.float(),
                                                                                                    target_data.float(),
                                                                                                    self.args.encoder_tl_loss_type,
                                                                                                    self.args.feature_tl_loss_type)
                source_mean_out = source_outputs.mean(0)
                source_clf_loss = self.criterion(source_mean_out, labels)
                # source_clf_loss = self.criterion(source_outputs.permute(1, 0, 2), labels)

                target_mean_out = target_outputs.mean(0)
                target_clf_loss = self.criterion(target_mean_out, labels)
                # target_clf_loss = self.criterion(target_outputs.permute(1, 0, 2), labels)

                loss = target_clf_loss

                # loss = source_clf_loss + target_clf_loss
                #
                # if self.args.encoder_tl_lamb > 0.0:
                #     loss = loss + self.args.encoder_tl_lamb * encoder_tl_loss
                # if self.args.feature_tl_lamb > 0.0:
                #     loss = loss + self.args.feature_tl_lamb * feature_tl_loss

                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_encoder_tl_loss += encoder_tl_loss.item()
                total_feature_tl_loss += feature_tl_loss.item()
                total_loss += loss.item()

                self.optimizer.zero_grad()

                if scaler:  # 混合精度计算
                    scaler.scale(loss).backward()
                    if self.args.clip_grad_norm is not None:
                        # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                        scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.network.parameters(), self.args.clip_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.args.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.network.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()

                train_num += float(labels.size(0))

                # _, source_predicted = source_mean_out.cpu().max(1)
                # source_train_correct += float(source_predicted.eq(labels.cpu()).sum().item())
                #
                # _, target_predicted = target_mean_out.cpu().max(1)
                # target_train_correct += float(target_predicted.eq(labels.cpu()).sum().item())

                source_acc1, source_acc5 = accuracy(source_mean_out, labels, topk=(1, 5))
                target_acc1, target_acc5 = accuracy(target_mean_out, labels, topk=(1, 5))
                source_train_correct1 += source_acc1
                source_train_correct5 += source_acc5
                target_train_correct1 += target_acc1
                target_train_correct5 += target_acc5

                reset_net(self.network)
                if i >= len(train_loader) - 3:
                    print(torch.argmax(source_mean_out, axis=1), torch.argmax(target_mean_out, axis=1), labels)

            self.scheduler.step()

            source_train_acc1 = source_train_correct1 / train_num
            target_train_acc1 = target_train_correct1 / train_num
            total_acc = (source_train_acc1 + target_train_acc1) / 2
            source_train_acc5 = source_train_correct5 / train_num
            target_train_acc5 = target_train_correct5 / train_num
            total_acc5 = (source_train_acc5 + target_train_acc5) / 2
            source_train_loss = source_train_loss / train_num
            target_train_loss = target_train_loss / train_num
            total_encoder_tl_loss = total_encoder_tl_loss / train_num
            total_feature_tl_loss = total_feature_tl_loss / train_num
            total_loss = total_loss / train_num
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  'source_clf_loss={:.5f} source_train_acc={:.4f} source_train_acc5={:.4f} '
                  'target_clf_loss={:.5f} target_train_acc={:.4f} target_train_acc5={:.4f} '
                  'total_loss={:.5f} train_acc={:.4f} train_acc5={:.4f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(epoch, self.args.epoch,
                                                                         (time.time() - start) / 60,
                                                                         source_train_loss, source_train_acc1,
                                                                         source_train_acc5,
                                                                         target_train_loss, target_train_acc1,
                                                                         target_train_acc5,
                                                                         total_loss, total_acc, total_acc5,
                                                                         total_encoder_tl_loss, total_feature_tl_loss))
            #
            # grad_data_name = self.network.dvs_conv1.fwd.module._modules['0']
            # grad_data = self.network.dvs_conv1.fwd.module._modules['0'].weight.grad
            # print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
            # grad_data_name = self.network.rgb_conv1.fwd.module._modules['0']
            # grad_data = self.network.rgb_conv1.fwd.module._modules['0'].weight.grad
            # print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
            # for l in range(7):
            #     grad_data_name = self.network.features._modules[str(l)].fwd.module._modules['0']
            #     grad_data = self.network.features._modules[str(l)].fwd.module._modules['0'].weight.grad
            #     print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
            # grad_data_name = self.network.fc.module._modules['0']
            # grad_data = self.network.fc.module._modules['0'].weight.grad
            # print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())

            val_loss, val_acc1, val_acc5 = self.test(dvs_val_loader)
            if type(val_loss) is list:
                all_val_loss = val_loss
                val_loss = sum(val_loss) / len(val_loss)
                all_val_acc1 = val_acc1
                val_acc1 = sum(val_acc1) / len(val_acc1)
                all_val_acc5 = val_acc5
                val_acc5 = sum(val_acc5) / len(val_acc5)

                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} val_acc1={val_acc1:.4f} ' \
                      f'all_val_acc1={all_val_acc1} val_acc5={val_acc5:.4f} all_val_acc5={all_val_acc5}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)
                for val_id in range(len(all_val_loss)):
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy1", scalar_value=all_val_acc1[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy5", scalar_value=all_val_acc5[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/loss", scalar_value=all_val_loss[val_id],
                                           global_step=epoch)
            else:
                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} '
                      f'val_acc1={val_acc1:.4f} val_acc5={val_acc5:.4f}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < target_train_acc1:
                self.best_train_acc = target_train_acc1

            if self.best_val_acc < val_acc1:
                self.best_val_acc = val_acc1
                self.save_model(epoch)

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
            # torch.cuda.empty_cache()
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

            info = psutil.virtual_memory()
            print(info)

            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")
        return self.best_train_acc, self.best_val_acc

    def test(self, test_loader):
        self.network.eval()
        if type(test_loader) is list:
            test_loss_list = [0] * len(test_loader)
            test_num_list = [0] * len(test_loader)
            test_correct_list = [0] * len(test_loader)
            test_correct5_list = [0] * len(test_loader)

            with torch.no_grad():
                for i, loader in enumerate(test_loader):
                    for _, (data, labels) in enumerate(loader):
                        if data.shape[1] == 3:
                            data = self.encoder_dict[self.args.encoder_type](data)
                        data, labels = data.to(self.device), labels.to(self.device)
                        data = data.permute(1, 0, 2, 3, 4)  # (N, T, 3, H, W) -> (T, N, 3, H, W)
                        outputs = self.network(data, data)
                        mean_out = outputs.mean(0)
                        loss = self.criterion(mean_out, labels)
                        # loss = self.criterion(outputs.permute(1, 0, 2), labels)
                        test_loss_list[i] += loss.item()
                        test_num_list[i] += float(labels.size(0))
                        test_acc1, test_acc5 = accuracy(mean_out, labels, topk=(1, 5))
                        test_correct_list[i] += test_acc1
                        test_correct5_list[i] += test_acc5

                        reset_net(self.network)
                    test_loss_list[i] /= test_num_list[i]
                    test_correct_list[i] /= test_num_list[i]
                    test_correct5_list[i] /= test_num_list[i]

            return test_loss_list, test_correct_list, test_correct5_list[i]
        else:
            test_loss = 0
            test_num = 0
            test_correct = 0
            test_correct5 = 0

            with torch.no_grad():
                for i, (data, labels) in enumerate(test_loader):
                    if data.shape[1] == 3:
                        data = self.encoder_dict[self.args.encoder_type](data)
                    data, labels = data.to(self.device), labels.to(self.device)
                    data = data.permute(1, 0, 2, 3, 4)  # (N, T, 3, H, W) -> (T, N, 3, H, W)
                    outputs = self.network(data, data)
                    mean_out = outputs.mean(0)
                    loss = self.criterion(mean_out, labels)
                    # loss = self.criterion(outputs.permute(1, 0, 2), labels)
                    test_loss += loss.item()
                    test_num += float(labels.size(0))
                    test_acc1, test_acc5 = accuracy(mean_out, labels, topk=(1, 5))
                    test_correct += test_acc1
                    test_correct5 += test_acc5
                    reset_net(self.network)
                    if i == len(test_loader) - 1:
                        print(torch.argmax(mean_out, axis=1), labels)
            return test_loss / test_num, test_correct / test_num, test_correct5 / test_num


class ANN2SNNTrainer(Trainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(ANN2SNNTrainer, self).__init__(args, device, writer, network, optimizer, criterion, scheduler, model_path)

    def train_ann(self, train_loader, val_loader):
        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_loss = 0
            train_num = 0
            train_correct = 0

            for i, (data, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                # data = self.encoder_dict[self.args.encoder_type](data)
                # print(data.shape)
                data, labels = data.to(self.device), labels.to(self.device)

                outputs = self.network(data.float())
                loss = self.criterion(outputs, labels)

                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_num += float(labels.size(0))
                _, predicted = outputs.cpu().max(1)
                train_correct += float(predicted.eq(labels.cpu()).sum().item())
            train_acc = train_correct / train_num
            train_loss = train_loss / train_num
            print('Epoch:[{}/{}]\t time cost: {:.2f}min\t train_loss={:.5f}\t train_acc={:.3f}'.format(epoch,
                                                                                                       self.args.epoch,
                                                                                                       (
                                                                                                               time.time() - start) / 60,
                                                                                                       train_loss,
                                                                                                       train_acc))

            val_loss, val_acc = self.test_ann(val_loader)
            print('Epoch:[{}/{}]\t val_loss={:.5f}\t val_acc={:.3f}'.format(epoch, self.args.epoch,
                                                                            val_loss, val_acc))

            self.writer.add_scalar(tag="ann_train/accuracy", scalar_value=train_acc, global_step=epoch)
            self.writer.add_scalar(tag="ann_train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                   global_step=epoch)
            self.writer.add_scalar(tag="ann_train/loss", scalar_value=train_loss, global_step=epoch)
            self.writer.add_scalar(tag="ann_val/accuracy", scalar_value=val_acc, global_step=epoch)
            self.writer.add_scalar(tag="ann_val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < train_acc:
                self.best_train_acc = train_acc
                self.save_model(epoch)

            if self.best_val_acc < val_acc:
                self.best_val_acc = val_acc

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
        return self.best_train_acc

    def test_ann(self, test_loader):
        self.network.eval()
        test_loss = 0
        test_num = 0
        test_correct = 0

        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                # data = self.encoder_dict[self.args.encoder_type](data)
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.network(data)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                test_num += float(labels.size(0))
                _, predicted = outputs.cpu().max(1)
                test_correct += float(predicted.eq(labels.cpu()).sum().item())
        return test_loss / test_num, test_correct / test_num

    def train_snn(self, train_loader, val_loader):
        self.best_train_acc = 0
        self.best_val_acc = 0
        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_loss = 0
            train_num = 0
            train_correct = 0
            train_tot = torch.zeros(self.args.T).to(self.device)

            for i, (data, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                data, labels = data.to(self.device), labels.to(self.device)
                data = data.permute(1, 0, 2, 3, 4)  # [T, N, 2, H, W]

                # forward
                spikes = 0
                for t in range(self.args.T):
                    x = data[t]
                    out = self.network(x.float())
                    spikes = spikes + out
                    train_tot[t] += (labels == spikes.max(1)[1]).sum()

                # backward process
                loss = self.criterion(spikes, labels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                reset_net(self.network)  # QCFS 所用snn模型必须要reset

                train_num += float(labels.size(0))
                _, predicted = spikes.cpu().max(1)
                train_correct += float(predicted.eq(labels.cpu()).sum().item())
            train_acc = train_correct / train_num
            train_loss = train_loss / train_num
            train_tot = train_tot / train_num
            print('Epoch:[{}/{}] time cost: {:.2f}min train_loss={:.5f} train_acc={:.3f} train_tot={}'.format(epoch,
                                                                                                              self.args.epoch,
                                                                                                              (
                                                                                                                      time.time() - start) / 60,
                                                                                                              train_loss,
                                                                                                              train_acc,
                                                                                                              train_tot))

            val_loss, val_acc, val_tot = self.test_snn(val_loader)
            print('Epoch:[{}/{}] val_loss={:.5f} val_acc={:.3f} val_tot={}'.format(epoch, self.args.epoch,
                                                                                   val_loss, val_acc, val_tot))

            self.writer.add_scalar(tag="train/accuracy", scalar_value=train_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=train_loss, global_step=epoch)
            self.writer.add_scalar(tag="val/accuracy", scalar_value=val_acc, global_step=epoch)
            self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < train_acc:
                self.best_train_acc = train_acc
                self.save_model(epoch)

            if self.best_val_acc < val_acc:
                self.best_val_acc = val_acc

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
        return self.best_train_acc

    def test_snn(self, test_loader):
        self.network.eval()
        test_loss = 0
        test_num = 0
        test_correct = 0
        tot = torch.zeros(self.args.T).to(self.device)

        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                # data = self.encoder_dict[self.args.encoder_type](data)
                data, labels = data.to(self.device), labels.to(self.device)
                data = data.permute(1, 0, 2, 3, 4)  # [T, N, 2, H, W]

                # forward
                spikes = 0
                for t in range(self.args.T):
                    x = data[t]
                    out = self.network(x.float())
                    spikes = spikes + out
                    tot[t] += (labels == spikes.max(1)[1]).sum()

                loss = self.criterion(spikes, labels)
                reset_net(self.network)  # QCFS 所用snn模型必须要reset

                test_loss += loss.item()
                test_num += float(labels.size(0))
                _, predicted = spikes.cpu().max(1)
                test_correct += float(predicted.eq(labels.cpu()).sum().item())
            return test_loss / test_num, test_correct / test_num, tot / test_num


class AttentionTrainer(Trainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(AttentionTrainer, self).__init__(args, device, writer, network, optimizer, criterion, scheduler,
                                               model_path)

    def train(self, source_train_loader, target_train_loader, target_val_loader):
        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_num = 0
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            target_train_loss = 0
            target_train_correct = 0

            target_train_iter = itertools.cycle(iter(target_train_loader))
            for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(
                    zip(source_train_loader, target_train_iter)):
                self.optimizer.zero_grad()

                source_data = self.encoder_dict[self.args.encoder_type](source_data)

                source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)

                source_outputs, target_outputs = self.network(source_data.float(), target_data.float())
                source_mean_out = source_outputs.mean(1)
                source_clf_loss = self.criterion(source_outputs, source_labels)

                target_mean_out = target_outputs.mean(1)
                target_clf_loss = self.criterion(target_outputs, target_labels)

                loss = source_clf_loss + target_clf_loss

                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_loss += loss.item()

                loss.mean().backward()
                self.optimizer.step()
                self.scheduler.step()

                train_num += float(source_labels.size(0))

                _, source_predicted = source_mean_out.cpu().max(1)
                source_train_correct += float(source_predicted.eq(source_labels.cpu()).sum().item())

                _, target_predicted = target_mean_out.cpu().max(1)
                target_train_correct += float(target_predicted.eq(target_labels.cpu()).sum().item())

            source_train_acc = source_train_correct / train_num
            target_train_acc = target_train_correct / train_num
            total_acc = (source_train_acc + target_train_acc) / 2
            source_train_loss = source_train_loss / train_num
            target_train_loss = target_train_loss / train_num
            total_loss = total_loss / train_num
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  'source_clf_loss={:.5f} source_train_acc={:.3f} '
                  'target_clf_loss={:.5f} target_train_acc={:.3f} '
                  'total_loss={:.5f} train_acc={:.3f}'.format(epoch, self.args.epoch,
                                                              (time.time() - start) / 60,
                                                              source_train_loss, source_train_acc,
                                                              target_train_loss, target_train_acc,
                                                              total_loss, total_acc))

            val_loss, val_acc, val_acc_2 = self.test(target_val_loader)
            print('Epoch:[{}/{}] val_loss={:.5f} val_acc={:.3f}  val_acc_2={:.3f}'.format(epoch, self.args.epoch,
                                                                                          val_loss, val_acc, val_acc_2))

            self.writer.add_scalar(tag="train/source_accuracy", scalar_value=source_train_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/target_accuracy", scalar_value=target_train_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/accuracy", scalar_value=total_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)
            self.writer.add_scalar(tag="val/accuracy", scalar_value=val_acc, global_step=epoch)
            self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < target_train_acc:
                self.best_train_acc = target_train_acc

            if self.best_val_acc < val_acc:
                self.best_val_acc = val_acc
                self.save_model(epoch)

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
        return self.best_train_acc

    def test(self, test_loader):
        self.network.eval()
        test_loss = 0
        test_num = 0
        test_correct = 0
        test_correct_2 = 0

        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs, outputs_2 = self.network(data, data)
                mean_out = outputs.mean(1)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                test_num += float(labels.size(0))
                _, predicted = mean_out.cpu().max(1)
                test_correct += float(predicted.eq(labels.cpu()).sum().item())

                mean_out_2 = outputs_2.mean(1)
                _, predicted_2 = mean_out_2.cpu().max(1)
                test_correct_2 += float(predicted_2.eq(labels.cpu()).sum().item())
        return test_loss / test_num, test_correct / test_num, test_correct_2 / test_num


class AlignmentTLTrainer_Edge(TLTrainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(AlignmentTLTrainer_Edge, self).__init__(args, device, writer, network, optimizer, criterion, scheduler,
                                                      model_path)

    def train(self, train_loader, dvs_val_loader):

        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_num = 1
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            source_train_correct5 = 0
            target_train_loss = 0
            target_train_correct = 0
            target_train_correct5 = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            for i, ((source_data, target_data), labels) in enumerate(train_loader):
                self.optimizer.zero_grad()

                if source_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if target_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    target_data = self.encoder_dict[self.args.encoder_type](target_data)

                source_data, labels = source_data.to(self.device), labels.to(self.device)
                target_data, labels = target_data.to(self.device), labels.to(self.device)

                source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(source_data.float(),
                                                                                                target_data.float(),
                                                                                                self.args.encoder_tl_loss_type,
                                                                                                self.args.feature_tl_loss_type)
                source_mean_out = source_outputs.mean(1)  # (N, num_classes)
                source_clf_loss = self.criterion(source_outputs, labels)

                target_mean_out = target_outputs.mean(1)  # (N, num_classes)
                target_clf_loss = self.criterion(target_outputs, labels)

                loss = source_clf_loss + target_clf_loss

                if self.args.encoder_tl_lamb > 0.0:
                    loss = loss + self.args.encoder_tl_lamb * encoder_tl_loss
                if self.args.feature_tl_lamb > 0.0:
                    loss = loss + self.args.feature_tl_lamb * feature_tl_loss

                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_encoder_tl_loss += encoder_tl_loss.item()
                total_feature_tl_loss += feature_tl_loss.item()
                total_loss += loss.item()

                loss.mean().backward()
                self.optimizer.step()
                # self.scheduler.step()

                train_num += float(labels.size(0))

                # _, source_predicted = source_mean_out.cpu().max(1)
                # source_train_correct += float(source_predicted.eq(labels.cpu()).sum().item())
                #
                # _, target_predicted = target_mean_out.cpu().max(1)
                # target_train_correct += float(target_predicted.eq(labels.cpu()).sum().item())

                source_acc1, source_acc5 = accuracy(source_mean_out, labels, topk=(1, 5))
                target_acc1, target_acc5 = accuracy(target_mean_out, labels, topk=(1, 5))
                source_train_correct += source_acc1
                source_train_correct5 += source_acc5
                target_train_correct += target_acc1
                target_train_correct5 += target_acc5

                reset_net(self.network)

            self.scheduler.step()

            source_train_acc1 = source_train_correct / train_num
            target_train_acc1 = target_train_correct / train_num
            total_acc = (source_train_acc1 + target_train_acc1) / 2
            source_train_acc5 = source_train_correct5 / train_num
            target_train_acc5 = target_train_correct5 / train_num
            total_acc5 = (source_train_acc5 + target_train_acc5) / 2
            source_train_loss = source_train_loss / train_num
            target_train_loss = target_train_loss / train_num
            total_encoder_tl_loss = total_encoder_tl_loss / train_num
            total_feature_tl_loss = total_feature_tl_loss / train_num
            total_loss = total_loss / train_num
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  'source_clf_loss={:.5f} source_train_acc={:.4f} source_train_acc5={:.4f} '
                  'target_clf_loss={:.5f} target_train_acc={:.4f} target_train_acc5={:.4f} '
                  'total_loss={:.5f} train_acc={:.4f} train_acc5={:.4f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(epoch, self.args.epoch,
                                                                         (time.time() - start) / 60,
                                                                         source_train_loss, source_train_acc1,
                                                                         source_train_acc5,
                                                                         target_train_loss, target_train_acc1,
                                                                         target_train_acc5,
                                                                         total_loss, total_acc, total_acc5,
                                                                         total_encoder_tl_loss, total_feature_tl_loss))

            val_loss, val_acc1, val_acc5 = self.test(dvs_val_loader)
            if type(val_loss) is list:
                all_val_loss = val_loss
                val_loss = sum(val_loss) / len(val_loss)
                all_val_acc1 = val_acc1
                val_acc1 = sum(val_acc1) / len(val_acc1)
                all_val_acc5 = val_acc5
                val_acc5 = sum(val_acc5) / len(val_acc5)

                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} val_acc1={val_acc1:.4f} ' \
                      f'all_val_acc1={all_val_acc1} val_acc5={val_acc5:.4f} all_val_acc5={all_val_acc5}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)
                for val_id in range(len(all_val_loss)):
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy1", scalar_value=all_val_acc1[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy5", scalar_value=all_val_acc5[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/loss", scalar_value=all_val_loss[val_id],
                                           global_step=epoch)
            else:
                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} '
                      f'val_acc1={val_acc1:.4f} val_acc5={val_acc5:.4f}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < target_train_acc1:
                self.best_train_acc = target_train_acc1

            if self.best_val_acc < val_acc1:
                self.best_val_acc = val_acc1
                self.save_model(epoch)

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
            # torch.cuda.empty_cache()
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

            info = psutil.virtual_memory()
            print(info)

            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")
        return self.best_train_acc, self.best_val_acc


class AlignmentTLTrainer_Edge_1(TLTrainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(AlignmentTLTrainer_Edge_1, self).__init__(args, device, writer, network, optimizer, criterion,
                                                        scheduler,
                                                        model_path)

    def train(self, train_loader, dvs_val_loader, second_loader=None):  # 这里多了一个second_loader参数

        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_num = 1
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            target_train_loss = 0
            target_train_correct = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            for i, ((source_data, target_data), labels) in enumerate(train_loader):
                self.optimizer.zero_grad()

                # 先将 source_data = data，接着进行边缘提取
                source_data = source_data.to(self.device).float()  # 直接使用原始数据作为 source_data
                target_data = target_data.to(self.device).float()  # 目标数据，先赋值为原始数据

                # # 对目标数据进行边缘提取
                # source_data1 = self.network.edge_extractor1(source_data)
                # source_data2 = self.network.edge_extractor2(source_data)
                # 将边缘提取的图通道变为2，方便后续对DVS的迁移，DVS是两通道，这样可以不修改模型
                # source_data = torch.cat((source_data1, source_data2), dim=1)

                # 统一编码处理：源数据如果为3通道则添加时间维度；目标数据为2通道也需要处理
                if source_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, C, H, W)
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if source_data.shape[1] == 2:
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if target_data.shape[1] == 1:  # MNIST
                    target_data = self.encoder_dict[self.args.encoder_type](target_data)


                # 前向传播（同一数据作为源和目标）
                source_data, labels = source_data.to(self.device), labels.to(self.device)
                target_data, labels = target_data.to(self.device), labels.to(self.device)

                source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(source_data.float(),
                                                                                                target_data.float(),
                                                                                                self.args.encoder_tl_loss_type,
                                                                                                self.args.feature_tl_loss_type)
                source_mean_out = source_outputs.mean(1)  # (N, num_classes)
                source_clf_loss = self.criterion(source_outputs, labels)

                target_mean_out = target_outputs.mean(1)  # (N, num_classes)
                target_clf_loss = self.criterion(target_outputs, labels)

                loss = source_clf_loss + target_clf_loss

                if self.args.encoder_tl_lamb > 0.0:
                    loss = loss + self.args.encoder_tl_lamb * encoder_tl_loss
                if self.args.feature_tl_lamb > 0.0:
                    loss = loss + self.args.feature_tl_lamb * feature_tl_loss

                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_encoder_tl_loss += encoder_tl_loss.item()
                total_feature_tl_loss += feature_tl_loss.item()
                total_loss += loss.item()

                loss.mean().backward()
                self.optimizer.step()
                # self.scheduler.step()

                train_num += float(labels.size(0))

                source_acc1, = accuracy(source_mean_out, labels, topk=(1,))
                target_acc1, = accuracy(target_mean_out, labels, topk=(1,))
                source_train_correct += source_acc1
                target_train_correct += target_acc1

                reset_net(self.network)

            self.scheduler.step()

            source_train_acc1 = source_train_correct / train_num
            target_train_acc1 = target_train_correct / train_num
            total_acc = (source_train_acc1 + target_train_acc1) / 2
            source_train_loss = source_train_loss / train_num
            target_train_loss = target_train_loss / train_num
            total_encoder_tl_loss = total_encoder_tl_loss / train_num
            total_feature_tl_loss = total_feature_tl_loss / train_num
            total_loss = total_loss / train_num
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  'source_clf_loss={:.5f} source_train_acc={:.4f} '
                  'target_clf_loss={:.5f} target_train_acc={:.4f} '
                  'total_loss={:.5f} train_acc={:.4f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(epoch, self.args.epoch,
                                                                         (time.time() - start) / 60,
                                                                         source_train_loss, source_train_acc1,
                                                                         target_train_loss, target_train_acc1,
                                                                         total_loss, total_acc,
                                                                         total_encoder_tl_loss, total_feature_tl_loss))

            val_loss, val_acc1, _ = self.test(dvs_val_loader)
            if type(val_loss) is list:
                all_val_loss = val_loss
                val_loss = sum(val_loss) / len(val_loss)
                all_val_acc1 = val_acc1
                val_acc1 = sum(val_acc1) / len(val_acc1)

                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} val_acc1={val_acc1:.4f} '
                      f'all_val_acc1={all_val_acc1}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)
                for val_id in range(len(all_val_loss)):
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy1", scalar_value=all_val_acc1[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/loss", scalar_value=all_val_loss[val_id],
                                           global_step=epoch)
            else:
                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} '
                      f'val_acc1={val_acc1:.4f}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < target_train_acc1:
                self.best_train_acc = target_train_acc1

            if self.best_val_acc < val_acc1:
                self.best_val_acc = val_acc1
                self.save_model(epoch)

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")
            # torch.cuda.empty_cache()
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

            info = psutil.virtual_memory()
            print(info)

            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")
        return self.best_train_acc, self.best_val_acc


class AlignmentTLTrainerWithProgressBar(AlignmentTLTrainer):
    """
    带进度条的AlignmentTLTrainer，支持灰度转换
    
    主要特性:
    - 添加tqdm进度条显示训练进度
    - 支持--use_grayscale参数进行RGB到灰度的转换
    - 实时显示Loss、RGB_Acc、DVS_Acc等指标
    - 保持与标准AlignmentTLTrainer完全兼容
    """
    
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super(AlignmentTLTrainerWithProgressBar, self).__init__(args, device, writer, network, optimizer, criterion, scheduler, model_path)
        
        # 导入PIL用于灰度转换
        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            self.Image = None
            print("Warning: PIL not available, grayscale conversion will be disabled")
    
    def rgb_to_grayscale_3channel(self, rgb_tensor):
        """
        将RGB tensor转换为灰度但保持三通道格式
        
        Args:
            rgb_tensor: (N, 3, H, W) 的RGB tensor
            
        Returns:
            grayscale_tensor: (N, 3, H, W) 的灰度tensor，三个通道值相同
        """
        if rgb_tensor.shape[1] != 3:
            return rgb_tensor
            
        # 使用标准的RGB到灰度转换权重: 0.299*R + 0.587*G + 0.114*B
        grayscale = 0.299 * rgb_tensor[:, 0:1, :, :] + \
                   0.587 * rgb_tensor[:, 1:2, :, :] + \
                   0.114 * rgb_tensor[:, 2:3, :, :]
        
        # 复制到三个通道
        grayscale_3channel = torch.cat([grayscale, grayscale, grayscale], dim=1)
        
        return grayscale_3channel

    def train(self, train_loader, dvs_val_loader):
        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_num = 1
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            source_train_correct5 = 0
            target_train_loss = 0
            target_train_correct = 0
            target_train_correct5 = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            # 添加进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f'Epoch [{epoch+1}/{self.args.epoch}]', 
                       leave=True)
            
            for i, ((source_data, target_data), labels) in pbar:
                self.optimizer.zero_grad()

                # 应用灰度转换（如果启用）
                if hasattr(self.args, 'use_grayscale') and self.args.use_grayscale and source_data.shape[1] == 3:
                    source_data = self.rgb_to_grayscale_3channel(source_data)

                # 编码处理
                if source_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if target_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, 3, H, W)
                    target_data = self.encoder_dict[self.args.encoder_type](target_data)

                source_data, labels = source_data.to(self.device), labels.to(self.device)
                target_data, labels = target_data.to(self.device), labels.to(self.device)

                source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(source_data.float(),
                                                                                                target_data.float(),
                                                                                                self.args.encoder_tl_loss_type,
                                                                                                self.args.feature_tl_loss_type)
                source_mean_out = source_outputs.mean(1)  # (N, num_classes)
                source_clf_loss = self.criterion(source_outputs, labels)

                target_mean_out = target_outputs.mean(1)  # (N, num_classes)
                target_clf_loss = self.criterion(target_outputs, labels)

                loss = source_clf_loss + target_clf_loss

                if self.args.encoder_tl_lamb > 0.0:
                    loss = loss + self.args.encoder_tl_lamb * encoder_tl_loss
                if self.args.feature_tl_lamb > 0.0:
                    loss = loss + self.args.feature_tl_lamb * feature_tl_loss

                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_encoder_tl_loss += encoder_tl_loss.item()
                total_feature_tl_loss += feature_tl_loss.item()
                total_loss += loss.item()

                loss.mean().backward()
                self.optimizer.step()

                train_num += float(labels.size(0))

                source_acc1, source_acc5 = accuracy(source_mean_out, labels, topk=(1, 5))
                target_acc1, target_acc5 = accuracy(target_mean_out, labels, topk=(1, 5))
                source_train_correct += source_acc1
                source_train_correct5 += source_acc5
                target_train_correct += target_acc1
                target_train_correct5 += target_acc5

                # 更新进度条信息
                current_avg_loss = total_loss / train_num
                current_rgb_acc = source_train_correct / train_num
                current_dvs_acc = target_train_correct / train_num
                
                # 根据是否使用灰度转换调整显示标签
                source_label = "Gray_Acc" if (hasattr(self.args, 'use_grayscale') and self.args.use_grayscale) else "RGB_Acc"
                
                pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    source_label: f'{current_rgb_acc:.3f}',
                    'DVS_Acc': f'{current_dvs_acc:.3f}'
                })

                reset_net(self.network)
            
            # 关闭进度条
            pbar.close()

            self.scheduler.step()

            # 计算平均值
            source_train_acc1 = source_train_correct / train_num
            target_train_acc1 = target_train_correct / train_num
            total_acc = (source_train_acc1 + target_train_acc1) / 2
            source_train_acc5 = source_train_correct5 / train_num
            target_train_acc5 = target_train_correct5 / train_num
            total_acc5 = (source_train_acc5 + target_train_acc5) / 2
            source_train_loss = source_train_loss / train_num
            target_train_loss = target_train_loss / train_num
            total_encoder_tl_loss = total_encoder_tl_loss / train_num
            total_feature_tl_loss = total_feature_tl_loss / train_num
            total_loss = total_loss / train_num
            
            # 根据是否使用灰度转换调整打印信息
            source_type = "grayscale" if (hasattr(self.args, 'use_grayscale') and self.args.use_grayscale) else "source"
            
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  '{}_clf_loss={:.5f} {}_train_acc={:.4f} {}_train_acc5={:.4f} '
                  'target_clf_loss={:.5f} target_train_acc={:.4f} target_train_acc5={:.4f} '
                  'total_loss={:.5f} train_acc={:.4f} train_acc5={:.4f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(epoch, self.args.epoch,
                                                                         (time.time() - start) / 60,
                                                                         source_type, source_train_loss, 
                                                                         source_type, source_train_acc1,
                                                                         source_type, source_train_acc5,
                                                                         target_train_loss, target_train_acc1,
                                                                         target_train_acc5,
                                                                         total_loss, total_acc, total_acc5,
                                                                         total_encoder_tl_loss, total_feature_tl_loss))

            # 梯度监控（如果网络有相应属性）
            try:
                grad_data_name = self.network.dvs_input.fwd.module._modules['0']
                grad_data = self.network.dvs_input.fwd.module._modules['0'].weight.grad
                print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
                grad_data_name = self.network.rgb_input.fwd.module._modules['0']
                grad_data = self.network.rgb_input.fwd.module._modules['0'].weight.grad
                print(grad_data_name, grad_data.max(), grad_data.min(), grad_data.mean(), grad_data.std())
            except:
                pass  # 如果网络结构不同，跳过梯度监控

            # 验证集评估
            val_loss, val_acc1, val_acc5 = self.test(dvs_val_loader)
            if type(val_loss) is list:
                all_val_loss = val_loss
                val_loss = sum(val_loss) / len(val_loss)
                all_val_acc1 = val_acc1
                val_acc1 = sum(val_acc1) / len(val_acc1)
                all_val_acc5 = val_acc5
                val_acc5 = sum(val_acc5) / len(val_acc5)

                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} val_acc1={val_acc1:.4f} ' \
                      f'all_val_acc1={all_val_acc1} val_acc5={val_acc5:.4f} all_val_acc5={all_val_acc5}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)
                for val_id in range(len(all_val_loss)):
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy1", scalar_value=all_val_acc1[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/accuracy5", scalar_value=all_val_acc5[val_id],
                                           global_step=epoch)
                    self.writer.add_scalar(tag=f"val{val_id + 1}/loss", scalar_value=all_val_loss[val_id],
                                           global_step=epoch)
            else:
                print(f'Epoch:[{epoch}/{self.args.epoch}] val_loss={val_loss:.5f} '
                      f'val_acc1={val_acc1:.4f} val_acc5={val_acc5:.4f}')

                self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
                self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
                self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
                self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
                self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss,
                                       global_step=epoch)
                self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy1", scalar_value=val_acc1, global_step=epoch)
                self.writer.add_scalar(tag="val/accuracy5", scalar_value=val_acc5, global_step=epoch)
                self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)

            if self.best_train_acc < target_train_acc1:
                self.best_train_acc = target_train_acc1

            if self.best_val_acc < val_acc1:
                self.best_val_acc = val_acc1
                self.save_model(epoch)

            print(f"Best train acc is {self.best_train_acc}, best val acc is: {self.best_val_acc}.")

            # 内存监控
            info = psutil.virtual_memory()
            print(info)
            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")

        return self.best_train_acc, self.best_val_acc