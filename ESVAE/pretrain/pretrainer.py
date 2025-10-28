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

    def train(self, train_loader, second_loader=None):  # , val_loader
        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            train_loss = 0
            train_num = 0
            train_correct = 0

            for i, (data, labels) in enumerate(train_loader):
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

    def train(self, train_loader):  # , target_val_loader

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


class AlignmentTLTrainer_Edge_1(TLTrainer):
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super().__init__(args, device, writer, network, optimizer, criterion, scheduler, model_path)
        self.best_total_loss = float('inf')
        self.best_model_path = os.path.join(model_path, "best_model.pth")

    def save_model_best(self, epoch):
        """保存当前最佳模型"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_total_loss,
        }, self.best_model_path)
        print(f"Best model saved at epoch {epoch} with loss {self.best_total_loss:.5f}")

    def train(self, train_loader):  # 修改为单数据加载接口
        # 初始化一些统计变量
        for epoch in range(self.args.epochs):
            self.network.train()
            start = time.time()
            train_num = 0
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            source_train_correct5 = 0
            target_train_loss = 0
            target_train_correct = 0
            target_train_correct5 = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            # 创建进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f'Epoch [{epoch+1}/{self.args.epochs}]',
                       ncols=120)

            for i, (data, labels) in pbar:  # 单次加载
                self.optimizer.zero_grad()

                # 先将 source_data = data，接着进行边缘提取
                source_data = data.to(self.device).float()  # 直接使用原始数据作为 source_data
                target_data = data.to(self.device).float()  # 目标数据，先赋值为原始数据

                # 对目标数据进行边缘提取
                target_data1 = self.network.edge_extractor1(target_data)
                target_data2 = self.network.edge_extractor2(target_data)
                # 将边缘提取的图通道变为2，方便后续对DVS的迁移，DVS是两通道，这样可以不修改模型
                target_data = torch.cat((target_data1, target_data2), dim=1)

                # 统一编码处理：源数据如果为3通道则添加时间维度；目标数据为2通道也需要处理
                if source_data.shape[1] == 3:  # (N, 3, H, W) -> (N, T, C, H, W)
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                if target_data.shape[1] == 2:
                    target_data = self.encoder_dict[self.args.encoder_type](target_data)
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
                # self.scheduler.step()  # 如果使用 scheduler，可在 epoch 结束后调用

                train_num += float(labels.size(0))

                # 使用 accuracy 计算 top1 和 top5（top5 在两处统计）
                source_acc1, source_acc5 = accuracy(source_mean_out, labels, topk=(1, 5))
                target_acc1, target_acc5 = accuracy(target_mean_out, labels, topk=(1, 5))
                source_train_correct += source_acc1
                source_train_correct5 += source_acc5
                target_train_correct += target_acc1
                target_train_correct5 += target_acc5

                # 更新进度条信息
                current_avg_loss = total_loss / train_num
                current_rgb_acc = source_train_correct / train_num
                current_edge_acc = target_train_correct / train_num
                pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    'RGB_Acc': f'{current_rgb_acc:.3f}',
                    'Edge_Acc': f'{current_edge_acc:.3f}'
                })

                reset_net(self.network)

            # 关闭进度条
            pbar.close()

            # 更新学习率每个 epoch 完成后
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
                  'source_clf_loss={:.5f} source_train_acc1={:.4f} source_train_acc5={:.4f} '
                  'target_clf_loss={:.5f} target_train_acc1={:.4f} target_train_acc5={:.4f} '
                  'total_loss={:.5f} total_acc1={:.4f} total_acc5={:.4f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(
                epoch+1, self.args.epochs, (time.time() - start) / 60,
                source_train_loss, source_train_acc1, source_train_acc5,
                target_train_loss, target_train_acc1, target_train_acc5,
                total_loss, total_acc, total_acc5,
                total_encoder_tl_loss, total_feature_tl_loss))

            # 可以添加基于训练指标更新 best_total_loss 或 best_train_acc 的策略，此处简单示例根据总损失判断
            if total_loss < self.best_total_loss:
                self.best_total_loss = total_loss
                self.best_train_acc = total_acc  # 最佳准确率
                self.save_model_best(epoch+1)

            # 记录训练相关日志（验证相关日志已移除）
            self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
            self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
            self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)

            # 打印内存信息
            info = psutil.virtual_memory()
            print(info)
            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")

        return self.best_train_acc, self.best_total_loss


class AlignmentTLTrainer_RGB2DVS(TLTrainer):
    """
    RGB到DVS的迁移学习训练器
    从RGB图像中学习知识，并将其迁移到DVS数据上
    """
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super().__init__(args, device, writer, network, optimizer, criterion, scheduler, model_path)
        self.best_total_loss = float('inf')
        self.best_model_path = os.path.join(model_path, "best_model.pth")

    def save_model_best(self, epoch):
        """保存当前最佳模型"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_total_loss,
        }, self.best_model_path)
        print(f"Best model saved at epoch {epoch} with loss {self.best_total_loss:.5f}")

    def train(self, train_loader):
        """
        训练函数：处理同时包含RGB和DVS数据的loader
        train_loader返回: ((rgb_img, dvs_img), label)
        """
        for epoch in range(self.args.epochs):
            self.network.train()
            start = time.time()
            train_num = 0
            total_loss = 0
            source_train_loss = 0
            source_train_correct = 0
            source_train_correct5 = 0
            target_train_loss = 0
            target_train_correct = 0
            target_train_correct5 = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0

            # 创建进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f'Epoch [{epoch+1}/{self.args.epochs}]',
                       ncols=120)

            for i, (data, labels) in pbar:
                self.optimizer.zero_grad()

                # 解包数据: data是(rgb_img, dvs_img)的元组
                source_data, target_data = data  # source是RGB, target是DVS

                # 编码处理
                # RGB数据: (N, 3, H, W) -> (N, T, 3, H, W)
                if source_data.shape[1] == 3:
                    source_data = self.encoder_dict[self.args.encoder_type](source_data)
                
                # DVS数据已经是(N, T, 2, H, W)格式，无需编码
                # 但如果是其他格式需要处理
                if len(target_data.shape) == 4:  # (N, 2, H, W)
                    # 需要添加时间维度
                    target_data = target_data.unsqueeze(1).repeat(1, self.args.T, 1, 1, 1)

                # 数据转移到设备
                source_data, labels = source_data.to(self.device), labels.to(self.device)
                target_data = target_data.to(self.device)

                # 前向传播
                source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(
                    source_data.float(),
                    target_data.float(),
                    self.args.encoder_tl_loss_type,
                    self.args.feature_tl_loss_type
                )

                # 计算分类损失
                source_mean_out = source_outputs.mean(1)  # (N, num_classes)
                source_clf_loss = self.criterion(source_outputs, labels)

                target_mean_out = target_outputs.mean(1)  # (N, num_classes)
                target_clf_loss = self.criterion(target_outputs, labels)

                # 总损失
                loss = source_clf_loss + target_clf_loss

                if self.args.encoder_tl_lamb > 0.0:
                    loss = loss + self.args.encoder_tl_lamb * encoder_tl_loss
                if self.args.feature_tl_lamb > 0.0:
                    loss = loss + self.args.feature_tl_lamb * feature_tl_loss

                # 累积损失
                source_train_loss += source_clf_loss.item()
                target_train_loss += target_clf_loss.item()
                total_encoder_tl_loss += encoder_tl_loss.item()
                total_feature_tl_loss += feature_tl_loss.item()
                total_loss += loss.item()

                # 反向传播
                loss.mean().backward()
                self.optimizer.step()

                train_num += float(labels.size(0))

                # 计算准确率 (top1 和 top5)
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
                pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    'RGB_Acc': f'{current_rgb_acc:.3f}',
                    'DVS_Acc': f'{current_dvs_acc:.3f}'
                })

                # 重置网络状态
                reset_net(self.network)

            # 关闭进度条
            pbar.close()

            # 更新学习率
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

            # 打印训练信息
            print('Epoch:[{}/{}] time cost: {:.2f}min '
                  'source_clf_loss={:.5f} source_train_acc1={:.4f} source_train_acc5={:.4f} '
                  'target_clf_loss={:.5f} target_train_acc1={:.4f} target_train_acc5={:.4f} '
                  'total_loss={:.5f} total_acc1={:.4f} total_acc5={:.4f} '
                  'encoder_tl_loss={:.5f} feature_tl_loss={:.5f}'.format(
                epoch+1, self.args.epochs, (time.time() - start) / 60,
                source_train_loss, source_train_acc1, source_train_acc5,
                target_train_loss, target_train_acc1, target_train_acc5,
                total_loss, total_acc, total_acc5,
                total_encoder_tl_loss, total_feature_tl_loss))

            # 保存最佳模型
            if total_loss < self.best_total_loss:
                self.best_total_loss = total_loss
                self.best_train_acc = total_acc
                self.save_model_best(epoch+1)

            # 记录训练日志
            self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/source_accuracy5", scalar_value=source_train_acc5, global_step=epoch)
            self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/target_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/target_accuracy5", scalar_value=target_train_acc5, global_step=epoch)
            self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/accuracy5", scalar_value=total_acc5, global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)

            # 打印内存信息
            info = psutil.virtual_memory()
            print(info)
            print(u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024))
            print(u'available：%.4f GB' % (info.available / 1024 / 1024 / 1024))
            print(u'used：%.4f GB' % (info.used / 1024 / 1024 / 1024))
            print(u'free：%.4f GB' % (info.free / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)
            print("------------------------------------------------------")

        return self.best_train_acc, self.best_total_loss