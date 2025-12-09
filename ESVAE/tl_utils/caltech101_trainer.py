# -*- coding: utf-8 -*-
"""
Caltech101专用训练器 - 支持DVS损失权重调整
用于解决RGB-DVS迁移学习中的梯度流不平衡问题
"""
import time
import torch
from tqdm import tqdm
from tl_utils.common_utils import accuracy

try:
    from spikingjelly.activation_based.functional import reset_net
except:
    from utils.common_utils import reset_net


class AlignmentTLTrainerWithDVSWeight:
    """
    带DVS损失权重的迁移学习训练器
    
    主要改进:
    - 支持 --dvs_loss_weight 参数调整DVS分支的损失权重
    - 解决共享层梯度被RGB分支主导的问题
    - 保持与原始训练器的完全兼容性
    """
    
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

        # 编码器字典
        from tl_utils.common_utils import LapPoissonEncoder, MyPoissonEncoder, TimeEncoder
        self.encoder_dict = {
            'poison_encoder': MyPoissonEncoder(self.args.T, self.device),
            'lap_encoder': LapPoissonEncoder(self.args.T, self.device),
            'time_encoder': TimeEncoder(self.args.T, self.device)
        }
    
    def rgb_to_grayscale_3channel(self, rgb_tensor):
        """将RGB tensor转换为灰度但保持三通道格式"""
        if rgb_tensor.shape[1] != 3:
            return rgb_tensor
            
        # 使用标准的RGB到灰度转换权重
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
            # 使用 0 而不是 1，避免精度被轻微低估
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

                # 标准损失：不使用DVS损失权重
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
                
                source_label = "Gray_Acc" if (hasattr(self.args, 'use_grayscale') and self.args.use_grayscale) else "RGB_Acc"
                
                pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    source_label: f'{current_rgb_acc:.3f}',
                    'DVS_Acc': f'{current_dvs_acc:.3f}'
                })

                reset_net(self.network)
            
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
                                                                         target_train_loss, target_train_acc1, target_train_acc5,
                                                                         total_loss, total_acc, total_acc5,
                                                                         total_encoder_tl_loss, total_feature_tl_loss))

            # 验证
            val_loss, val_acc1, val_acc5 = self.test(dvs_val_loader)
            print('Epoch:[{}/{}] val_loss={:.5f} val_acc1={:.4f} val_acc5={:.4f}'.format(epoch, self.args.epoch,
                                                                                         val_loss, val_acc1, val_acc5))

            # 保存最佳模型：按验证集准确率来判断，而不是训练集
            # 这样可以避免出现 “val_acc 变好但不保存 / val_acc 变差却覆盖最佳模型” 的情况
            if val_acc1 > self.best_val_acc:
                self.best_val_acc = val_acc1
                # 同时记录下这一轮对应的目标域训练准确率，方便日志查看
                self.best_train_acc = target_train_acc1
                print('Saving..')
                torch.save(self.network.state_dict(), self.model_path)

            print(f'Best train acc is {self.best_train_acc:.6f}, best val acc is: {self.best_val_acc:.3f}.')

            # TensorBoard记录
            self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/source_acc", scalar_value=source_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/target_loss", scalar_value=target_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/target_acc", scalar_value=target_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/total_loss", scalar_value=total_loss, global_step=epoch)
            self.writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)
            self.writer.add_scalar(tag="val/acc", scalar_value=val_acc1, global_step=epoch)

        return self.best_train_acc, self.best_val_acc

    def test(self, test_loader):
        self.network.eval()
        test_loss = 0
        test_num = 0
        test_correct = 0
        test_correct5 = 0

        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.network(data.float(), data.float())
                mean_out = outputs.mean(1)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                test_num += float(labels.size(0))
                test_acc1, test_acc5 = accuracy(mean_out, labels, topk=(1, 5))
                test_correct += test_acc1
                test_correct5 += test_acc5
                reset_net(self.network)

        return test_loss / test_num, test_correct / test_num, test_correct5 / test_num

