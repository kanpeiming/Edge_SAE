# -*- coding: utf-8 -*-
"""
Edge to DVS Transfer Learning Trainer
边缘图到DVS的迁移学习训练器
"""

import os
import time
import torch
from tqdm import tqdm
from tl_utils.common_utils import LapPoissonEncoder, MyPoissonEncoder, TimeEncoder, accuracy

try:
    from spikingjelly.activation_based.functional import reset_net
except:
    from utils.common_utils import reset_net

from pretrain.pretrainer import TLTrainer


class AlignmentTLTrainer_Edge2DVS(TLTrainer):
    """
    Edge/RGB到DVS迁移学习训练器
    支持：
    - Edge数据（2通道）作为源域
    - RGB数据（3通道）作为源域
    DVS数据（2通道）作为目标域
    """
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super().__init__(args, device, writer, network, optimizer, criterion, scheduler, model_path)
        self.best_total_loss = float('inf')
        
        if model_path.endswith('.pth'):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path
        
        self.best_model_path = os.path.join(model_dir, "best_model.pth")

    def save_model_best(self, epoch):
        """保存当前最佳模型"""
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_total_loss,
        }, self.best_model_path)

    def train(self, train_loader):
        """Edge到DVS迁移学习训练"""
        for epoch in range(self.args.epoch):
            self.network.train()
            start = time.time()
            
            source_train_loss = 0
            target_train_loss = 0
            total_encoder_tl_loss = 0
            total_feature_tl_loss = 0
            total_loss = 0
            train_num = 0
            
            source_train_correct = 0
            source_train_correct5 = 0
            target_train_correct = 0
            target_train_correct5 = 0
            
            pbar = tqdm(enumerate(train_loader), 
                       total=len(train_loader),
                       desc=f'Epoch {epoch+1}/{self.args.epoch}',
                       ncols=120)
            
            for i, (data, labels) in pbar:
                self.optimizer.zero_grad()
                
                # 解包数据: data是(source_img, dvs_img)的元组
                source_data, target_data = data  # source是Edge/RGB, target是DVS
                source_label, dvs_label = labels
                labels = source_label.to(self.device)  # 使用source标签
                
                # 源数据编码: (N, C, H, W) -> (N, T, C, H, W)
                # 自动检测通道数：2通道为Edge，3通道为RGB
                if len(source_data.shape) == 4:
                    source_channels = source_data.shape[1]
                    source_data = self.encoder_dict[self.args.encoder_type](source_data, out_channel=source_channels)
                
                # DVS数据处理
                if len(target_data.shape) == 4:  # (N, 2, H, W)
                    target_data = target_data.unsqueeze(1).repeat(1, self.args.T, 1, 1, 1)
                elif len(target_data.shape) == 5:  # (N, T, 2, H, W)
                    pass
                else:
                    raise ValueError(f"Unexpected DVS data shape: {target_data.shape}")
                
                # 数据转移到设备
                source_data = source_data.to(self.device)
                target_data = target_data.to(self.device)
                
                # 前向传播
                source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(
                    source_data.float(),
                    target_data.float(),
                    self.args.encoder_tl_loss_type,
                    self.args.feature_tl_loss_type
                )
                
                # 计算分类损失
                source_mean_out = source_outputs.mean(1)
                source_clf_loss = self.criterion(source_outputs, labels)
                
                target_mean_out = target_outputs.mean(1)
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
                
                # 计算准确率
                source_acc1, source_acc5 = accuracy(source_mean_out, labels, topk=(1, 5))
                target_acc1, target_acc5 = accuracy(target_mean_out, labels, topk=(1, 5))
                source_train_correct += source_acc1
                source_train_correct5 += source_acc5
                target_train_correct += target_acc1
                target_train_correct5 += target_acc5
                
                # 更新进度条
                current_avg_loss = total_loss / train_num
                current_source_acc = source_train_correct / train_num
                current_dvs_acc = target_train_correct / train_num
                pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    'Source_Acc': f'{current_source_acc:.3f}',
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
            
            # 打印训练信息
            print('Epoch:[{}/{}] time: {:.2f}min '
                  'source_loss={:.5f} source_acc1={:.4f} '
                  'dvs_loss={:.5f} dvs_acc1={:.4f} '
                  'total_loss={:.5f} total_acc1={:.4f}'.format(
                epoch+1, self.args.epoch, (time.time() - start) / 60,
                source_train_loss, source_train_acc1,
                target_train_loss, target_train_acc1,
                total_loss, total_acc))
            
            # 保存最佳模型
            if total_loss < self.best_total_loss:
                self.best_total_loss = total_loss
                self.best_train_acc = total_acc
                self.save_model_best(epoch+1)
            
            # 记录训练日志
            self.writer.add_scalar(tag="train/source_accuracy1", scalar_value=source_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/source_loss", scalar_value=source_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/dvs_accuracy1", scalar_value=target_train_acc1, global_step=epoch)
            self.writer.add_scalar(tag="train/dvs_loss", scalar_value=target_train_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/accuracy1", scalar_value=total_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/encoder_tl_loss", scalar_value=total_encoder_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/feature_tl_loss", scalar_value=total_feature_tl_loss, global_step=epoch)
            self.writer.add_scalar(tag="train/lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)
        
        return self.best_train_acc, self.best_total_loss
    
    def test(self, test_loader):
        """Edge2DVS测试方法（使用DVS数据）"""
        self.network.eval()
        test_loss = 0
        test_num = 0
        test_correct = 0
        test_correct5 = 0
        
        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                dvs_data = data
                labels = labels.to(self.device)
                
                # DVS数据维度处理
                if len(dvs_data.shape) == 4:  # (N, 2, H, W)
                    dvs_data = dvs_data.unsqueeze(1).repeat(1, self.args.T, 1, 1, 1)
                
                dvs_data = dvs_data.to(self.device)
                
                # 测试时只使用DVS分支
                outputs = self.network(dvs_data.float(), dvs_data.float())
                
                if isinstance(outputs, tuple):
                    _, target_outputs, _, _ = outputs
                    outputs = target_outputs
                
                mean_out = outputs.mean(1)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                test_num += float(labels.size(0))
                
                test_acc1, test_acc5 = accuracy(mean_out, labels, topk=(1, 5))
                test_correct += test_acc1.item()
                test_correct5 += test_acc5.item()
                
                reset_net(self.network)
        
        return test_loss / test_num, test_correct / test_num, test_correct5 / test_num

