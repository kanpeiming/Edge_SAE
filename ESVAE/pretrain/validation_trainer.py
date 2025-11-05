# -*- coding: utf-8 -*-
"""
验证训练器扩展
为AlignmentTLTrainer_RGB2DVS添加验证集支持
"""

import torch
from tqdm import tqdm
from .pretrainer import AlignmentTLTrainer_RGB2DVS, reset_net


class AlignmentTLTrainer_RGB2DVS_WithValidation(AlignmentTLTrainer_RGB2DVS):
    """
    带验证集的RGB到DVS迁移学习训练器
    """
    
    def train_with_validation(self, train_loader, val_loader):
        """
        带验证集的训练方法（已移除早停机制，与tl.py保持一致）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        
        Returns:
            best_train_acc: 最佳训练准确率
            best_train_loss: 最佳训练损失
        """
        best_val_acc = 0.0
        
        for epoch in range(self.args.epochs):
            # 训练阶段
            train_acc, train_loss = self._train_one_epoch(train_loader, epoch)
            
            # 验证阶段
            val_acc, val_loss = self._validate_one_epoch(val_loader, epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录到tensorboard
            self.writer.add_scalar('train/accuracy', train_acc, epoch)
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('val/accuracy', val_acc, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            
            # 保存最佳模型（移除早停机制）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_train_acc = train_acc
                self.best_total_loss = train_loss
                self.save_model_best(epoch)
                print(f"✓ 新的最佳验证准确率: {best_val_acc:.4f}")
            
            print(f'Epoch [{epoch+1}/{self.args.epochs}] '
                  f'Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | '
                  f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f} | '
                  f'Best Val Acc: {best_val_acc:.4f}')
        
        return self.best_train_acc, self.best_total_loss
    
    def _train_one_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.network.train()
        total_loss = 0
        source_train_loss = 0
        target_train_loss = 0
        source_train_correct = 0
        target_train_correct = 0
        train_num = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f'Train Epoch [{epoch+1}/{self.args.epochs}]',
                   ncols=120)
        
        for i, (data, labels) in pbar:
            self.optimizer.zero_grad()
            
            # 解包数据
            source_data, target_data = data
            
            # 编码处理
            if source_data.shape[1] == 3:
                source_data = self.encoder_dict[self.args.encoder_type](source_data)
            if len(target_data.shape) == 4:
                target_data = target_data.unsqueeze(1).repeat(1, self.args.T, 1, 1, 1)
            
            # 数据转移到设备
            source_data, labels = source_data.to(self.device), labels.to(self.device)
            target_data = target_data.to(self.device)
            
            # 前向传播
            source_outputs, target_outputs, encoder_tl_loss, feature_tl_loss = self.network(
                source_data.float(), target_data.float(),
                self.args.encoder_tl_loss_type, self.args.feature_tl_loss_type
            )
            
            # 计算损失
            source_mean_out = source_outputs.mean(1)
            source_clf_loss = self.criterion(source_outputs, labels)
            target_mean_out = target_outputs.mean(1)
            target_clf_loss = self.criterion(target_outputs, labels)
            
            loss = source_clf_loss + target_clf_loss
            if self.args.encoder_tl_lamb > 0.0:
                loss = loss + self.args.encoder_tl_lamb * encoder_tl_loss
            if self.args.feature_tl_lamb > 0.0:
                loss = loss + self.args.feature_tl_lamb * feature_tl_loss
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            source_train_loss += source_clf_loss.item()
            target_train_loss += target_clf_loss.item()
            train_num += labels.size(0)
            
            _, source_predicted = source_mean_out.max(1)
            _, target_predicted = target_mean_out.max(1)
            source_train_correct += source_predicted.eq(labels).sum().item()
            target_train_correct += target_predicted.eq(labels).sum().item()
            
            # 重置网络状态
            reset_net(self.network)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'S_Acc': f'{source_train_correct/train_num:.3f}',
                'T_Acc': f'{target_train_correct/train_num:.3f}'
            })
        
        train_acc = (source_train_correct + target_train_correct) / (2 * train_num)
        train_loss = total_loss / len(train_loader)
        
        return train_acc, train_loss
    
    def _validate_one_epoch(self, val_loader, epoch):
        """验证一个epoch - 只使用DVS数据验证DVS分类性能"""
        self.network.eval()
        total_loss = 0
        dvs_val_correct = 0
        val_num = 0
        
        with torch.no_grad():
            pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                       desc=f'Val Epoch [{epoch+1}/{self.args.epochs}]',
                       ncols=120)
            
            for i, (data, labels) in pbar:
                # 验证集只有DVS数据，不是配对数据
                dvs_data = data
                labels = labels.to(self.device)
                
                # DVS数据编码处理
                if len(dvs_data.shape) == 4:  # (N, 2, H, W)
                    dvs_data = dvs_data.unsqueeze(1).repeat(1, self.args.T, 1, 1, 1)  # (N, T, 2, H, W)
                
                dvs_data = dvs_data.to(self.device)
                
                # 只进行DVS前向传播（测试模式）
                # 在测试模式下，网络返回单个输出：target_clf
                dvs_outputs = self.network(dvs_data.float(), dvs_data.float())
                
                # 计算DVS分类损失
                dvs_mean_out = dvs_outputs.mean(1)  # (N, num_classes)
                dvs_clf_loss = self.criterion(dvs_outputs, labels)
                
                # 统计
                total_loss += dvs_clf_loss.item()
                val_num += labels.size(0)
                
                _, dvs_predicted = dvs_mean_out.max(1)
                dvs_val_correct += dvs_predicted.eq(labels).sum().item()
                
                # 重置网络状态
                reset_net(self.network)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{dvs_clf_loss.item():.4f}',
                    'DVS_Acc': f'{dvs_val_correct/val_num:.3f}'
                })
        
        val_acc = dvs_val_correct / val_num  # 只关注DVS准确率
        val_loss = total_loss / len(val_loader)
        
        return val_acc, val_loss
