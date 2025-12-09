# -*- coding: utf-8 -*-
"""
RGB-only训练器 (消融实验1)
用于验证对齐函数的有效性

使用rgb_input层进行预训练，后续只迁移共享特征层参数

用途：
- 预训练阶段：使用RGB 3通道输入 + rgb_input层
- 迁移学习阶段：只迁移features、bottleneck、classifier层的参数
- DVS微调阶段：使用dvs_input层（2通道），共享特征层已有预训练参数
"""

import os
import time
import torch
from tqdm import tqdm


class RGBOnlyTrainer:
    """
    RGB-only训练器 (消融实验1)
    使用rgb_input层进行预训练，后续只迁移共享特征层参数
    """
    def __init__(self, args, device, writer, model, optimizer, criterion, scheduler, encoder, save_dir):
        """
        初始化RGB-only训练器
        
        Args:
            args: 训练参数
            device: 设备 (cuda/cpu)
            writer: TensorBoard writer
            model: 模型 (需要有rgb_input层)
            optimizer: 优化器
            criterion: 损失函数
            scheduler: 学习率调度器
            encoder: 编码器 (TimeEncoder等)
            save_dir: 模型保存目录
        """
        self.args = args
        self.device = device
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.encoder = encoder
        self.save_dir = save_dir
        
        self.best_train_acc = 0
        self.best_test_acc = 0
    
    def forward_rgb(self, data):
        """
        使用rgb_input层进行前向传播
        
        Args:
            data: 输入数据 (N, T, 3, H, W)
            
        Returns:
            outputs: 模型输出 (N, T, num_classes)
        """
        if hasattr(self.model, 'rgb_input'):
            # 手动调用rgb_input层和后续层
            x, _ = self.model.rgb_input(data)
            x = self.model.features(x)
            x = torch.flatten(x, 2)
            x = self.model.bottleneck(x)
            x = self.model.bottleneck_lif_node(x)
            outputs = self.model.classifier(x)
        else:
            # VGGSNNwoAP等模型直接调用
            outputs = self.model(data)
        return outputs
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            train_loss: 训练损失
            train_acc: 训练准确率
            time_cost: 训练耗时(分钟)
        """
        self.model.train()
        start = time.time()
        train_loss = 0
        train_num = 0
        train_correct = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f'Epoch [{epoch+1}/{self.args.epoch}]', leave=True)
        
        for i, (data, labels) in pbar:
            self.optimizer.zero_grad()
            
            # RGB图像编码: (N, 3, H, W) -> (N, T, 3, H, W)
            data = self.encoder(data, out_channel=3)  # 保持3通道
            data, labels = data.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.forward_rgb(data)
            mean_out = outputs.mean(1)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            train_loss += loss.item()
            loss.mean().backward()
            self.optimizer.step()
            
            # 统计准确率
            train_num += float(labels.size(0))
            _, predicted = mean_out.cpu().max(1)
            train_correct += float(predicted.eq(labels.cpu()).sum().item())
            
            # 更新进度条
            current_loss = train_loss / (i + 1)
            current_acc = train_correct / train_num
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.3f}'})
        
        pbar.close()
        
        train_acc = train_correct / train_num
        train_loss = train_loss / train_num
        time_cost = (time.time() - start) / 60
        
        return train_loss, train_acc, time_cost
    
    def test(self, test_loader):
        """
        测试模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            test_loss: 测试损失
            test_acc: 测试准确率
        """
        self.model.eval()
        test_loss = 0
        test_num = 0
        test_correct = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                # RGB图像编码
                data = self.encoder(data, out_channel=3)
                data, labels = data.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.forward_rgb(data)
                mean_out = outputs.mean(1)
                loss = self.criterion(outputs, labels)
                
                # 统计
                test_loss += loss.item()
                test_num += float(labels.size(0))
                _, predicted = mean_out.cpu().max(1)
                test_correct += float(predicted.eq(labels.cpu()).sum().item())
        
        test_acc = test_correct / test_num
        test_loss = test_loss / test_num
        
        return test_loss, test_acc
    
    def save_checkpoint(self, epoch, train_acc, test_acc):
        """
        保存最佳模型
        
        Args:
            epoch: 当前epoch
            train_acc: 训练准确率
            test_acc: 测试准确率
            
        Returns:
            save_path: 保存路径
        """
        save_path = os.path.join(self.save_dir, "rgb_only_pretrained_best.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_train_acc': self.best_train_acc,
            'best_test_acc': self.best_test_acc,
            'args': self.args
        }, save_path)
        return save_path
    
    def train(self, train_loader, test_loader):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            
        Returns:
            best_train_acc: 最佳训练准确率
            best_test_acc: 最佳测试准确率
        """
        print("\n开始RGB-only预训练...")
        
        for epoch in range(self.args.epoch):
            # 训练
            train_loss, train_acc, time_cost = self.train_epoch(train_loader, epoch)
            print(f'Epoch:[{epoch}/{self.args.epoch}]\t time cost: {time_cost:.2f}min\t '
                  f'train_loss={train_loss:.5f}\t train_acc={train_acc:.3f}')
            
            # 测试
            test_loss, test_acc = self.test(test_loader)
            print(f'Epoch:[{epoch}/{self.args.epoch}]\t test_loss={test_loss:.5f}\t test_acc={test_acc:.3f}')
            
            # 记录到TensorBoard
            self.writer.add_scalar(tag="train/accuracy", scalar_value=train_acc, global_step=epoch)
            self.writer.add_scalar(tag="train/loss", scalar_value=train_loss, global_step=epoch)
            self.writer.add_scalar(tag="test/accuracy", scalar_value=test_acc, global_step=epoch)
            self.writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=epoch)
            
            # 更新最佳准确率
            if train_acc > self.best_train_acc:
                self.best_train_acc = train_acc
            
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                save_path = self.save_checkpoint(epoch, train_acc, test_acc)
                print(f"✓ Saved best model with test_acc={test_acc:.3f}")
            
            print(f"Best train acc: {self.best_train_acc:.3f}, Best test acc: {self.best_test_acc:.3f}")
            
            # 学习率调度
            self.scheduler.step()
        
        print(f'\nRGB-only预训练完成!')
        print(f'最佳训练准确率: {self.best_train_acc:.4f}')
        print(f'最佳测试准确率: {self.best_test_acc:.4f}')
        
        return self.best_train_acc, self.best_test_acc

