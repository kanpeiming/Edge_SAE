#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
边缘分支预训练器（3通道多阈值Canny）

边缘检测任务：RGB图像 -> 3通道边缘图预测（弱/中/强边缘）
损失：BCE Loss（重加权）
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class EdgePretrainer:
    """边缘分支预训练器"""
    
    def __init__(self,
                 args,
                 device,
                 writer: SummaryWriter,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_path: str):
        self.args = args
        self.device = device
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_path = model_path
        
        # BCE损失（重加权，处理边缘/背景不平衡）
        # 边缘像素较少，给予更高权重
        # 弱边缘权重较小（更多像素），强边缘权重较大（更少像素）
        # pos_weight 形状需要是 (1, C, 1, 1) 以匹配 (B, C, H, W) 的输入
        pos_weight = torch.tensor([8.0, 10.0, 12.0]).view(1, 3, 1, 1).to(device)  # [弱, 中, 强边缘]
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train(self, train_loader, val_loader):
        """主训练循环"""
        
        print("\n" + "="*80)
        print("开始边缘分支预训练".center(80))
        print("="*80)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            start_time = time.time()
            
            # 训练指标
            total_loss = 0
            num_batches = 0
            
            # 进度条
            pbar = tqdm(enumerate(train_loader),
                       total=len(train_loader),
                       desc=f'Epoch {epoch+1:3d}/{self.args.epochs}',
                       ncols=100,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for batch_idx, (rgb_imgs, edge_labels, _) in pbar:
                self.optimizer.zero_grad()
                
                # 移动到设备
                rgb_imgs = rgb_imgs.to(self.device).float()      # (B, 3, H, W)
                edge_labels = edge_labels.to(self.device).float()  # (B, 3, H, W)
                
                # 前向传播
                edge_preds = self.model(rgb_imgs)  # (B, 3, H, W)
                
                # 调试：第一个batch打印形状
                if batch_idx == 0:
                    print(f"\n[调试] 第一个batch形状:")
                    print(f"  rgb_imgs: {rgb_imgs.shape}")
                    print(f"  edge_labels: {edge_labels.shape}")
                    print(f"  edge_preds: {edge_preds.shape}")
                
                # 计算损失
                loss = self.criterion(edge_preds, edge_labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{total_loss/num_batches:.4f}'
                })
            
            pbar.close()
            self.scheduler.step()
            
            # 平均损失
            avg_train_loss = total_loss / num_batches
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # Epoch总结
            epoch_time = (time.time() - start_time) / 60
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f'\n{"="*80}')
            print(f'Epoch [{epoch+1:3d}/{self.args.epochs}] - Time: {epoch_time:.2f}min - LR: {lr:.6f}')
            print(f'{"-"*80}')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss:   {val_loss:.4f}')
            
            # TensorBoard记录
            self.writer.add_scalar('train/loss', avg_train_loss, epoch)
            self.writer.add_scalar('train/lr', lr, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                prev_best = self.best_val_loss
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch)
                if prev_best == float('inf'):
                    print(f'  ✓ Best Model Saved! Val Loss: {val_loss:.4f}')
                else:
                    print(f'  ✓ New Best Model! Val Loss: {val_loss:.4f} (Prev: {prev_best:.4f})')
            
            print(f'{"="*80}\n')
        
        print("\n" + "="*80)
        print("边缘预训练完成".center(80))
        print(f"Best Val Loss: {self.best_val_loss:.4f} at Epoch {self.best_epoch+1}".center(80))
        print("="*80 + "\n")
        
        return self.best_val_loss
    
    def validate(self, val_loader):
        """验证循环"""
        self.model.eval()
        
        val_loss = 0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(val_loader,
                   desc='Validating',
                   ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]',
                   leave=False)
        
        with torch.no_grad():
            for rgb_imgs, edge_labels, _ in pbar:
                rgb_imgs = rgb_imgs.to(self.device).float()
                edge_labels = edge_labels.to(self.device).float()
                
                # 前向传播
                edge_preds = self.model(rgb_imgs)
                loss = self.criterion(edge_preds, edge_labels)
                
                val_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{val_loss/num_batches:.4f}'
                })
        
        pbar.close()
        avg_val_loss = val_loss / num_batches
        
        return avg_val_loss
    
    def save_model(self, epoch):
        """保存模型检查点"""
        os.makedirs(self.model_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args,
        }
        
        # 保存最佳模型
        best_path = os.path.join(self.model_path, 'best_edge_branch.pth')
        torch.save(checkpoint, best_path)
        
        # 保存最新模型
        latest_path = os.path.join(self.model_path, 'latest_edge_branch.pth')
        torch.save(checkpoint, latest_path)

