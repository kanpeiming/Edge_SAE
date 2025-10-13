# -*- coding: utf-8 -*-
"""
Trainer for Structure-Guided DVS Classification with Contrastive Learning

@description: Training with cross-attention fusion and contrastive learning
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import sys
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tl_utils.common_utils import accuracy

try:
    from spikingjelly.activation_based.functional import reset_net
except:
    def reset_net(model):
        """Fallback reset function if spikingjelly not available"""
        for m in model.modules():
            if hasattr(m, 'reset'):
                m.reset()


# ============================================================================
# Contrastive Learning Loss (对比学习损失)
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE对比学习损失
    拉近同类RGB-DVS特征，推远异类特征
    
    核心思想：
    - 同类样本的RGB和DVS特征应该接近
    - 不同类样本的RGB和DVS特征应该远离
    - 缓解样本量不匹配问题：通过特征对齐学习跨模态表示
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, dvs_feat, rgb_feat, labels):
        """
        计算对比学习损失
        
        Args:
            dvs_feat: (B, D) - DVS特征向量
            rgb_feat: (B, D) - RGB特征向量
            labels: (B,) - 类别标签
        
        Returns:
            loss: 对比学习损失
        """
        # Normalize features
        dvs_feat = F.normalize(dvs_feat, dim=1)
        rgb_feat = F.normalize(rgb_feat, dim=1)
        
        batch_size = dvs_feat.shape[0]
        
        # 计算相似度矩阵
        # sim_matrix[i, j] = similarity(dvs_i, rgb_j)
        sim_matrix = torch.matmul(dvs_feat, rgb_feat.T) / self.temperature  # (B, B)
        
        # 创建正样本mask：同类为正样本
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(dvs_feat.device)  # (B, B)
        
        # 对于每个DVS样本，找到同类的RGB样本作为正样本
        # 使用InfoNCE loss
        # loss = -log(exp(sim_pos) / sum(exp(sim_all)))
        
        # 计算exp(sim)
        sim_exp = torch.exp(sim_matrix)  # (B, B)
        
        # 正样本的相似度和
        pos_sim = (sim_exp * mask_positive).sum(dim=1)  # (B,)
        
        # 所有样本的相似度和（分母）
        all_sim = sim_exp.sum(dim=1)  # (B,)
        
        # InfoNCE loss
        loss = -torch.log(pos_sim / (all_sim + 1e-8))
        
        # 平均损失
        loss = loss.mean()
        
        return loss


class EdgeGuidedTrainer:
    """
    Trainer for Structure-Guided DVS Classification with Contrastive Learning.
    
    核心特性:
    - Cross-Attention融合 (DVS动态查询RGB结构)
    - 对比学习 (缓解样本量不匹配)
    - 支持更多epoch训练 (DVS样本少)
    """
    
    def __init__(self,
                 args,
                 device,
                 writer: SummaryWriter,
                 model: nn.Module,
                 optimizer,
                 criterion,
                 scheduler,
                 model_path: str,
                 baseline_model: Optional[nn.Module] = None,
                 use_contrastive: bool = True,
                 contrastive_weight: float = 0.3):
        self.args = args
        self.device = device
        self.writer = writer
        self.model = model
        self.baseline_model = baseline_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_path = model_path
        
        self.best_train_acc = 0
        self.best_val_acc = 0
        self.best_epoch = 0
        
        # 对比学习配置
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss(temperature=0.07)
            print(f"✓ 对比学习已启用 (权重: {contrastive_weight})")
        
        # For baseline comparison
        if baseline_model is not None:
            self.baseline_optimizer = torch.optim.Adam(
                baseline_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        
        print("\n" + "="*80)
        print("Starting Training".center(80))
        print("="*80)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            if self.baseline_model is not None:
                self.baseline_model.train()
            
            start_time = time.time()
            
            # Training metrics
            total_loss = 0
            dvs_correct = 0
            dvs_correct_top5 = 0
            baseline_correct = 0
            baseline_count = 0
            total_samples = 0
            
            # Progress bar for batches
            pbar = tqdm(enumerate(train_loader), 
                       total=len(train_loader),
                       desc=f'Epoch {epoch+1:3d}/{self.args.epochs}',
                       ncols=100,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for batch_idx, (data, labels) in pbar:
                self.optimizer.zero_grad()
                
                # Unpack data: can be (rgb_img, dvs_data) tuple or just dvs_data
                if isinstance(data, (tuple, list)):
                    rgb_data, dvs_data = data
                    rgb_data = rgb_data.to(self.device).float()
                    dvs_data = dvs_data.to(self.device).float()
                else:
                    # Backward compatibility: only DVS data
                    rgb_data = None
                    dvs_data = data.to(self.device).float()
                
                labels = labels.to(self.device)
                
                # Forward pass with RGB-based structure guidance
                # 使用return_features获取中间特征用于对比学习
                if rgb_data is not None and self.use_contrastive:
                    dvs_outputs, features = self.model(
                        rgb_data=rgb_data, 
                        dvs_data=dvs_data, 
                        return_features=True
                    )
                elif rgb_data is not None:
                    dvs_outputs = self.model(rgb_data=rgb_data, dvs_data=dvs_data)
                else:
                    dvs_outputs = self.model(dvs_data=dvs_data)
                
                # 计算分类损失
                cls_loss = self.criterion(dvs_outputs, labels)
                loss = cls_loss
                
                # 计算对比学习损失（如果启用）
                contrastive_loss_value = 0.0
                if self.use_contrastive and rgb_data is not None:
                    contrastive_loss_value = self.contrastive_loss(
                        features['dvs_repr'],  # DVS特征表示
                        features['rgb_repr'],  # RGB特征表示
                        labels
                    )
                    loss = cls_loss + self.contrastive_weight * contrastive_loss_value
                
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_samples += labels.size(0)
                
                dvs_mean_out = dvs_outputs.mean(1)  # Average over time
                acc1, acc5 = accuracy(dvs_mean_out, labels, topk=(1, 5))
                dvs_correct += acc1.item()
                dvs_correct_top5 += acc5.item()
                
                # Reset neuron states
                reset_net(self.model)
                
                # Baseline comparison (periodic evaluation to save time)
                if self.baseline_model is not None and batch_idx % 5 == 0:
                    self.baseline_optimizer.zero_grad()
                    # Baseline model only uses DVS data
                    baseline_outputs = self.baseline_model(dvs_data)
                    baseline_loss = self.criterion(baseline_outputs, labels)
                    baseline_loss.backward()
                    self.baseline_optimizer.step()
                    
                    with torch.no_grad():
                        baseline_mean_out = baseline_outputs.mean(1)
                        baseline_acc1, _ = accuracy(baseline_mean_out, labels, topk=(1, 5))
                        baseline_correct += baseline_acc1.item()
                        baseline_count += labels.size(0)
                    reset_net(self.baseline_model)
                
                # Update progress bar with current batch metrics
                current_acc = (dvs_correct / total_samples) * 100
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            
            pbar.close()
            self.scheduler.step()
            
            # Average metrics for the epoch
            avg_loss = total_loss / len(train_loader)
            train_acc = dvs_correct / total_samples
            train_acc_top5 = dvs_correct_top5 / total_samples
            baseline_acc = baseline_correct / baseline_count if baseline_count > 0 else None
            
            # Validation
            val_loss, val_acc, val_acc_top5 = self.validate(val_loader)
            
            # Epoch summary
            epoch_time = (time.time() - start_time) / 60
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f'\n{"="*80}')
            print(f'Epoch [{epoch+1:3d}/{self.args.epochs}] - Time: {epoch_time:.2f}min - LR: {lr:.6f}')
            print(f'{"-"*80}')
            print(f'  Train  │ Loss: {avg_loss:.4f} │ Acc@1: {train_acc*100:6.2f}% │ Acc@5: {train_acc_top5*100:6.2f}%')
            if self.use_contrastive and contrastive_loss_value > 0:
                print(f'  Contrastive Loss: {contrastive_loss_value:.4f}')
            if baseline_acc:
                improvement = (train_acc - baseline_acc) * 100
                print(f'  Baseline│              │ Acc@1: {baseline_acc*100:6.2f}% │ Improve: +{improvement:.2f}%')
            print(f'  Val    │ Loss: {val_loss:.4f} │ Acc@1: {val_acc*100:6.2f}% │ Acc@5: {val_acc_top5*100:6.2f}%')
            
            # TensorBoard logging
            self.writer.add_scalar('train/loss', avg_loss, epoch)
            self.writer.add_scalar('train/acc_top1', train_acc, epoch)
            self.writer.add_scalar('train/acc_top5', train_acc_top5, epoch)
            self.writer.add_scalar('train/lr', lr, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/acc_top1', val_acc, epoch)
            self.writer.add_scalar('val/acc_top5', val_acc_top5, epoch)
            if self.use_contrastive and contrastive_loss_value > 0:
                self.writer.add_scalar('train/contrastive_loss', contrastive_loss_value, epoch)
            
            if baseline_acc:
                self.writer.add_scalar('train/baseline_acc', baseline_acc, epoch)
                self.writer.add_scalar('train/improvement', train_acc - baseline_acc, epoch)
            
            # Save best model
            if val_acc > self.best_val_acc:
                prev_best = self.best_val_acc
                self.best_val_acc = val_acc
                self.best_train_acc = train_acc
                self.best_epoch = epoch
                self.save_model(epoch)
                if prev_best == 0:
                    print(f'  ✓ Best Model Saved! Val Acc: {val_acc*100:.2f}%')
                else:
                    print(f'  ✓ New Best Model! Val Acc: {val_acc*100:.2f}% (Prev: {prev_best*100:.2f}%)')
            
            print(f'{"="*80}\n')
        
        print("\n" + "="*80)
        print("Training Completed".center(80))
        print(f"Best Val Acc: {self.best_val_acc*100:.2f}% at Epoch {self.best_epoch+1}".center(80))
        print("="*80 + "\n")
        
        return self.best_train_acc, self.best_val_acc
    
    def validate(self, val_loader):
        """Validation loop."""
        self.model.eval()
        
        val_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0
        
        # Progress bar for validation
        pbar = tqdm(val_loader, 
                   desc='Validating',
                   ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]',
                   leave=False)
        
        with torch.no_grad():
            for data, labels in pbar:
                # Unpack data: can be (rgb_img, dvs_data) tuple or just dvs_data
                if isinstance(data, (tuple, list)):
                    rgb_data, dvs_data = data
                    rgb_data = rgb_data.to(self.device).float()
                    dvs_data = dvs_data.to(self.device).float()
                else:
                    # Backward compatibility: only DVS data
                    rgb_data = None
                    dvs_data = data.to(self.device).float()
                
                labels = labels.to(self.device)
                
                # Forward pass with RGB-based edge guidance
                if rgb_data is not None:
                    outputs = self.model(rgb_data=rgb_data, dvs_data=dvs_data)
                else:
                    outputs = self.model(dvs_data=dvs_data)
                    
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                total += labels.size(0)
                
                mean_out = outputs.mean(1)
                acc1, acc5 = accuracy(mean_out, labels, topk=(1, 5))
                correct += acc1.item()
                correct_top5 += acc5.item()
                
                # Update progress bar
                current_acc = (correct / total) * 100
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
                
                reset_net(self.model)
        
        pbar.close()
        
        avg_loss = val_loss / len(val_loader)
        avg_acc = correct / total
        avg_acc_top5 = correct_top5 / total
        
        return avg_loss, avg_acc, avg_acc_top5
    
    def test(self, test_loader):
        """Test the best model."""
        print("\n" + "="*80)
        print("Testing Best Model".center(80))
        print("="*80)
        
        # Load best model
        best_model_path = os.path.join(self.model_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
            print(f"  Train Acc: {checkpoint['best_train_acc']*100:.2f}%")
            print(f"  Val Acc: {checkpoint['best_val_acc']*100:.2f}%")
        else:
            print("⚠ Warning: No saved model found, using current model")
        
        print(f'\n{"-"*80}')
        test_loss, test_acc, test_acc_top5 = self.validate(test_loader)
        
        print(f'\n{"-"*80}')
        print(f'Test Results:')
        print(f'  Loss : {test_loss:.4f}')
        print(f'  Acc@1: {test_acc*100:.2f}%')
        print(f'  Acc@5: {test_acc_top5*100:.2f}%')
        print("="*80 + "\n")
        
        return test_loss, test_acc, test_acc_top5
    
    def save_model(self, epoch):
        """Save model checkpoint."""
        os.makedirs(self.model_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_train_acc': self.best_train_acc,
            'args': self.args,
        }
        
        # Save best model
        torch.save(checkpoint, os.path.join(self.model_path, 'best_model.pth'))
        
        # Save latest model
        torch.save(checkpoint, os.path.join(self.model_path, 'latest_model.pth'))

