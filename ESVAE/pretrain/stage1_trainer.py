# -*- coding: utf-8 -*-
"""
Stage 1: Event-aware Bridge Pretraining Trainer

目标：
- 在不破坏Stage 0学到的结构表示的前提下
- 引入真实DVS数据，使backbone具备事件感知能力

核心策略：
- 所有输入统一走 dvs_input（不使用rgb_input）
- RGB-edge 和 DVS 通过 category-level 匹配
- 使用 EventBridgeHead 预测事件统计
- 分类损失作为稳定正则项（小权重）
"""

import os
import time
import torch
from torch import nn
from tqdm import tqdm
from pretrain.pretrainer import TLTrainer
from pretrain.Edge import compute_event_statistics
from tl_utils.common_utils import accuracy

try:
    from spikingjelly.activation_based.functional import reset_net
except:
    from utils.common_utils import reset_net


class Stage1BridgeTrainer(TLTrainer):
    """
    Stage 1: Event-aware Bridge Pretraining Trainer
    
    核心流程：
    1. RGB-edge 和 DVS 都走 dvs_input
    2. 提取 bottleneck 特征
    3. 使用 EventBridgeHead 预测 DVS 事件统计
    4. Bridge loss（主导） + 弱化的分类loss
    """
    
    def __init__(self, args, device, writer, network, optimizer, criterion, scheduler, model_path):
        super().__init__(args, device, writer, network, optimizer, criterion, scheduler, model_path)
        
        # Stage 1 特定配置
        self.bridge_loss_weight = getattr(args, 'bridge_loss_weight', 1.0)
        self.cls_loss_weight = getattr(args, 'cls_loss_weight', 0.1)  # 小权重
        
        # Bridge loss类型
        self.bridge_loss_fn = nn.L1Loss()  # L1 loss for event statistics
        
        # 最佳模型路径
        if model_path.endswith('.pth'):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path
        self.best_model_path = os.path.join(model_dir, "stage1_best_model.pth")
        self.best_total_loss = float('inf')
    
    def save_model_best(self, epoch):
        """保存当前最佳模型"""
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_total_loss,
        }, self.best_model_path)
        print(f"✓ Best model saved at epoch {epoch} with loss {self.best_total_loss:.5f}")
    
    def train(self, train_loader):
        """
        Stage 1训练主循环
        
        训练策略：
        1. RGB-edge 和 DVS 都走 dvs_input
        2. 提取 bottleneck 特征
        3. 使用 EventBridgeHead 预测 DVS 事件统计
        4. Bridge loss + 弱化的分类loss
        """
        for epoch in range(self.args.epochs):
            self.network.train()
            start = time.time()
            
            # 统计变量
            total_loss = 0
            bridge_loss_sum = 0
            edge_cls_loss_sum = 0
            dvs_cls_loss_sum = 0
            edge_correct = 0
            dvs_correct = 0
            train_num = 0
            
            # 进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f'Stage1 Epoch [{epoch+1}/{self.args.epochs}]',
                       ncols=140)
            
            for i, ((edge_data, dvs_data), labels) in pbar:
                self.optimizer.zero_grad()
                
                # 数据移到设备
                edge_data = edge_data.to(self.device).float()  # (N, 2, H, W) 已预处理
                dvs_data = dvs_data.to(self.device).float()
                labels = labels.to(self.device)
                
                # Step 1: 编码为时间序列
                # Edge: (N, 2, H, W) -> (N, T, 2, H, W)
                if edge_data.dim() == 4 and edge_data.shape[1] == 2:
                    edge_data = self.encoder_dict[self.args.encoder_type](edge_data)
                
                # DVS: (N, T, 2, H, W) 已经是时间序列
                if dvs_data.dim() == 4:  # (N, 2, H, W)
                    dvs_data = self.encoder_dict[self.args.encoder_type](dvs_data)
                
                # 计算真实DVS的事件统计（提前计算，不需要梯度）
                # 需要resize到与EventBridgeHead输出相同的尺寸
                with torch.no_grad():
                    target_size = (self.args.img_shape, self.args.img_shape)
                    true_event_stats = compute_event_statistics(dvs_data, stat_type='density', target_size=target_size)
                
                # ============================================
                # Step 2: Edge路径 - 完整前向+反向传播
                # ============================================
                edge_input, _ = self.network.dvs_input(edge_data)
                edge_features = self.network.features(edge_input)
                edge_features_flat = torch.flatten(edge_features, 2)
                edge_bottleneck = self.network.bottleneck(edge_features_flat)
                edge_bottleneck_spike, _ = self.network.bottleneck_lif_node(
                    edge_bottleneck, return_mem=True
                )
                edge_outputs = self.network.classifier(edge_bottleneck_spike)
                
                # Bridge预测（从Edge预测DVS事件统计）
                pred_event_stats = self.network.event_bridge_head(edge_bottleneck_spike)
                
                # 计算Edge的损失
                edge_cls_loss = self.criterion(edge_outputs, labels)
                bridge_loss = self.bridge_loss_fn(pred_event_stats, true_event_stats)
                edge_total_loss = (self.bridge_loss_weight * bridge_loss + 
                                  self.cls_loss_weight * edge_cls_loss)
                
                # 计算Edge准确率（在删除前）
                edge_mean_out = edge_outputs.mean(1)
                edge_acc, _ = accuracy(edge_mean_out, labels, topk=(1, 5))
                
                # 反向传播Edge部分
                edge_total_loss.mean().backward()
                
                # 保存统计值
                edge_loss_val = edge_total_loss.item()
                bridge_loss_val = bridge_loss.item()
                edge_cls_loss_val = edge_cls_loss.item()
                edge_acc_val = edge_acc.item()
                
                # 释放Edge相关显存
                del edge_input, edge_features, edge_features_flat, edge_bottleneck
                del edge_bottleneck_spike, edge_outputs, pred_event_stats
                del edge_mean_out, edge_cls_loss, bridge_loss, edge_total_loss
                
                # 重置网络状态（为DVS路径准备）
                reset_net(self.network)
                
                # ============================================
                # Step 3: DVS路径 - 完整前向+反向传播
                # ============================================
                dvs_input, _ = self.network.dvs_input(dvs_data)
                dvs_features = self.network.features(dvs_input)
                dvs_features_flat = torch.flatten(dvs_features, 2)
                dvs_bottleneck = self.network.bottleneck(dvs_features_flat)
                dvs_bottleneck_spike, _ = self.network.bottleneck_lif_node(
                    dvs_bottleneck, return_mem=True
                )
                dvs_outputs = self.network.classifier(dvs_bottleneck_spike)
                
                # 计算DVS的损失
                dvs_cls_loss = self.criterion(dvs_outputs, labels)
                dvs_total_loss = self.cls_loss_weight * dvs_cls_loss
                
                # 计算DVS准确率（在删除前）
                dvs_mean_out = dvs_outputs.mean(1)
                dvs_acc, _ = accuracy(dvs_mean_out, labels, topk=(1, 5))
                
                # 反向传播DVS部分（累积梯度）
                dvs_total_loss.mean().backward()
                
                # 保存统计值
                dvs_loss_val = dvs_total_loss.item()
                dvs_cls_loss_val = dvs_cls_loss.item()
                dvs_acc_val = dvs_acc.item()
                
                # 释放DVS相关显存
                del dvs_input, dvs_features, dvs_features_flat, dvs_bottleneck
                del dvs_bottleneck_spike, dvs_outputs, dvs_mean_out
                del dvs_cls_loss, dvs_total_loss
                
                # ============================================
                # Step 4: 更新参数
                # ============================================
                self.optimizer.step()
                
                # 重置网络状态
                reset_net(self.network)
                
                # ============================================
                # Step 5: 统计
                # ============================================
                total_loss += (edge_loss_val + dvs_loss_val)
                bridge_loss_sum += bridge_loss_val
                edge_cls_loss_sum += edge_cls_loss_val
                dvs_cls_loss_sum += dvs_cls_loss_val
                edge_correct += edge_acc_val
                dvs_correct += dvs_acc_val
                train_num += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{edge_loss_val + dvs_loss_val:.4f}',
                    'Bridge': f'{bridge_loss_val:.4f}',
                    'EdgeAcc': f'{edge_acc_val:.2f}',
                    'DVSAcc': f'{dvs_acc_val:.2f}'
                })
            
            # Epoch结束统计
            avg_loss = total_loss / train_num
            avg_bridge_loss = bridge_loss_sum / train_num
            avg_edge_cls_loss = edge_cls_loss_sum / train_num
            avg_dvs_cls_loss = dvs_cls_loss_sum / train_num
            avg_edge_acc = edge_correct / train_num
            avg_dvs_acc = dvs_correct / train_num
            
            elapsed = time.time() - start
            
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}] 完成 (耗时: {elapsed:.1f}s)")
            print(f"  总损失: {avg_loss:.5f}")
            print(f"  Bridge损失: {avg_bridge_loss:.5f}")
            print(f"  Edge分类损失: {avg_edge_cls_loss:.5f} | 准确率: {avg_edge_acc:.2f}%")
            print(f"  DVS分类损失: {avg_dvs_cls_loss:.5f} | 准确率: {avg_dvs_acc:.2f}%")
            
            # TensorBoard记录
            self.writer.add_scalar('stage1/train_loss', avg_loss, epoch)
            self.writer.add_scalar('stage1/bridge_loss', avg_bridge_loss, epoch)
            self.writer.add_scalar('stage1/edge_cls_loss', avg_edge_cls_loss, epoch)
            self.writer.add_scalar('stage1/dvs_cls_loss', avg_dvs_cls_loss, epoch)
            self.writer.add_scalar('stage1/edge_accuracy', avg_edge_acc, epoch)
            self.writer.add_scalar('stage1/dvs_accuracy', avg_dvs_acc, epoch)
            
            # 保存最佳模型
            if avg_loss < self.best_total_loss:
                self.best_total_loss = avg_loss
                self.save_model_best(epoch)
            
            # 学习率调整
            if self.scheduler is not None:
                self.scheduler.step()
        
        return avg_edge_cls_loss, avg_dvs_cls_loss  # 返回loss而不是acc
    
    def test(self, test_loader):
        """
        测试函数：在DVS测试集上评估
        """
        self.network.eval()
        test_loss = 0
        test_correct = 0
        test_correct5 = 0
        test_num = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device).float()
                labels = labels.to(self.device)
                
                # 前向传播（只使用DVS数据）
                outputs = self.network(None, data)
                
                mean_out = outputs.mean(1)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                test_acc1, test_acc5 = accuracy(mean_out, labels, topk=(1, 5))
                test_correct += test_acc1.item()
                test_correct5 += test_acc5.item()
                test_num += 1
                
                reset_net(self.network)
        
        return test_loss / test_num, test_correct / test_num, test_correct5 / test_num

