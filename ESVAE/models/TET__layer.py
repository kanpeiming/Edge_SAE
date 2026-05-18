# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: TET_layer.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network
via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training,
except 'ChannelAttentionLayer' and 'TemporalAttentionLayer'.
"""

import torch
import torch.nn as nn


class ChannelAttentionLayer(nn.Module):
    """
    A channel-based attention class that allocates attention along channel dimensions.
    """

    def __init__(self, in_dim, T):
        super(ChannelAttentionLayer, self).__init__()
        self.T = T
        self.channel_in = in_dim
        self.query = SeqToANNContainer(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.key = SeqToANNContainer(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.value = SeqToANNContainer(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        batch_size, T, C, H, W = x.shape
        proj_query = self.key(x).sum(1).reshape(batch_size, C, H * W)  # (N, C, H*W)
        proj_key = self.query(y).sum(1).reshape(batch_size, C, H * W).permute(0, 2, 1)  # (N, H*W, C)
        proj_value = self.value(x).reshape(batch_size, T, C, H * W)  # (N, T, C, H*W)

        similarity = torch.bmm(proj_query, proj_key)  # (N, C, C)
        score = self.softmax(similarity).permute(0, 2, 1)  # (N, C, C) 沿第2维加和等于1，即score[0, 0, :].sum() = 1
        score = score.unsqueeze(1).repeat(1, T, 1, 1)  # (N, T, C, C)

        out = torch.matmul(score, proj_value).reshape(batch_size, T, C, H, W)  # (N, T, C, H, W)
        return out


class TemporalAttentionLayer(nn.Module):
    """
    A temporal attention class that allocates attention along time dimensions.
    """

    def __init__(self, in_dim, T):
        super(TemporalAttentionLayer, self).__init__()
        self.T = T
        self.channel_in = in_dim
        self.query = SeqToANNContainer(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.key = SeqToANNContainer(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.value = SeqToANNContainer(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))

        self.score_net = nn.Sequential(nn.Linear(self.T * 2, 64), nn.ReLU(), nn.Linear(64, self.T), nn.Sigmoid())

    def forward(self, x, y):
        batch_size, T, C, H, W = x.shape
        proj_query = self.key(x).sum(-1).sum(-1).sum(-1)  # (N, T)
        proj_key = self.query(y).sum(-1).sum(-1).sum(-1)  # (N, T)
        proj_value = self.value(x).reshape(batch_size, T, C * H * W)  # (N, T, C * H * W)

        temporal_feature = torch.cat([proj_query, proj_key], dim=-1)
        score = self.score_net(temporal_feature)  # (N, T)
        score = score.unsqueeze(1).repeat(1, T, 1)  # (N, T, T)

        out = torch.matmul(score, proj_value).reshape(batch_size, T, C, H, W)  # (N, T, C, H, W)
        return out


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, return_mem=False):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()
        self.return_mem = return_mem

    def forward(self, x):
        x = self.fwd(x)
        if self.return_mem:
            x, mem = self.act(x, self.return_mem)
            return x, mem
        else:
            x = self.act(x)
            return x


class LayerWithAttention(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, T, attention_type):
        super(LayerWithAttention, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        if attention_type == 'C':
            # Channel Attention
            self.attention = ChannelAttentionLayer(out_plane, T)
        elif attention_type == 'T':
            # Temporal Attention
            self.attention = TemporalAttentionLayer(out_plane, T)
        else:
            self.attention = None
        self.act = LIFSpike()
        self.attention_type = attention_type

    def forward(self, xy):
        x, y = xy
        x = self.fwd(x)
        y = self.fwd(y)
        if self.attention:
            x = self.attention(x, y)
            y = self.attention(y, y)
        x = self.act(x)
        y = self.act(y)
        return (x, y)


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x, return_mem=False):
        mem = 0
        mem_pot = []
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            if return_mem:
                mem_pot.append(mem)
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        if return_mem:
            return torch.stack(spike_pot, dim=1), torch.stack(mem_pot, dim=1)
        else:
            return torch.stack(spike_pot, dim=1)


class EventMidFrameAttention(nn.Module):
    """
    事件中间稳定帧引导的时序注意力模块
    
    核心思想：
    - 取事件序列中间两帧作为稳定边缘先验（Q）
    - 用完整事件序列构造 K 和 V
    - 通过轻量卷积注意力增强事件表示
    - 使用残差连接保持原有特征
    
    Args:
        in_channels: 输入通道数
        reduction: 注意力通道压缩比例，默认8
    """
    def __init__(self, in_channels, reduction=8):
        super(EventMidFrameAttention, self).__init__()
        self.in_channels = in_channels
        
        # 用于生成 Q/K/V 的轻量卷积
        self.query_conv = SeqToANNContainer(nn.Conv2d(in_channels, in_channels // reduction, 1))
        self.key_conv = SeqToANNContainer(nn.Conv2d(in_channels, in_channels // reduction, 1))
        self.value_conv = SeqToANNContainer(nn.Conv2d(in_channels, in_channels, 1))
        
        # 输出投影
        self.out_conv = SeqToANNContainer(nn.Conv2d(in_channels, in_channels, 1))
        
        # 残差权重（可学习）
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: (N, T, C, H, W) 事件特征序列
        Returns:
            out: (N, T, C, H, W) 增强后的事件特征
        """
        batch_size, T, C, H, W = x.shape
        
        # 取中间两帧作为稳定先验
        t1 = T // 2 - 1
        t2 = T // 2
        # 处理 T < 2 的边界情况
        if T < 2:
            t1, t2 = 0, 0
        elif T == 2:
            t1, t2 = 0, 1
        
        # 平均中间两帧得到稳定先验 (N, 1, C, H, W)
        stable_prior = (x[:, t1:t1+1, :, :, :] + x[:, t2:t2+1, :, :, :]) / 2.0
        
        # 生成 Query（来自稳定先验）
        # (N, 1, C, H, W) -> (N, 1, C', H, W) -> (N, C', H*W)
        Q = self.query_conv(stable_prior).squeeze(1).view(batch_size, -1, H * W)
        
        # 生成 Key 和 Value（来自完整序列）
        # (N, T, C, H, W) -> (N, T, C', H, W) -> (N, T, C', H*W)
        K = self.key_conv(x).view(batch_size, T, -1, H * W)
        V = self.value_conv(x).view(batch_size, T, C, H * W)
        
        # 计算注意力权重
        # Q: (N, C', H*W), K: (N, T, C', H*W)
        # 对每个时间步计算相似度
        attention_scores = []
        for t in range(T):
            # (N, C', H*W) @ (N, C', H*W).T -> (N, H*W, H*W)
            score = torch.bmm(Q.transpose(1, 2), K[:, t, :, :])  # (N, H*W, H*W)
            attention_scores.append(score)
        
        # (N, T, H*W, H*W)
        attention_scores = torch.stack(attention_scores, dim=1)
        attention_weights = self.softmax(attention_scores)
        
        # 应用注意力到 Value
        # (N, T, H*W, H*W) @ (N, T, C, H*W).transpose -> (N, T, H*W, C)
        out = torch.matmul(attention_weights, V.transpose(2, 3))  # (N, T, H*W, C)
        out = out.transpose(2, 3).contiguous()  # (N, T, C, H*W)
        out = out.view(batch_size, T, C, H, W)
        
        # 输出投影
        out = self.out_conv(out)
        
        # 残差连接（可学习权重）
        out = x + self.gamma * out
        
        return out


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y
