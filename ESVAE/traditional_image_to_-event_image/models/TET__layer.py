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

        self.score_net = nn.Sequential(nn.Linear(self.T*2, 64), nn.ReLU(), nn.Linear(64, self.T), nn.Sigmoid())

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


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y

