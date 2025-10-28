# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: common_utils.py
@time: 2022/4/19 16:31
"""

import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
try:
    from spikingjelly.clock_driven.encoding import PoissonEncoder
except:
    from spikingjelly.activation_based.encoding import PoissonEncoder


def seed_all(seed=42):
    """
    set random seed.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 将rgb三通道图片复制为时间序列rgb
class TimeEncoder(torch.nn.Module):
    def __init__(self, T, device):
        """
        将rgb三通道图片复制为时间序列rgb图，(N, 3, H, W) -> (N, T, 3, H, W)
        Args:
            T: SNN时间步长度
        """
        super().__init__()
        self.T = T
        self.device = device

    def forward(self, x: torch.Tensor, out_channel=3):
        out = []
        for _ in range(self.T):
            out.append(x[:, :out_channel, ...].unsqueeze(1))
        return torch.cat(out, 1).to(x).to(self.device)  # (N, T, 3, H, W)



# 重新定义泊松编码
class MyPoissonEncoder(PoissonEncoder):
    def __init__(self, T, device):
        """
        泊松编码生成脉冲
        必须确保 ``0 <= x <= 1``。
        Args:
            T: SNN时间步长度
        """
        super().__init__()
        self.T = T
        self.device = device

    def forward(self, x: torch.Tensor, out_channel=3):
        out = []
        for _ in range(self.T):
            out.append(torch.rand_like(x[:, :out_channel, ...]).le(x[:, :out_channel, ...]).unsqueeze(1))
            # torch.rand_like(x)生成与x相同shape的介于[0, 1)之间的随机数， 这个随机数小于等于x中对应位置的元素，则发放脉冲
        return torch.cat(out, 1).to(x).to(self.device)  # (N, T, 2, H, W)


# 定义拉普拉斯泊松编码
class LapPoissonEncoder(PoissonEncoder):
    def __init__(self, T, device, laplace_size=3, original_ratio=0.1, lap_threshold=0.3):
        """
        用拉普拉斯提取图像边缘，并产生对应脉冲图；
        再用原图生成脉冲图，二者混合平滑，得到最终脉冲输出。
        必须确保 ``0 <= x <= 1``。
        Args:
            T: SNN时间步长度
            laplace_size: 拉普拉斯变换尺寸
            original_ratio: 平滑时原图像脉冲序列占比
            lap_threshold: 拉普拉斯生成脉冲图的阈值
        """
        super().__init__()
        self.T = T
        self.device = device
        self.laplace_size = laplace_size
        self.original_ratio = original_ratio
        self.lap_threshold = lap_threshold

    # get laplace image from resize image
    def get_lap(self, tensor_data):
        """
        拉普拉斯获取边缘
        """
        shape = tensor_data.size()
        out = []
        for i in range(shape[0]):
            img = transforms.ToPILImage()(tensor_data[i]).convert('RGB')
            cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            cv_img_lap = cv2.Laplacian(cv_img, cv2.CV_8U, ksize=self.laplace_size)
            tensor_out = transforms.ToTensor()(cv_img_lap)
            tensor_out = tensor_out.reshape((1, 1, shape[2], shape[3]))
            out.append(tensor_out)
        return torch.cat(out, 0)

    def smooth(self, data, lap_data):
        """
        将原图spike与拉普拉斯spike混合
        :param data: 原图spike图像
        :param lap_data: 拉普拉斯变换得到的spike图像
        :param original_ratio: 原始图像占比
        :return: 二者混合平滑后的spike图像
        """
        result = self.original_ratio * data.float() + (1 - self.original_ratio) * lap_data.float()
        result = result > torch.rand(result.size()).to(self.device)
        return result

    def forward(self, x: torch.Tensor):
        out = []
        for _ in range(self.T):
            lap_data = self.get_lap(x).to(self.device)  # (N, 1, H, W)
            source_data = torch.rand_like(x).le(x).to(x).to(self.device)  # (N, 3, H, W)
            lap_spike = lap_data > torch.ones(lap_data.size()).to(self.device) * self.lap_threshold  # (N, 1, H, W)
            smooth_out = self.smooth(source_data, lap_spike)  # (N, 3, H, W)
            out.append(smooth_out.unsqueeze(1))
        return torch.cat(out, 1).to(x).to(self.device)  # (N, T, 3, H, W)


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def set_optimizer(opt, lr, momentum, weight_decay, parameters):
    opt_name = opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = None
    return optimizer


def set_lr_scheduler(lr_scheduler, epochs, optimizer,
                     lr_step_size=30, lr_gamma=0.1,
                     lr_warmup_epochs=5, lr_warmup_method="linear", lr_warmup_decay=0.01):
        lr_scheduler = lr_scheduler.lower()
        if lr_scheduler == "step":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        elif lr_scheduler == "cosa":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - lr_warmup_epochs
            )
        elif lr_scheduler == "exp":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        else:
            main_lr_scheduler = None
        if lr_warmup_epochs > 0:
            if lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
                )
            elif lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=lr_warmup_decay, total_iters=lr_warmup_epochs
                )
            else:
                warmup_lr_scheduler = None
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        return lr_scheduler


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    try:
        with torch.inference_mode():
            maxk = max(topk)
            if target.ndim == 2:
                target = target.max(dim=1)[1]

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[None])

            res = []
            for k in topk:
                correct_k = correct[:k].flatten().sum(dtype=torch.float32)
                res.append(correct_k)
            return res
    except:
        with torch.no_grad():
            maxk = max(topk)
            if target.ndim == 2:
                target = target.max(dim=1)[1]

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[None])

            res = []
            for k in topk:
                correct_k = correct[:k].flatten().sum(dtype=torch.float32)
                res.append(correct_k)
            return res