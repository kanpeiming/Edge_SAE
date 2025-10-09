import torch
import torch.nn as nn
from copy import deepcopy
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

from spikingjelly.activation_based import layer
from utils import linear_CKA, temporal_linear_CKA, MSE, temporal_MSE, MMD_loss
from models.TET__layer import *


__all__ = ['SpikingResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
           'spiking_resnet152', 'spiking_resnext50_32x4d', 'spiking_resnext101_32x8d',
           'spiking_wide_resnet50_2', 'spiking_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, return_mem=False, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride
        self.return_mem = return_mem

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.return_mem:
            out_mem = out + identity
            out_result = self.sn2(out_mem)

            return out_result, out_mem
        else:
            out += identity
            out = self.sn2(out)
            return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, return_mem=False, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride
        self.return_mem = return_mem

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.return_mem:
            out_mem = out + identity
            out_result = self.sn3(out_mem)

            return out_result, out_mem
        else:
            out += identity
            out = self.sn3(out)
            return out


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(SpikingResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class TLSpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(TLSpikingResNet, self).__init__()

        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.rgb_conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        self.rgb_bn1 = norm_layer(self.inplanes)
        self.rgb_sn1 = spiking_neuron(**deepcopy(kwargs))

        self.dvs_conv1 = layer.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        self.dvs_bn1 = norm_layer(self.inplanes)
        self.dvs_sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, return_mem=False, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron,
                                       return_mem=False, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron,
                                       return_mem=False, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron,
                                       return_mem=True, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: callable = None, return_mem=False, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if return_mem and i == blocks-1:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, spiking_neuron=spiking_neuron, return_mem=return_mem, **kwargs))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, source, target, encoder_tl_loss_type='TCKA', feature_tl_loss_type='MSE'):
        """
        Args:
            source: 源域输入，(N, T, 3, H, W) 为 RGB图像，(N, T, 2, H, W)为DVS图像
            target: 目标域输入，(N, T, 3, H, W) 为 RGB图像，(N, T, 2, H, W)为DVS图像
            encoder_tl_loss_type: 包括 'TCKA'(分别计算各时间步mem的CKA，最后求平均), 'CKA'(将各时间步的spike求频率后计算CKA)
            feature_tl_loss_type: 包括 'TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'.
        Returns:
            if self.training:  训练阶段
                Returns:
                    source_clf: 源域分类神经元输出，0/1脉冲序列, (N, T, class_num)
                    target_clf: 目标域分类神经元输出，0/1脉冲序列, (N, T, class_num)
                    encoder_tl_loss: 编码器计算的迁移损失，实数
                    feature_tl_loss: 提取特征计算的迁移损失，实数
            else:  测试阶段
                Returns:
                    target_clf: 目标域分类神经元输出，0/1脉冲序列, (N, T, class_num)
        """
        # See note [TorchScript super()]
        if self.training:
            T, batch_size = source.shape[0:2]
            # source input encoder
            # print("source: ", source.shape)
            source = self.rgb_conv1(source)
            # print("source: ", source.shape)
            source_encoder_mem = self.rgb_bn1(source)
            # print("source: ", source_encoder_mem.shape)
            source_encoder_spike = self.rgb_sn1(source_encoder_mem)
            # print("source: ", source_encoder_spike.shape)
            source = self.maxpool(source_encoder_spike)
            # print("source: ", source.shape)

            # target input encoder
            # print("target: ", target.shape)
            target = self.dvs_conv1(target)
            # print("target: ", target.shape)
            target_encoder_mem = self.dvs_bn1(target)
            # print("target: ", target_encoder_mem.shape)
            target_encoder_spike = self.dvs_sn1(target_encoder_mem)
            # print("target: ", target_encoder_spike.shape)
            target = self.maxpool(target_encoder_spike)
            # print("target: ", target.shape)
            #
            # print("encoder_mem: ", source_encoder_mem.shape, target_encoder_mem.shape)
            # print("encoder_spike", source_encoder_spike.shape, target_encoder_spike.shape)

            if encoder_tl_loss_type == 'TCKA':
                # tl_loss = 1 - linear_CKA(source, target, "FLATTEN")
                encoder_tl_loss = 1 - temporal_linear_CKA(source_encoder_mem.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)),
                                                          target_encoder_mem.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)))
            elif encoder_tl_loss_type == 'CKA':
                encoder_tl_loss = 1 - linear_CKA(source.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)),
                                                 target.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)), "SUM")
            else:
                raise Exception(f"The value of encoder_tl_loss_type should in ['TCKA', 'CKA'], "
                                f"and your input is {encoder_tl_loss_type}")

            source = self.layer1(source)
            source = self.layer2(source)
            source = self.layer3(source)
            source_feature_spike, source_feature_mem = self.layer4(source)

            target = self.layer1(target)
            target = self.layer2(target)
            target = self.layer3(target)
            target_feature_spike, target_feature_mem = self.layer4(target)
            # print("feature_mem: ", source_feature_mem.shape, target_feature_mem.shape)
            # print("feature_spike: ", source_feature_spike.shape, target_feature_spike.shape)

            if feature_tl_loss_type == 'TMSE':
                feature_tl_loss = temporal_MSE(source_feature_mem.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)),
                                               target_feature_mem.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)))
            elif feature_tl_loss_type == 'MSE':
                feature_tl_loss = MSE(source_feature_spike.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)),
                                      target_feature_spike.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)), "SUM")
            elif feature_tl_loss_type == 'CKA':
                feature_tl_loss = 1 - linear_CKA(source_feature_spike.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)),
                                                 target_feature_spike.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)), "SUM")
            elif feature_tl_loss_type == 'TCKA':
                feature_tl_loss = 1 - temporal_linear_CKA(source_feature_mem.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)),
                                                          target_feature_mem.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)))
            elif feature_tl_loss_type == 'MMD':
                feature_tl_loss = MMD_loss(source_feature_spike.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)),
                                           target_feature_spike.permute(1, 0, 2, 3, 4).view((batch_size, T, -1)), "SUM")
            else:
                raise Exception(f"The value of feature_tl_loss_type should in ['TMSE', 'MSE', 'TCKA', 'CKA', 'MMD'], "
                                f"and your input is {feature_tl_loss_type}")

            source = self.avgpool(source_feature_spike)
            # print(source.shape)
            if self.avgpool.step_mode == 's':
                source = torch.flatten(source, 1)
            elif self.avgpool.step_mode == 'm':
                source = torch.flatten(source, 2)
            # print(source.shape)
            source_x_result = self.fc(source)
            # print(source_x_result.shape)

            target = self.avgpool(target_feature_spike)
            if self.avgpool.step_mode == 's':
                target = torch.flatten(target, 1)
            elif self.avgpool.step_mode == 'm':
                target = torch.flatten(target, 2)
            target_x_result = self.fc(target)

            return source_x_result, target_x_result, encoder_tl_loss, feature_tl_loss
        else:
            if target.shape[2] == 3:  # (T, N, 3, H, W)
                target = self.rgb_conv1(target)
                target = self.rgb_bn1(target)
                target = self.rgb_sn1(target)
            else:
                target = self.dvs_conv1(target)
                target = self.dvs_bn1(target)
                target = self.dvs_sn1(target)
            target = self.maxpool(target)

            target = self.layer1(target)
            target = self.layer2(target)
            target = self.layer3(target)
            target, _ = self.layer4(target)

            target = self.avgpool(target)
            if self.avgpool.step_mode == 's':
                target = torch.flatten(target, 1)
            elif self.avgpool.step_mode == 'm':
                target = torch.flatten(target, 2)
            target = self.fc(target)
            return target

    def forward(self, source_x, target_x, encoder_tl_loss_type='TCKA', feature_tl_loss_type='MSE'):
        return self._forward_impl(source_x, target_x, encoder_tl_loss_type, feature_tl_loss_type)


def _spiking_resnet(arch, block, layers, pretrained, progress, spiking_neuron, tl=False, **kwargs):
    if tl:
        model = TLSpikingResNet(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    else:
        model = SpikingResNet(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def spiking_resnet18(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    A spiking version of ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_resnet34(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module

    A spiking version of ResNet-34 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_resnet50(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module

    A spiking version of ResNet-50 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_resnet101(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module

    A spiking version of ResNet-101 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_resnet152(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module

    A spiking version of ResNet-152 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_resnext50_32x4d(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module

    A spiking version of ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_resnext101_32x8d(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module

    A spiking version of ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _spiking_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_wide_resnet50_2(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module

    A spiking version of Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, spiking_neuron, tl, **kwargs)


def spiking_wide_resnet101_2(pretrained=False, progress=True, spiking_neuron: callable=None, tl=False, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module

    A spiking version of Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, spiking_neuron, tl, **kwargs)

