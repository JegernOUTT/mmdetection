import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..registry import BACKBONES
from mmdet.models.backbones.base_backbone import BaseBackbone, filter_by_out_idices
from mmdet.models.utils.activations import Swish, Mish

__all__ = ['MixNetS', 'MixNetM', 'MixNetL']

NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
    'Mish': Mish()
}


def _round_channels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


def _split_channels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


def conv3x3_bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )


def conv1x1_bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y


class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_in_channels = _split_channels(in_channels, self.num_groups)
        self.split_out_channels = _split_channels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _split_channels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class MixNetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=[3],
            expand_ksize=[1],
            project_ksize=[1],
            stride=1,
            expand_ratio=1,
            non_linear='ReLU',
            se_ratio=0.0):
        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                GroupedConv2d(in_channels, expand_channels, expand_ksize),
                nn.BatchNorm2d(expand_channels),
                NON_LINEARITY[non_linear]
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm2d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            GroupedConv2d(expand_channels, out_channels, project_ksize),
            nn.BatchNorm2d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BaseMixnet(BaseBackbone):
    def __init__(self,
                 config: list,
                 stem_channels: int,
                 depth_multiplier: float = 1.0,
                 feature_size: int = 1536,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super(BaseMixnet, self).__init__(out_indices)

        # depth multiplier
        if depth_multiplier != 1.0:
            stem_channels = _round_channels(stem_channels * depth_multiplier)

            for i, conf in enumerate(config):
                conf_ls = list(conf)
                conf_ls[0] = _round_channels(conf_ls[0] * depth_multiplier)
                conf_ls[1] = _round_channels(conf_ls[1] * depth_multiplier)
                config[i] = tuple(conf_ls)

        # stem convolution
        self.stem_conv = conv3x3_bn(3, stem_channels, 2)

        # building MixNet blocks
        self.block_names = []
        for idx, (in_channels, out_channels, kernel_size, expand_ksize,
                  project_ksize, stride, expand_ratio, non_linear, se_ratio) in enumerate(config):
            block_name = f'block_{idx}' if stride == 1 else f'strided_block_{idx}'
            self.add_module(block_name, MixNetBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                expand_ksize=expand_ksize,
                project_ksize=project_ksize,
                stride=stride,
                expand_ratio=expand_ratio,
                non_linear=non_linear,
                se_ratio=se_ratio
            ))
            self.block_names.append(block_name)

        # last several layers
        self.head_conv = conv1x1_bn(config[-1][1], feature_size)
        self._initialize_weights()

    @filter_by_out_idices
    def forward(self, x):
        skips = []
        x = self.stem_conv(x)

        for block_name in self.block_names:
            if block_name.startswith('strided'):
                skips.append(x)
            x = getattr(self, block_name)(x)

        x = self.head_conv(x)
        skips.append(x)
        return skips

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


@BACKBONES.register_module
class MixNetS(BaseMixnet):
    # [in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio]
    mixnet_s = [(16, 16, [3], [1], [1], 1, 1, 'ReLU', 0.0),
                (16, 24, [3], [1, 1], [1, 1], 2, 6, 'ReLU', 0.0),
                (24, 24, [3], [1, 1], [1, 1], 1, 3, 'ReLU', 0.0),
                (24, 40, [3, 5, 7], [1], [1], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], [1], [1, 1], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5], [1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5], [1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 120, [3, 5, 7], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9, 11], [1], [1], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5)]

    def __init__(self,
                 feature_size: int = 1536,
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3)):
        super().__init__(config=self.mixnet_s, stem_channels=16,
                         feature_size=feature_size, out_indices=out_indices)


@BACKBONES.register_module
class MixNetM(BaseMixnet):
    # [in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio]
    mixnet_m = [(24, 24, [3], [1], [1], 1, 1, 'ReLU', 0.0),
                (24, 32, [3, 5, 7], [1, 1], [1, 1], 2, 6, 'ReLU', 0.0),
                (32, 32, [3], [1, 1], [1, 1], 1, 3, 'ReLU', 0.0),
                (32, 40, [3, 5, 7, 9], [1], [1], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], [1], [1], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 120, [3], [1], [1], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9], [1], [1], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5)]

    def __init__(self,
                 feature_size: int = 1536,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super().__init__(config=self.mixnet_m, stem_channels=24,
                         feature_size=feature_size, out_indices=out_indices)


@BACKBONES.register_module
class MixNetL(BaseMixnet):
    # [in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio]
    mixnet_m = [(24, 24, [3], [1], [1], 1, 1, 'ReLU', 0.0),
                (24, 32, [3, 5, 7], [1, 1], [1, 1], 2, 6, 'ReLU', 0.0),
                (32, 32, [3], [1, 1], [1, 1], 1, 3, 'ReLU', 0.0),
                (32, 40, [3, 5, 7, 9], [1], [1], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], [1], [1], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 120, [3], [1], [1], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9], [1], [1], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5)]

    def __init__(self,
                 feature_size: int = 1536,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super().__init__(config=self.mixnet_m, stem_channels=24, depth_multiplier=1.3,
                         feature_size=feature_size, out_indices=out_indices)
