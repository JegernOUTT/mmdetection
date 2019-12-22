import math
from typing import Optional, Sequence

import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.base_backbone import BaseBackbone, filter_by_out_idices
from mmdet.models.utils.activations import Mish


def conv_block(inp: int, oup: int, stride: int = 1, kernel: int = 3):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        Mish()
    )


class SpatialResampling(nn.Module):
    def __init__(self,
                 upsample_input_channels: int, upsample_factor: int,
                 downsample_input_channels: int, downsample_factor: int,
                 output_chanels: int,
                 scaling_factor: float = 0.5):
        super().__init__()
        assert 4 >= downsample_factor > 0 == downsample_factor % 2
        assert upsample_factor > 0 and upsample_factor % 2 == 0
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor

        self.before_upsample_conv = conv_block(
            upsample_input_channels,
            int(upsample_input_channels * scaling_factor), kernel=1)
        self.after_upsample_conv = conv_block(
            int(upsample_input_channels * scaling_factor),
            output_chanels, kernel=1)

        self.before_downsampling_conv = conv_block(
            downsample_input_channels,
            int(downsample_input_channels * scaling_factor), kernel=1)
        if self.downsample_factor >= 2:
            self.spatial_resampling_conv = conv_block(
                int(downsample_input_channels * scaling_factor),
                int(downsample_input_channels * scaling_factor), stride=2, kernel=3)
        if self.downsample_factor == 4:
            self.spatial_resampling_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.after_downsampling_conv = conv_block(
            int(downsample_input_channels * scaling_factor),
            output_chanels, kernel=1)

    def forward(self, x_up, x_down):
        x_up = self.before_upsample_conv(x_up)
        if self.upsample_factor > 1:
            x_up = F.interpolate(x_up, scale_factor=self.upsample_factor, mode='nearest')
        x_up = self.after_upsample_conv(x_up)

        x_down = self.before_downsampling_conv(x_down)
        if self.downsample_factor >= 2:
            x_down = self.spatial_resampling_conv(x_down)
        if self.downsample_factor == 4:
            x_down = self.spatial_resampling_maxpool(x_down)
        x_down = self.after_downsampling_conv(x_down)
        return x_up + x_down


class SpiNet(BaseBackbone):
    def __init__(self,
                 config: list,
                 input_channel: int,
                 last_channel: int,
                 out_indices: Optional[Sequence[int]]):
        super(SpiNet, self).__init__(out_indices)
        self.config = config
        self.last_channel = last_channel
        self.stem = stem(3, 32, 2)
        self.separable_conv = separable_conv(32, 16)

    @filter_by_out_idices
    def forward(self, x):
        skips = []
        x = self.stem(x)
        x = self.separable_conv(x)

        for block_name in self.block_names:
            if block_name.startswith('strided'):
                skips.append(x)
            x = getattr(self, block_name)(x)

        x = self.conv_before_pooling(x)
        skips.append(x)
        return skips

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SpiNet49(SpiNet):
    def __init__(self, input_channel: int, last_channel: int, out_indices: Optional[Sequence[int]]):
        config = [

        ]

        super().__init__(config=,
                         input_channel=input_channel,
                         last_channel=last_channel,
                         out_indices=out_indices)
