import logging
from typing import Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES
from ..utils import ConvModule


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


@BACKBONES.register_module
class ConvnetLprVehicle(nn.Module):
    def __init__(self,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3, 4),
                 activation='relu'):
        super().__init__()
        self._norm_cfg = norm_cfg
        self._out_indices = out_indices
        self._kwargs = dict(conv_cfg=None, norm_cfg=self._norm_cfg, activation=None)
        if activation == 'relu':
            self._activation = nn.ReLU(inplace=True)
        elif activation == 'mish':
            self._activation = Mish()
        else:
            assert False

        # 2
        self.conv_1 = ConvModule(3, 16, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_2 = ConvModule(16, 16, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 4
        self.conv_3 = ConvModule(16, 32, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_4 = ConvModule(32, 32, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 8
        self.conv_5 = ConvModule(32, 64, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_6 = ConvModule(64, 64, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 16
        self.conv_7 = ConvModule(64, 128, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_8 = ConvModule(128, 128, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 32
        self.conv_9 = ConvModule(128, 256, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_10 = ConvModule(256, 256, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

    def _get_by_idxes(self, data):
        return tuple([
            data[idx]
            for idx in self._out_indices
        ])

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        x = self._activation(self.conv_1(x))
        x = skip_2 = self._activation(self.conv_2(x))

        x = self._activation(self.conv_3(x))
        x = skip_4 = self._activation(self.conv_4(x))

        x = self._activation(self.conv_5(x))
        x = skip_8 = self._activation(self.conv_6(x))

        x = self._activation(self.conv_7(x))
        x = skip_16 = self._activation(self.conv_8(x))

        x = self._activation(self.conv_9(x))
        x = self._activation(self.conv_10(x))

        return x if self._out_indices is None else self._get_by_idxes((skip_2, skip_4, skip_8, skip_16, x))


@BACKBONES.register_module
class ConvnetLprPlate(nn.Module):
    def __init__(self,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 activation: str = 'relu',
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3)):
        super().__init__()
        self._norm_cfg = norm_cfg
        self._activation = activation
        self._out_indices = out_indices
        self._kwargs = dict(conv_cfg=None, norm_cfg=self._norm_cfg, activation=None)
        if activation == 'relu':
            self._activation = nn.ReLU(inplace=True)
        elif activation == 'mish':
            self._activation = Mish()
        else:
            assert False
        # 2
        self.conv_1 = ConvModule(3, 16, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_2 = ConvModule(16, 16, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 4
        self.conv_3 = ConvModule(16, 32, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_4 = ConvModule(32, 32, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 8
        self.conv_5 = ConvModule(32, 64, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_6 = ConvModule(64, 64, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

    def _get_by_idxes(self, data):
        return tuple([
            data[idx]
            for idx in self._out_indices
        ])

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        x = self._activation(self.conv_1(x))
        x = skip_2 = self._activation(self.conv_2(x))

        x = self._activation(self.conv_3(x))
        x = skip_4 = self._activation(self.conv_4(x))

        x = self._activation(self.conv_5(x))
        x = self._activation(self.conv_6(x))

        return x if self._out_indices is None else self._get_by_idxes((skip_2, skip_4, x))
