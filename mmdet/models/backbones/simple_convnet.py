from typing import Sequence, Optional

import torch.nn as nn

from ..registry import BACKBONES
from ..utils import build_conv_layer


@BACKBONES.register_module
class ConvnetLprVehicle(nn.Module):
    def __init__(self,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 activation: str = 'relu',
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3, 4)):
        super().__init__()
        self._norm_cfg = norm_cfg
        self._activation = activation
        self._out_indices = out_indices
        self._kwargs = dict(conv_cfg=None, norm_cfg=self._norm_cfg, activation=self._activation)

        # 2
        self.conv_1 = build_conv_layer(None, 3, 16, kernel_size=7, stride=2, padding=3, bias=False, **self._kwargs)

        # 4
        self.conv_2 = build_conv_layer(None, 16, 32, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_3 = build_conv_layer(None, 32, 32, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 8
        self.conv_4 = build_conv_layer(None, 32, 64, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_5 = build_conv_layer(None, 64, 128, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 16
        self.conv_6 = build_conv_layer(None, 128, 128, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_7 = build_conv_layer(None, 128, 256, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 32
        self.conv_8 = build_conv_layer(None, 256, 256, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_9 = build_conv_layer(None, 256, 512, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

    def _get_by_idxes(self, data):
        return tuple([
            data[idx]
            for idx in self._out_indices
        ])

    def forward(self, x):
        x = skip_2 = self.conv_1(x)

        x = self.conv_2(x)
        x = skip_4 = self.conv_3(x)

        x = self.conv_4(x)
        x = skip_8 = self.conv_5(x)

        x = self.conv_6(x)
        x = skip_16 = self.conv_7(x)

        x = self.conv_8(x)
        x = self.conv_9(x)

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
        self._kwargs = dict(conv_cfg=None, norm_cfg=self._norm_cfg, activation=self._activation)

        # 2
        self.conv_1 = build_conv_layer(None, 3, 16, kernel_size=7, stride=2, padding=3, bias=False, **self._kwargs)

        # 4
        self.conv_2 = build_conv_layer(None, 16, 32, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_3 = build_conv_layer(None, 32, 64, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

        # 8
        self.conv_4 = build_conv_layer(None, 64, 128, kernel_size=3, stride=1, padding=1, bias=False, **self._kwargs)
        self.conv_5 = build_conv_layer(None, 128, 128, kernel_size=3, stride=2, padding=1, bias=False, **self._kwargs)

    def _get_by_idxes(self, data):
        return tuple([
            data[idx]
            for idx in self._out_indices
        ])

    def forward(self, x):
        x = skip_2 = self.conv_1(x)

        x = self.conv_2(x)
        x = skip_4 = self.conv_3(x)

        x = self.conv_4(x)
        x = skip_8 = self.conv_5(x)

        return x if self._out_indices is None else self._get_by_idxes((skip_2, skip_4, skip_8, x))
