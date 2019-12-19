from typing import Sequence, Optional

import torch.nn as nn

from .base_backbone import BaseBackbone, filter_by_out_idices
from ..registry import BACKBONES
from ..utils import ConvModule
from ..utils.mish import Mish


@BACKBONES.register_module
class ConvnetLprVehicle(BaseBackbone):
    def __init__(self,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3, 4),
                 activation='relu'):
        super().__init__(out_indices=out_indices)
        self._norm_cfg = norm_cfg
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

    @filter_by_out_idices
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

        return skip_2, skip_4, skip_8, skip_16, x


@BACKBONES.register_module
class ConvnetLprPlate(BaseBackbone):
    def __init__(self,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 activation: str = 'relu',
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3)):
        super().__init__(out_indices)
        self._norm_cfg = norm_cfg
        self._activation = activation
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

    @filter_by_out_idices
    def forward(self, x):
        x = self._activation(self.conv_1(x))
        x = skip_2 = self._activation(self.conv_2(x))

        x = self._activation(self.conv_3(x))
        x = skip_4 = self._activation(self.conv_4(x))

        x = self._activation(self.conv_5(x))
        x = self._activation(self.conv_6(x))

        return skip_2, skip_4, x
