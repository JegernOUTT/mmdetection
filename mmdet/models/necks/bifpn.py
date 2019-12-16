import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet.core import auto_fp16

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class BIFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,
                                                      levels=self.backbone_end_level - self.start_level,
                                                      conv_cfg=conv_cfg,
                                                      norm_cfg=norm_cfg,
                                                      activation=activation))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNWeightedAdd(nn.Module):
    def __init__(self, inputs_count, epsilon=1e-4, init=0.5):
        super(BiFPNWeightedAdd, self).__init__()
        self._epsilon = epsilon
        self._inputs_count = inputs_count
        self.w = nn.Parameter(torch.zeros((self._inputs_count,), dtype=torch.float32).fill_(init))
        self.relu = nn.ReLU()

    def forward(self, inputs):
        w = self.relu(self.w)
        weighted_sum = torch.stack([w[idx] * inputs[idx] for idx in range(len(inputs))], dim=0).sum(dim=0)
        return weighted_sum / (w.sum(dim=0) + self._epsilon)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BiFPNModule, self).__init__()
        self._activation_type = activation
        self._levels = levels
        self._top_bottom_fpn_convs = nn.ModuleList()
        self._top_bottom_weighted_sum = nn.ModuleList()
        self._bottom_top_fpn_convs = nn.ModuleList()
        self._bottom_top_weighted_sum = nn.ModuleList()
        self._base_conv_params = {
            'in_channels': channels,
            'out_channels': channels,
            'groups': channels,
            'conv_cfg': conv_cfg,
            'norm_cfg': norm_cfg,
            'activation': self._activation_type,
            'inplace': False,
        }

        for _ in range(self._levels - 1):  # 1, 2, 3
            self._top_bottom_fpn_convs.append(nn.Sequential(
                ConvModule(kernel_size=3, **self._base_conv_params),
                ConvModule(kernel_size=1, padding=1, **self._base_conv_params)
            ))
            self._top_bottom_weighted_sum.append(BiFPNWeightedAdd(inputs_count=2))

        for idx in range(self._levels - 1):  # 1, 2, 3
            self._bottom_top_fpn_convs.append(nn.Sequential(
                ConvModule(kernel_size=3, **self._base_conv_params),
                ConvModule(kernel_size=1, padding=1, **self._base_conv_params)
            ))
            self._bottom_top_weighted_sum.append(BiFPNWeightedAdd(inputs_count=3 if idx != 3 else 2))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == self._levels

        # build top-down
        pathtd = [t for t in inputs]
        for conv_idx, i in enumerate(range(self._levels - 1, 0, -1)):
            pathtd[i - 1] = self._top_bottom_weighted_sum[conv_idx]([
                pathtd[i - 1],
                F.interpolate(pathtd[i], scale_factor=2)])
            pathtd[i - 1] = self._top_bottom_fpn_convs[conv_idx](pathtd[i - 1])

        # build down-top
        for conv_idx, i in enumerate(range(0, self._levels - 2, 1)):
            pathtd[i + 1] = self._bottom_top_weighted_sum[conv_idx]([
                pathtd[i + 1],
                F.max_pool2d(pathtd[i], kernel_size=2),
                inputs[i + 1]])
            pathtd[i + 1] = self._bottom_top_fpn_convs[conv_idx](pathtd[i + 1])

        pathtd[self._levels - 1] = self._bottom_top_weighted_sum[-1]([
            pathtd[self._levels - 1],
            F.max_pool2d(pathtd[self._levels - 2], kernel_size=2)])
        pathtd[self._levels - 1] = self._bottom_top_fpn_convs[-1](pathtd[self._levels - 1])
        return pathtd
