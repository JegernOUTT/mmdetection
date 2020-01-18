from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..registry import BACKBONES
from ..utils.activations import Mish, HardMish
from mmdet.models.backbones.base_backbone import BaseBackbone, filter_by_out_idices

__all__ = ['VoVNet27Slim', 'VoVNet39', 'VoVNet57']

model_urls = {
    'vovnet39': 'https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1',
    'vovnet57': 'https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1'
}


_ACTIVATIONS = {
    'mish': Mish,
    'hardmish': HardMish,
    'relu': nn.ReLU
}


def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1, activation='relu'):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         _ACTIVATIONS[activation.lower()]()),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, activation,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         _ACTIVATIONS[activation.lower()]()),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 activation,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i, activation=activation))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat', activation=activation)))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num,
                 activation):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name,
                        _OSA_module(in_ch,
                                    stage_ch,
                                    concat_ch,
                                    layer_per_block,
                                    module_name,
                                    activation=activation))
        for i in range(block_per_stage - 1):
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name,
                            _OSA_module(concat_ch,
                                        stage_ch,
                                        concat_ch,
                                        layer_per_block,
                                        module_name,
                                        activation=activation,
                                        identity=True))


class VoVNetBase(BaseBackbone):
    def __init__(self,
                 arch,
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 activation: str = 'relu',
                 pretrained: bool = True,
                 progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super(VoVNetBase, self).__init__(out_indices)

        if pretrained:
            from torch.hub import load_state_dict_from_url
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            self.load_state_dict(state_dict)

        # Stem module
        stem = conv3x3(3, 64, 'stem', '1', 2, activation=activation)
        stem += conv3x3(64, 64, 'stem', '2', 1, activation=activation)
        stem += conv3x3(64, 128, 'stem', '3', 2, activation=activation)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):  # num_stages
            name = f'stage{i + 2}'
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block,
                                       i + 2,
                                       activation))
        self._initialize_weights()

    @filter_by_out_idices
    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            skips.append(x)
        return skips

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@BACKBONES.register_module
class VoVNet57(VoVNetBase):
    r"""Constructs a VoVNet-57 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, pretrained: bool = False, progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4), activation='relu'):
        super().__init__(
            arch='vovnet57',
            config_stage_ch=[128, 160, 192, 224],
            config_concat_ch=[256, 512, 768, 1024],
            block_per_stage=[1, 1, 4, 3],
            layer_per_block=5,
            pretrained=pretrained,
            progress=progress,
            activation=activation,
            out_indices=out_indices)


@BACKBONES.register_module
class VoVNet39(VoVNetBase):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, pretrained: bool = False, progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4), activation='relu'):
        super().__init__(
            arch='vovnet39',
            config_stage_ch=[128, 160, 192, 224],
            config_concat_ch=[256, 512, 768, 1024],
            block_per_stage=[1, 1, 2, 2],
            layer_per_block=5,
            pretrained=pretrained,
            progress=progress,
            activation=activation,
            out_indices=out_indices)


@BACKBONES.register_module
class VoVNet27Slim(VoVNetBase):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, pretrained: bool = False, progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4), activation='relu'):
        super().__init__(
            arch='vovnet27_slim',
            config_stage_ch=[64, 80, 96, 112],
            config_concat_ch=[128, 256, 384, 512],
            block_per_stage=[1, 1, 1, 1],
            layer_per_block=5,
            pretrained=pretrained,
            progress=progress,
            activation=activation,
            out_indices=out_indices)
