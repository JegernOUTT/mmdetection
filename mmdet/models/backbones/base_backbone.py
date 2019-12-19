import logging
from typing import Sequence, Optional

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

__all__ = ['filter_by_out_idices', 'BaseBackbone']


def filter_by_out_idices(forward_func):
    def _filter_func(self):
        outputs = forward_func(self)
        if self._out_indices is None:
            return outputs[-1]
        return tuple([
            outputs[idx]
            for idx in self._out_indices
        ])

    return _filter_func


class BaseBackbone(nn.Module):
    def __init__(self, out_indices: Optional[Sequence[int]] = (0, 1, 2, 3)):
        super().__init__()
        self._out_indices = out_indices

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)


class ClassifierPretrainWrapper(nn.Module):
    def __init__(self, backbone_module: nn.Module, num_classes: int):
        super().__init__()
        self.backbone_module = backbone_module
        self.classifier_layers = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)]
        self.classifier_layers = nn.Sequential(*self.last_layers)

    def save_backbone(self, path):
        torch.save({'state_dict': self.backbone_module.state_dict()}, path)

    def forward(self, x):
        x = self.backbone_module(x)
        return self.classifier_layers(x)
