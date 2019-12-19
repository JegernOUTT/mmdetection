import logging
from typing import Sequence, Optional

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
