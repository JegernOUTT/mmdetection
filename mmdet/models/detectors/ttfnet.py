from abc import ABC

from .augmix_detector import AbstractAugmixDetector
from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class TTFNet(SingleStageDetector, ABC):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TTFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)


@DETECTORS.register_module
class AugmixTTFNet(AbstractAugmixDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AugmixTTFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained)

    def get_objectness_tensor_by_bboxhead_output(self, x):
        return x[0]

