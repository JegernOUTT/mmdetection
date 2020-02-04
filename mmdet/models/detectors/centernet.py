from .augmix_detector import AbstractAugmixDetector
from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class CenterNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)


@DETECTORS.register_module
class AugmixCenterNet(AbstractAugmixDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AugmixCenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                              test_cfg, pretrained)

    def get_objectness_tensor_by_bboxhead_output(self, x):
        return x[0][0][0],
