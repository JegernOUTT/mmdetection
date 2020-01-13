import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        if 'debug' in self.train_cfg and self.train_cfg['debug']:
            self._debug_data_pipeline(img, img_metas, gt_bboxes, gt_labels)
        x = self.extract_feat(img)
        # print(list(map(lambda x_: x_.shape, x)))
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def forward_export(self, imgs):
        x = self.extract_feat(imgs)
        return self.bbox_head(x) \
            if getattr(self.bbox_head, 'forward_export', None) is None else self.bbox_head.forward_export(x)

    def _debug_data_pipeline(self,
                             img, img_metas,
                             gt_bboxes, gt_labels):
        import numpy as np
        from detector_utils import Object, bbox_from_xyxy, Size2D
        from detector_utils.utils.visualize import draw_bboxes, render_image, RenderMethod

        for img_idx, img in enumerate(img):
            numpy_img = img.detach().cpu().numpy()
            img_meta = img_metas[img_idx]
            mean, std = map(np.array, (img_meta['img_norm_cfg']['mean'], img_meta['img_norm_cfg']['std']))
            numpy_img = ((np.transpose(numpy_img, (1, 2, 0)) + mean) * std).astype(np.uint8)
            h, w = numpy_img.shape[:2]

            bboxes = gt_bboxes[img_idx].detach().cpu().numpy()
            labels = gt_labels[img_idx].detach().cpu().numpy()
            if len(bboxes) == 0:
                render_image(numpy_img, render_method=RenderMethod.OpenCV)
                continue

            objects = [
                Object(bbox=bbox_from_xyxy(bbox, image_size=Size2D(width=w, height=h)), category_id=class_idx - 1)
                for bbox, class_idx in zip(bboxes, labels)
            ]
            cat_idxes = range(max(labels))
            numpy_img = draw_bboxes(numpy_img, objects=objects, categories=dict(zip(cat_idxes, map(str, cat_idxes))))
            render_image(numpy_img, render_method=RenderMethod.OpenCV)
