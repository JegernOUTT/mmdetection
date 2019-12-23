from abc import abstractmethod, ABC

import torch
import torch.nn.functional as F

from .single_stage import SingleStageDetector
from ..registry import DETECTORS

__all__ = ['AbstractAugmixDetector']


@DETECTORS.register_module
class AbstractAugmixDetector(SingleStageDetector, ABC):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AbstractAugmixDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                     test_cfg, pretrained)

    @abstractmethod
    def get_objectness_tensor_by_bboxhead_output(self, x):
        pass

    # noinspection PyMethodOverriding
    def forward_train(self,
                      img,
                      img_augmix_0,
                      img_augmix_1,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        all_losses, objectness_outs = [], []
        for img in [img, img_augmix_0, img_augmix_1]:
            if 'debug' in self.train_cfg and self.train_cfg['debug']:
                self._debug_data_pipeline(img, img_metas, gt_bboxes, gt_labels)
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            objectness_outs.append(self.get_objectness_tensor_by_bboxhead_output(outs))
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
            losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            all_losses.append(losses)

        all_losses = {
            loss_name: torch.stack([l[loss_name] for l in all_losses], dim=0).mean(dim=0)
            for loss_name in all_losses[0].keys()
        }

        all_losses['js_loss'] = self.js_loss(*objectness_outs)
        return all_losses

    def js_loss(self, logits_clean: torch.Tensor, logits_aug1: torch.Tensor, logits_aug2: torch.Tensor):
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((logits_clean + logits_aug1 + logits_aug2) / 3., 1e-7, 1).log()
        return 12. * (F.kl_div(p_mixture, logits_clean, reduction='batchmean') +
                      F.kl_div(p_mixture, logits_aug1, reduction='batchmean') +
                      F.kl_div(p_mixture, logits_aug2, reduction='batchmean')) / 3.
