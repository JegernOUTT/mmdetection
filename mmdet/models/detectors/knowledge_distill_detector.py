import math
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_stage import SingleStageDetector
from ..registry import DETECTORS

__all__ = ['AbstractKnowledgeDistillationDetector']


class DCGanLoss:
    class Discriminator(nn.Module):
        def __init__(self, input_channels: int, ndf: int = 32):
            self._input_channels = input_channels
            self._ndf = ndf

            super(DCGanLoss.Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(self._input_channels, self._ndf, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self._ndf, self._ndf * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self._ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self._ndf * 2, self._ndf * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self._ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self._ndf * 4, self._ndf * 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self._ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self._ndf * 8, self._ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self._ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self._ndf * 8, 1),
            )
            self.weights_init()

        def forward(self, x):
            return self.main(x)

        def weights_init(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def __init__(self, input_channels: int, device):
        self._device = device
        self._criterion = nn.BCEWithLogitsLoss()
        self._lr = 1e-3
        self._discriminator = DCGanLoss.Discriminator(input_channels=input_channels).to(device)
        self._discriminator_opt = torch.optim.Adam(self._discriminator.parameters(), lr=self._lr)

    def __call__(self, teacher_feature, student_feature):
        self._discriminator.zero_grad()
        bs = teacher_feature.size(0)
        out_real = self._discriminator(teacher_feature).view(-1)
        err_real = self._criterion(out_real, torch.full((bs, ), 1, device=self._device))
        err_real.backward()

        out_fake = self._discriminator(student_feature.detach()).view(-1)
        err_fake = self._criterion(out_fake, torch.full((bs, ), 0, device=self._device))
        err_fake.backward()
        self._discriminator_opt.step()

        for param in self._discriminator.parameters():
            param.requires_grad = False
        output = self._discriminator(student_feature).view(-1)
        err_grads = self._criterion(output, torch.full((bs,), 1, device=self._device))
        for param in self._discriminator.parameters():
            param.requires_grad = True
        print(f'Err_real {err_real.mean()}, err_fake: {err_fake.mean()}, err_gen: {err_grads.mean()}')

        return err_grads


class KLDivLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, teacher_feature, student_feature):
        return F.kl_div(torch.clamp(student_feature, 1e-7, 1.).log(),
                        teacher_feature, reduction='batchmean')


class JSDivLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, teacher_feature, student_feature):
        p_mixture = torch.clamp((teacher_feature + student_feature) / 2., 1e-7, 1.).log()
        return F.kl_div(p_mixture, student_feature, reduction='batchmean') + \
               F.kl_div(p_mixture, teacher_feature, reduction='batchmean')


KD_LOSS_MAPPING = {
    'dcgan_loss': DCGanLoss,
    'kl_div': KLDivLoss,
    'js_div': JSDivLoss
}


@DETECTORS.register_module
class AbstractKnowledgeDistillationDetector(SingleStageDetector, ABC):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AbstractKnowledgeDistillationDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                                    test_cfg, pretrained)
        if train_cfg is None:
            return

        self._distillation_config = self.train_cfg.distillation
        self._backbone_levels = self._distillation_config.backbone_levels
        self._head_levels = self._distillation_config.head_levels
        self._losses_fn = dict()

    def _create_loss_fn(self, loss_name, feature):
        assert loss_name in KD_LOSS_MAPPING, f"Loss must be one of: {KD_LOSS_MAPPING.keys()}"
        ch = feature.size(1)
        class_type = KD_LOSS_MAPPING[loss_name]
        return class_type(input_channels=ch, device=feature.device)

    # noinspection PyMethodOverriding
    def forward_train(self,
                      img,
                      img_metas,
                      teacher_data,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        if 'debug' in self.train_cfg and self.train_cfg['debug']:
            self._debug_data_pipeline(img, img_metas, gt_bboxes, gt_labels)
        features = self.extract_feat(img)
        outs = self.bbox_head(features)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        for level in self._backbone_levels:
            teacher_feature, student_feature = teacher_data['features'][level.idx], features[level.idx]
            loss_log_name = f'feature_kd_loss_{level.idx}'
            if loss_log_name not in self._losses_fn:
                self._losses_fn[loss_log_name] = self._create_loss_fn(level.loss, student_feature)
            loss_fn = self._losses_fn[loss_log_name]

            if level.student_sigmoid:
                student_feature = torch.clamp(student_feature.sigmoid(), min=1e-4, max=1 - 1e-4)
            if level.teacher_sigmoid:
                teacher_feature = torch.clamp(teacher_feature.sigmoid(), min=1e-4, max=1 - 1e-4)

            losses[loss_log_name] = \
                level.coeff * loss_fn(student_feature=student_feature, teacher_feature=teacher_feature)

        for level in self._head_levels:
            teacher_feature, student_feature = teacher_data['outputs'][level.idx], outs[level.idx]
            loss_log_name = f'out_kd_loss_{level.idx}'
            if loss_log_name not in self._losses_fn:
                self._losses_fn[loss_log_name] = self._create_loss_fn(level.loss, student_feature)
            loss_fn = self._losses_fn[loss_log_name]

            if level.student_sigmoid:
                student_feature = torch.clamp(student_feature.sigmoid(), min=1e-4, max=1 - 1e-4)
            if level.teacher_sigmoid:
                teacher_feature = torch.clamp(teacher_feature.sigmoid(), min=1e-4, max=1 - 1e-4)

            losses[loss_log_name] = \
                level.coeff * loss_fn(student_feature=student_feature, teacher_feature=teacher_feature)

        return losses
