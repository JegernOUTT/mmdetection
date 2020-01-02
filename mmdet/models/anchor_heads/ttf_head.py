from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init

from mmdet.core import multi_apply, bbox_areas, force_fp32
from mmdet.core.anchor.guided_anchor_target import calc_region
from mmdet.models.losses import ct_focal_loss, diou_loss
from mmdet.models.utils import (build_norm_layer, bias_init_with_prob, ConvModule, AdvancedRFB)
from mmdet.ops import ModulatedDeformConvPack
from mmdet.ops.nms import simple_nms
from .anchor_head import AnchorHead
from ..registry import HEADS
from ..utils.activations import Mish


@HEADS.register_module
class TTFHead(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 head_conv=256,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 norm_cfg=dict(type='BN'),
                 build_neck: bool = True,
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 wh_weight=5.,
                 max_objs=128,
                 receptive_field_layer='deformable'):
        super(AnchorHead, self).__init__()
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.planes = planes
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.max_objs = max_objs
        self.receptive_field_layer = receptive_field_layer
        self.fp16_enabled = False
        self.build_neck = build_neck

        self.num_fg = num_classes - 1
        self.wh_planes = 4 if wh_agnostic else 4 * self.num_fg
        self.base_loc = None

        self.upsample_scale_factors = [1, 2, 4, 8]
        self.down_ratio = 4
        self.channel_norm_planes = 256
        self.final_conv_planes = 128

        for idx, plane in enumerate(inplanes):
            self.add_module(f'channel_norm_layer_{idx}',
                            nn.Sequential(nn.Conv2d(plane, self.channel_norm_planes, 1, padding=0, bias=False),
                                          build_norm_layer(norm_cfg, self.channel_norm_planes)[1],
                                          Mish()))

        for idx, scale_factor in enumerate(self.upsample_scale_factors):
            if scale_factor > 1:
                self.add_module(f'upsample_layer_{idx}', nn.Upsample(scale_factor=scale_factor, mode='nearest'))
            else:
                self.add_module(f'upsample_layer_{idx}', nn.Identity())

        if self.receptive_field_layer == 'deformable':
            final_conv = ModulatedDeformConvPack(self.channel_norm_planes, self.final_conv_planes, 3, stride=1,
                                                 padding=1, dilation=1, deformable_groups=1)
        elif self.receptive_field_layer == 'conv':
            final_conv = nn.Conv2d(self.channel_norm_planes, self.final_conv_planes,
                                   3, stride=1, padding=1, dilation=1, bias=False)
        elif self.receptive_field_layer == 'rfb':
            final_conv = AdvancedRFB(self.channel_norm_planes, self.final_conv_planes)
        else:
            assert False
        self.final_layer = nn.Sequential(final_conv,
                                         build_norm_layer(norm_cfg, self.final_conv_planes)[1],
                                         Mish())

        # heads
        self.wh = self.build_head(self.wh_planes, wh_head_conv_num, wh_conv)
        self.hm = self.build_head(self.num_fg, hm_head_conv_num)

    def build_head(self, out_channel, conv_num=1, head_conv_plane=None):
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        for i in range(conv_num):
            head_convs.append(ConvModule(self.final_conv_planes, head_conv_plane, 3, padding=1))

        inp = self.final_conv_planes if conv_num <= 0 else head_conv_plane
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """
        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """

        channel_norm_outputs = []
        for idx, x in enumerate(feats):
            out = getattr(self, f'channel_norm_layer_{idx}')(x)
            channel_norm_outputs.append(out)
        del feats

        upsample_outputs = []
        for idx, x in enumerate(channel_norm_outputs):
            out = getattr(self, f'upsample_layer_{idx}')(x)
            upsample_outputs.append(out)
        del channel_norm_outputs

        upsample_output = reduce(lambda x, y: x + y, upsample_outputs[1:], upsample_outputs[0])
        del upsample_outputs
        final_output = self.final_layer(upsample_output)

        hm = self.hm(final_output)
        wh = F.relu(self.wh(final_output)) * self.wh_offset_base

        return hm, wh

    def forward_export(self, feats):
        pred_heatmap, pred_wh = self.forward(feats)

        _, _, height, width = map(int, pred_heatmap.size())
        pred_heatmap = pred_heatmap.sigmoid()

        heat = simple_nms(pred_heatmap)

        base_step = self.down_ratio
        shifts_x = np.arange(0, width * base_step, base_step, dtype=np.float32)
        shifts_y = np.arange(0, height * base_step, base_step, dtype=np.float32)
        shift_y, shift_x = np.meshgrid(shifts_x, shifts_y)
        base_loc = torch.tensor(np.stack((shift_x, shift_y), axis=0)).to(pred_wh.device)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((base_loc - pred_wh[:, [0, 1]],
                                base_loc + pred_wh[:, [2, 3]]), dim=1)

        res = torch.cat([heat, pred_boxes], 1)

        return res

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   img_metas,
                   cfg,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score

        topk = getattr(cfg, 'max_per_img', 100)
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_fg, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for idx in range(bboxes.shape[0]):
            scores_per_img = scores[idx]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[idx][scores_keep]
            labels_per_img = clses[idx][scores_keep]
            img_shape = img_metas[idx]['pad_shape']
            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            if rescale:
                scale_factor = img_metas[idx]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)
            result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss = self.loss_calc(pred_heatmap, pred_wh, *all_targets)
        return {'losses/ttfnet_loss_heatmap': hm_loss, 'losses/ttfnet_loss_wh': wh_loss}

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        if len(gt_boxes) == 0:
            return heatmap, box_target, reg_weight

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.wh_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.wh_gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1,
                                                             use_round=False)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x / self.down_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.wh_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()

        return heatmap, box_target, reg_weight

    def target_generator(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, reg_weight = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()

            return heatmap, box_target, reg_weight

    def loss_calc(self,
                  pred_hm,
                  pred_wh,
                  heatmap,
                  box_target,
                  wh_weight):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        wh_loss = diou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) * self.wh_weight

        return hm_loss, wh_loss


class ShortcutConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(Mish())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
