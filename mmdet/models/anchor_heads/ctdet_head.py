import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
from mmdet.core import auto_fp16
from mmdet.core import multi_apply, force_fp32
from mmdet.models.utils import (build_norm_layer, bias_init_with_prob, ConvModule)
from mmdet.ops import ModulatedDeformConvPack

from .anchor_head import AnchorHead
from ..registry import HEADS


def _tranpose_and_gather_feat(feat, ind):
    def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask].view(-1, dim)
        return feat

    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


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
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def forward(self, out, target):
        return self._neg_loss(out, target)


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        # mask = mask.unsqueeze(2).expand_as(pred).float()
        mask = mask.expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


@HEADS.register_module
class CenternetDetectionHead(AnchorHead):
    def __init__(self,
                 require_upsampling=True,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 base_down_ratio=32,
                 hm_head_conv=128,
                 hm_offset_heads_conv=128,
                 wh_heads_conv=128,
                 with_deformable=True,
                 hm_head_conv_num=2,
                 hm_offset_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 shortcut_kernel=3,
                 norm_cfg=dict(type='BN'),
                 shortcut_cfg=(1, 2, 3),
                 num_stacks=1,  # It can be > 1 in backbones such as hourglass
                 ellipse_gaussian=True,
                 exp_wh=True,
                 hm_weight=1.,
                 hm_offset_weight=1.,
                 wh_weight=0.1,
                 max_objs=512):
        super(AnchorHead, self).__init__()
        assert len(planes) in [2, 3, 4]
        shortcut_num = min(len(inplanes) - 1, len(planes))
        assert shortcut_num == len(shortcut_cfg)

        self._require_upsampling = require_upsampling
        self._inplanes = inplanes
        self._planes = planes
        self._base_down_ratio = base_down_ratio
        self._with_deformable = with_deformable

        self._hm_head_conv = hm_head_conv
        self._hm_offset_heads_conv = hm_offset_heads_conv
        self._wh_heads_conv = wh_heads_conv

        self._hm_head_conv_num = hm_head_conv_num
        self._hm_offset_head_conv_num = hm_offset_head_conv_num
        self._wh_head_conv_num = wh_head_conv_num

        self._max_objs = max_objs
        self._num_stacks = num_stacks

        self._hm_loss = FocalLoss()
        self._hm_offset_loss = RegL1Loss()
        self._wh_loss = RegL1Loss()

        self._hm_weight = hm_weight
        self._hm_offset_weight = hm_offset_weight
        self._wh_weight = wh_weight

        self.num_classes = num_classes
        self._num_fg = self.num_classes - 1
        self._ellipse_gaussian = ellipse_gaussian
        self._exp_wh = exp_wh

        self._output_stride = base_down_ratio // 2 ** len(planes)
        self._hm_offset_planes, self._wh_planes = 2, 2

        if self._require_upsampling:
            # repeat upsampling n times. 32x to 4x by default.
            self.upsample_layers = nn.ModuleList([
                self._build_upsample(inplanes[-1], planes[0], norm_cfg=norm_cfg),
                self._build_upsample(planes[0], planes[1], norm_cfg=norm_cfg)
            ])
            for i in range(2, len(planes)):
                self.upsample_layers.append(
                    self._build_upsample(planes[i - 1], planes[i], norm_cfg=norm_cfg))

            padding = (shortcut_kernel - 1) // 2
            self.shortcut_layers = self._build_shortcut(
                inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
                kernel_size=shortcut_kernel, padding=padding)

        self.hm = self._build_head(self._num_fg, self._hm_head_conv_num, self._hm_head_conv)
        self.hm_offset = self._build_head(self._wh_planes, self._hm_offset_head_conv_num, self._hm_offset_heads_conv)
        self.wh = self._build_head(self._wh_planes, self._wh_head_conv_num, self._wh_heads_conv)

    def init_weights(self):
        if self._require_upsampling:
            for _, m in self.shortcut_layers.named_modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for _, m in self.upsample_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for _, m in self.hm_offset.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

    @auto_fp16()
    def forward(self, feats):
        """
        Args:
            feats: list(tensor).
        Returns:
            hm: tensor, (batch, class_num, h, w).
            hm_offset: tensor, (batch, 2, h, w).
            wh: tensor, (batch, 2, h, w)
        """
        if self._require_upsampling:
            x = feats[-1]
            for i, upsample_layer in enumerate(self.upsample_layers):
                x = upsample_layer(x)
                if i < len(self.shortcut_layers):
                    shortcut = self.shortcut_layers[i](feats[-i - 2])
                    x = x + shortcut

            hm = self.hm(x)
            hm_offset = self.hm_offset(x)
            wh = self.wh(x)
            return [[hm, hm_offset, wh]],
        else:
            if isinstance(feats, torch.Tensor):
                x = [feats]
            else:
                x = feats
                assert isinstance(feats, list)
                assert len(feats) == self._num_stacks

            out = []
            for stack_idx in range(self._num_stacks):
                current_feat = x[stack_idx]
                hm = self.hm(current_feat)
                hm_offset = self.hm_offset(current_feat)
                wh = self.wh(current_feat)
                out.append([hm, hm_offset, wh])
            return out,

    @force_fp32(apply_to=('outputs',))
    def loss(self,
             outputs,
             bboxes_gt, labels_gt,
             image_metas, cfg, gt_bboxes_ignore=None):
        pad_shape = image_metas[0]['pad_shape']
        device = outputs[0][0].device
        targets = self._create_targets(bboxes_gt, labels_gt, pad_shape=pad_shape, device=device)
        if 'debug' in cfg and cfg['debug']:
            self._debug_data(*targets[:3], image_metas, cfg)
        hm_loss, hm_offset_loss, wh_loss = self._loss_calc(outputs, *targets)
        return {'losses/hm_loss': hm_loss, 'losses/hm_offset_loss': hm_offset_loss, 'losses/wh_loss': wh_loss}

    @force_fp32(apply_to=('outputs',))
    def get_bboxes(self, outputs, img_metas, cfg, rescale=False, top_k=100):
        def _nms(hm, kernel=3):
            pad = (kernel - 1) // 2
            hmax = nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == hm).float()
            return hm * keep

        def _topk(scores, topk):
            batch, cat, height, width = scores.size()

            # both are (batch, 80, topk)
            topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

            topk_inds = topk_inds % (height * width)
            topk_ys = (topk_inds / width).int().float()
            topk_xs = (topk_inds % width).int().float()

            # both are (batch, topk). select topk from 80 * topk
            topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
            topk_clses = (topk_ind / topk).int()
            topk_ind = topk_ind.unsqueeze(2)
            topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
            topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
            topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

            return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

        hm, hm_offset, wh = outputs[-1]
        batch, cat, height, width = hm.size()
        hm = torch.sigmoid(hm)
        hm = _nms(hm)
        if self._exp_wh:
            wh = torch.exp(wh)
        scores, inds, classes, ys, xs = _topk(hm, topk=top_k)

        # hm_offset
        hm_offset = hm_offset.permute(0, 2, 3, 1).contiguous()
        hm_offset = hm_offset.view(hm_offset.size(0), -1, hm_offset.size(3))
        hm_offset_inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), hm_offset.size(2))
        hm_offset = hm_offset.gather(1, hm_offset_inds).view(batch, top_k, 2)

        # wh
        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        wh_inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, wh_inds).view(batch, top_k, 2)

        xs = xs.view(batch, top_k, 1) + hm_offset[:, :, [0]]
        ys = ys.view(batch, top_k, 1) + hm_offset[:, :, [1]]

        classes = classes.view(batch, top_k, 1).float()
        scores = scores.view(batch, top_k, 1)
        bboxes = torch.cat([xs - wh[..., [0]] / 2, ys - wh[..., [1]] / 2,
                            xs + wh[..., [0]] / 2, ys + wh[..., [1]] / 2], dim=2)
        bboxes *= self._output_stride

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for idx in range(bboxes.shape[0]):
            scores_per_img = scores[idx]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[idx][scores_keep]
            labels_per_img = classes[idx][scores_keep]
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

    def _build_shortcut(self,
                        inplanes,
                        planes,
                        shortcut_cfg,
                        kernel_size=3,
                        padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def _build_upsample(self, inplanes, planes, norm_cfg=None):
        if self._with_deformable:
            mdcn = ModulatedDeformConvPack(inplanes, planes, 3, stride=1,
                                           padding=1, dilation=1, deformable_groups=1)
        else:
            mdcn = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, dilation=1)
        up = nn.Upsample(scale_factor=2, mode='nearest')

        layers = [mdcn]
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)

        return nn.Sequential(*layers)

    def _build_head(self, out_channel, conv_num=1, head_conv_plane=None):
        head_convs = []
        for i in range(conv_num):
            inp = self._planes[-1] if i == 0 else head_conv_plane
            head_convs.append(ConvModule(inp, head_conv_plane, 3, padding=1))

        inp = self._planes[-1] if conv_num <= 0 else head_conv_plane
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    @force_fp32(apply_to=('hm', 'hm_offset', 'wh'))
    def _loss_calc(self,
                   outputs,
                   hm_gt, hm_offset_gt, wh_gt, ind, reg_mask):
        assert len(outputs) == self._num_stacks

        hm_loss, hm_offset_loss, wh_loss = 0, 0, 0
        for stack_idx in range(self._num_stacks):
            hm, hm_offset, wh = outputs[stack_idx]

            hm = torch.clamp(hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
            hm_loss += self._hm_loss(hm, hm_gt) / self._num_stacks

            if self._hm_offset_weight > 0:
                hm_offset_loss += self._hm_offset_loss(hm_offset,
                                                       reg_mask, ind,
                                                       hm_offset_gt) / self._num_stacks

            if self._wh_weight > 0:
                wh_loss += self._wh_loss(wh,
                                         reg_mask, ind,
                                         wh_gt) / self._num_stacks

        return self._hm_weight * hm_loss, \
               self._hm_offset_weight * hm_offset_loss, \
               self._wh_weight * wh_loss

    def _create_targets(self, bboxes_gt, labels_gt, pad_shape, device):
        feature_h, feature_w = map(lambda x: int(x / self._output_stride), pad_shape[:2])

        with torch.no_grad():
            hm, hm_offset, wh, ind, reg_mask = multi_apply(
                self._create_target,
                bboxes_gt,
                labels_gt,
                feature_hw=(feature_h, feature_w),
                device=device
            )
            hm, hm_offset, wh, ind, reg_mask = [torch.stack(t, dim=0).detach()
                                                for t in (hm, hm_offset, wh, ind, reg_mask)]

        return hm, hm_offset, wh, ind, reg_mask

    def _create_target(self, bboxes_gt, labels_gt, feature_hw, device):
        feat_h, feat_w = feature_hw

        bboxes_gt = bboxes_gt.detach().cpu().numpy()
        labels_gt = labels_gt.detach().cpu().numpy()

        hm = np.zeros((self._num_fg, feat_h, feat_w), dtype=np.float32)
        hm_offset = np.zeros((self._max_objs, 2), dtype=np.float32)
        wh = np.zeros((self._max_objs, 2), dtype=np.float32)
        ind = np.zeros((self._max_objs,), dtype=np.int64)
        reg_mask = np.zeros((self._max_objs,), dtype=np.uint8)

        # Sort by area, large bboxes are in front
        if len(bboxes_gt) > 0:
            bboxes_gt = bboxes_gt[:self._max_objs]
            indices = np.argsort(-(bboxes_gt[:, 2] - bboxes_gt[:, 0]) * (bboxes_gt[:, 3] - bboxes_gt[:, 1]))
            for bbox_idx in indices:
                x1, y1, x2, y2 = bboxes_gt[bbox_idx]
                x, y, w, h = (x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1
                x, y, w, h = map(lambda x: x / self._output_stride, (x, y, w, h))

                cat_id = labels_gt[bbox_idx] - 1
                x_idx, y_idx = map(int, [x, y])
                x_offset, y_offset = x - x_idx, y - y_idx

                if not self._ellipse_gaussian:
                    radius = self._calc_gaussian_radius((x, y, x + w, y + h))
                    radius = max(0, int(radius))

                    diameter = 2 * radius + 1
                    gaussian = self._get_gaussian_2d((diameter, diameter), sigma=diameter / 6)

                    left, right = min(x_idx, radius), min(feat_w - x_idx, radius + 1)
                    top, bottom = min(y_idx, radius), min(feat_h - y_idx, radius + 1)
                    w_radius, h_radius = radius, radius
                else:
                    w_radius, h_radius = self._calc_ellipse_gaussian_radiuses((x, y, x + w, y + h))
                    w_radius, h_radius = max(0, int(w_radius)), max(0, int(h_radius))

                    diameter_h, diameter_w = 2 * h_radius + 1, 2 * w_radius + 1
                    sigma_x = diameter_w / 6
                    sigma_y = diameter_h / 6
                    gaussian = self._get_ellipse_gaussian_2d((diameter_w, diameter_h),
                                                             sigma_x=sigma_x, sigma_y=sigma_y)

                    left, right = min(x_idx, w_radius), min(feat_w - x_idx, w_radius + 1)
                    top, bottom = min(y_idx, h_radius), min(feat_h - y_idx, h_radius + 1)

                masked_hm = hm[cat_id, y_idx - top:y_idx + bottom, x_idx - left:x_idx + right]
                masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
                masked_hm = np.maximum(masked_hm, masked_gaussian)

                hm[cat_id, y_idx - top:y_idx + bottom, x_idx - left:x_idx + right] = masked_hm
                hm_offset[bbox_idx] = [x_offset, y_offset]
                if self._exp_wh:
                    eps = np.finfo(np.float32).eps
                    wh[bbox_idx] = np.log([w + eps, h + eps])
                else:
                    wh[bbox_idx] = [w, h]

                ind[bbox_idx] = y_idx * feat_w + x_idx
                reg_mask[bbox_idx] = 1

        hm, hm_offset, wh, ind, reg_mask = map(lambda x: torch.from_numpy(x).to(device=device),
                                               (hm, hm_offset, wh, ind[..., np.newaxis], reg_mask[..., np.newaxis]))
        return hm, hm_offset, wh, ind, reg_mask

    @staticmethod
    def _calc_ellipse_gaussian_radiuses(xyxy, alpha=0.54):
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        return w / 2. * alpha, h / 2. * alpha

    @staticmethod
    def _calc_gaussian_radius(xyxy, min_overlap=0.7):
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1

        a1 = 1
        b1 = (h + w)
        c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (h + w)
        c2 = (1 - min_overlap) * w * h
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (h + w)
        c3 = (min_overlap - 1) * w * h
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        return min(r1, r2, r3)

    @staticmethod
    def _get_gaussian_2d(shape, sigma=1.):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h.astype(np.float32)

    @staticmethod
    def _get_ellipse_gaussian_2d(shape, sigma_x=1., sigma_y=1.):
        n, m = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h.astype(np.float32)

    def _debug_data(self, hm, hm_offset, wh, img_metas, cfg):
        import cv2

        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        for batch_idx in range(hm.size(0)):
            img = cv2.imread(img_metas[batch_idx]['filename'])
            cv2.imshow('image', cv2.resize(img, (160, 128)))
            hm_cpu = hm.detach().cpu().numpy()
            for idx, c in enumerate(range(self._num_fg)):
                cv2.imshow(f'xy_heatmap_{c}', (hm_cpu[batch_idx, idx, ...][..., np.newaxis] * 255).astype(np.uint8))
            cv2.waitKey(0)
