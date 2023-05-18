#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES, build_loss
from mmrotate.core import build_bbox_coder
from mmdet.core.anchor.point_generator import MlvlPointGenerator


@ROTATED_LOSSES.register_module()
class RotatedDTLoss(nn.Module):
    def __init__(self, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedDTLoss, self).__init__()
        self.cls_channels = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
            self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
            self.bbox_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
        self.loss_type = loss_type

    def convert_shape(self, logits):
        cls_scores, bbox_preds, angle_preds, centernesses = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]
        cls_scores = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_channels) for x in cls_scores
        ], dim=1).view(-1, self.cls_channels)
        bbox_preds = torch.cat([
            torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(batch_size, -1, 5) for x, y in
            zip(bbox_preds, angle_preds)
        ], dim=1).view(-1, 5)
        centernesses = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in centernesses
        ], dim=1).view(-1, 1)
        return cls_scores, bbox_preds, centernesses

    def forward(self, teacher_logits, student_logits, ratio=0.01, img_metas=None,
                **kwargs):
        """

        Args:
            teacher_logits (Tuple): logits of teacher, with keys:
                "cls_score": list[Tensor], Box scores for each scale level, each is a 4D-tensor,
                the channel number is num_points * num_classes.
                "bbox_pred": list[Tensor], Box energies / deltas for each scale level, each is a 4D-tensor,
                the channel number is num_points * 4.
                "angle_pred": list[Tensor], Box angle for each scale level, each is a 4D-tensor,
                the channel number is num_points * 1.
                "centerness": list[Tensor], centerness for each scale level, each is a 4D-tensor,
                the channel number is num_points * 1.
            student_logits (Tuple): logits of student, same as teacher.
            ratio (float): sampling ratio for loss calculation
            img_metas (Optional | Dict): img metas

        Returns:

        """

        t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape(student_logits)

        with torch.no_grad():
            # Region Selection
            count_num = int(t_cls_scores.size(0) * ratio)
            teacher_probs = t_cls_scores.sigmoid()
            max_vals = torch.max(teacher_probs, 1)[0]
            sorted_vals, sorted_inds = torch.topk(max_vals, t_cls_scores.size(0))
            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        loss_cls = QFLv2(
            s_cls_scores.sigmoid(),
            t_cls_scores.sigmoid(),
            weight=mask,
            reduction="sum",
        ) / fg_num
        if self.bbox_loss_type == 'l1':
            loss_bbox = (self.bbox_loss(
                s_bbox_preds[b_mask],
                t_bbox_preds[b_mask],
            ) * t_centernesses.sigmoid()[b_mask]).mean()
        else:
            all_level_points = self.prior_generator.grid_priors(
                [featmap.size()[-2:] for featmap in teacher_logits[0]],
                dtype=s_bbox_preds.dtype,
                device=s_bbox_preds.device)
            flatten_points = torch.cat(
                [points.repeat(len(teacher_logits[0][0]), 1) for points in all_level_points])
            s_bbox_preds = self.bbox_coder.decode(flatten_points, s_bbox_preds)[b_mask]
            t_bbox_preds = self.bbox_coder.decode(flatten_points, t_bbox_preds)[b_mask]
            loss_bbox = self.bbox_loss(
                s_bbox_preds,
                t_bbox_preds,
            ) * t_centernesses.sigmoid()[b_mask]
            nan_indexes = ~torch.isnan(loss_bbox)
            if nan_indexes.sum() == 0:
                loss_bbox = torch.zeros(1, device=s_cls_scores.device).sum()
            else:
                loss_bbox = loss_bbox[nan_indexes].mean()

        loss_centerness = F.binary_cross_entropy(
            s_centernesses[b_mask].sigmoid(),
            t_centernesses[b_mask].sigmoid(),
            reduction='mean'
        )

        unsup_losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness
        )

        return unsup_losses


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss
