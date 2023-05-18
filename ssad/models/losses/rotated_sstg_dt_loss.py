#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES
from .utils.sample_tools import xywha2rbox
from .utils.ot_tools import OT_Loss


@ROTATED_LOSSES.register_module()
class RotatedSingleStageDTLoss(nn.Module):
    def __init__(self, cls_channels=16, loss_type='pr_origin_p5', cls_loss_type='bce',
                 aux_loss=None, sigma_scale=0.5, rbox_pts_ratio=0.25, aux_loss_cfg=dict(),
                 dynamic_weight='ang', dynamic_fix_weight=None):
        """
        Symmetry Aware Single Stage Dense Teacher Loss.
        Args:
            cls_channels:
            loss_type:
            aux_loss (Optional | str): additional loss for auxiliary
        """
        super(RotatedSingleStageDTLoss, self).__init__()
        self.cls_channels = cls_channels
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        self.loss_type = loss_type
        assert cls_loss_type in ['bce']
        self.cls_loss_type = cls_loss_type
        if aux_loss:
            assert aux_loss in ['ot_loss_norm', 'ot_ang_loss_norm']
            self.ot_weight = aux_loss_cfg.pop('loss_weight', 1.)
            self.cost_type = aux_loss_cfg.pop('cost_type', 'all')
            assert self.cost_type in ['all', 'dist', 'score']
            self.clamp_ot = aux_loss_cfg.pop('clamp_ot', False)
            self.gc_loss = OT_Loss(**aux_loss_cfg)
        self.aux_loss = aux_loss
        self.apply_ot = self.aux_loss
        self.sigma_scale = sigma_scale
        self.rbox_pts_ratio = rbox_pts_ratio
        assert dynamic_weight in ['ang', '10ang', '50ang', '100ang']
        self.dynamic_weight = dynamic_weight
        if dynamic_fix_weight:
            self.dynamic_fix_weight = dynamic_fix_weight
        else:
            if self.dynamic_weight == 'ang':
                self.dynamic_fix_weight = 1.0
            else:
                self.dynamic_fix_weight = 1.0

    def forward(self, teacher_logits, student_logits, ratio=0.01, teacher_preds=None, img_metas=None):
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
            teacher_preds (Optional | List): teacher's prediction results.
                List[list[array]] -> Samples[classes[rotate_boxes]]
            img_metas (Optional | Dict): img metas

        Returns:

        """
        gpu_device = teacher_logits[0][0].device
        if self.loss_type in ['pr_origin_p5']:
            mask = torch.zeros((len(teacher_preds), 1, 1024, 1024), device=gpu_device)
            base_angs = torch.zeros((len(teacher_preds), 1, 1024, 1024), device=gpu_device)
            for img_idx, bbox_result in enumerate(teacher_preds):
                bboxes = []
                for cls_idx, bbox_per_cls in enumerate(bbox_result):
                    if bbox_per_cls.shape[0] > 0:
                        bboxes.append(
                            np.hstack([bbox_per_cls, cls_idx * np.ones((bbox_per_cls.shape[0], 1))])
                        )
                if len(bboxes) == 0:
                    continue
                # simple verification
                bboxes = np.vstack(bboxes)
                valid_idx = (bboxes[:, -2] > 0.5).nonzero()[0]
                if len(valid_idx) == 0:
                    continue
                mask[img_idx] = xywha2rbox(bboxes[valid_idx], gpu_device,
                                           img_meta=img_metas['unsup_strong'][img_idx],
                                           ratio=self.rbox_pts_ratio).to(gpu_device)
            if self.loss_type in ['pr_origin_p5']:
                mask = F.interpolate(mask.float(), (128, 128)).bool().squeeze(1)
            else:
                raise RuntimeError(f"Not support {self.loss_type}")
        else:
            raise RuntimeError(f"Not support {self.loss_type}")
        num_valid = sum([_.sum() for _ in mask]) if isinstance(mask, list) else mask.sum()
        if num_valid == 0:
            if self.apply_ot:
                return dict(loss_raw=torch.tensor(0., device=teacher_logits[0][0].device),
                            loss_gc=torch.tensor(0., device=teacher_logits[0][0].device))
            return dict(loss_raw=torch.tensor(0., device=teacher_logits[0][0].device))
        else:
            if self.loss_type in ['pr_origin_p5']:
                t_cls_scores, t_bbox_preds, t_angle_preds, t_centernesses = teacher_logits
                s_cls_scores, s_bbox_preds, s_angle_preds, s_centernesses = student_logits
                target_size = (128, 128)
                t_cls_scores = F.interpolate(t_cls_scores[0], target_size).permute(0, 2, 3, 1)[mask]
                t_bbox_preds = F.interpolate(t_bbox_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                t_angle_preds = F.interpolate(t_angle_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                t_centernesses = F.interpolate(t_centernesses[0], target_size).permute(0, 2, 3, 1)[mask]

                s_cls_scores = F.interpolate(s_cls_scores[0], target_size).permute(0, 2, 3, 1)[mask]
                s_bbox_preds = F.interpolate(s_bbox_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                s_angle_preds = F.interpolate(s_angle_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                s_centernesses = F.interpolate(s_centernesses[0], target_size).permute(0, 2, 3, 1)[mask]

                t_bbox_preds = torch.cat([t_bbox_preds, t_angle_preds], dim=-1)
                s_bbox_preds = torch.cat([s_bbox_preds, s_angle_preds], dim=-1)
            else:
                raise RuntimeError(f"Not support {self.loss_type}")
            with torch.no_grad():
                if self.dynamic_weight in ['ang', '10ang', '50ang', '100ang']:
                    # determine loss weights via angle difference
                    # [-pi, pi] -> [0, pi] -> [0, 1] -> [1, 2]
                    loss_weight = torch.abs(t_bbox_preds[:, -1] - s_bbox_preds[:, -1]) / np.pi
                    if self.dynamic_weight == '10ang':
                        loss_weight = torch.clamp(10 * loss_weight.unsqueeze(-1), 0, 1) + 1
                    elif self.dynamic_weight == '50ang':
                        loss_weight = torch.clamp(50 * loss_weight.unsqueeze(-1), 0, 1) + 1
                    elif self.dynamic_weight == '100ang':
                        loss_weight = torch.clamp(100 * loss_weight.unsqueeze(-1), 0, 1) + 1
                    else:
                        loss_weight = loss_weight.unsqueeze(-1) + 1
                else:
                    raise RuntimeError(f"Not support {self.dynamic_weight}")
            if self.cls_loss_type == 'bce':
                loss_cls = F.binary_cross_entropy(
                    s_cls_scores.sigmoid(),
                    t_cls_scores.sigmoid(),
                    reduction="none",
                )
                loss_cls = (loss_cls * loss_weight).mean()
            else:
                raise RuntimeError(f"Not support {self.cls_loss_type}")
            loss_bbox = self.bbox_loss(
                s_bbox_preds,
                t_bbox_preds,
            ) * t_centernesses.sigmoid()
            loss_bbox = (loss_bbox * loss_weight).mean()
            loss_centerness = F.binary_cross_entropy(
                s_centernesses.sigmoid(),
                t_centernesses.sigmoid(),
                reduction='none'
            )
            loss_centerness = (loss_centerness * loss_weight).mean()
            unsup_losses = dict(loss_raw=self.dynamic_fix_weight * (loss_cls + loss_bbox + loss_centerness))
            if self.aux_loss:
                loss_gc = torch.zeros(1).to(gpu_device)
                if self.loss_type in ['pr_origin_p5']:
                    if self.aux_loss in ['ot_ang_loss_norm']:
                        t_score_map = teacher_logits[2][0]
                        s_score_map = student_logits[2][0]
                    else:
                        t_score_map = teacher_logits[0][0]
                        s_score_map = student_logits[0][0]

                    assert t_score_map.shape[0] == 1, f"Only support batch size equals to 1 for now."
                    if teacher_logits[0][0].shape[-2:] != mask.shape[-2:]:
                        t_score_map = F.interpolate(t_score_map, mask.shape[-2:]).permute(0, 2, 3, 1)
                        s_score_map = F.interpolate(s_score_map, mask.shape[-2:]).permute(0, 2, 3, 1)
                    else:
                        t_score_map = t_score_map.permute(0, 2, 3, 1)
                        s_score_map = s_score_map.permute(0, 2, 3, 1)

                    if self.aux_loss in ['ot_loss_norm']:
                        t_score_map = torch.softmax(t_score_map, dim=-1)
                        s_score_map = torch.softmax(s_score_map, dim=-1)
                    elif self.aux_loss in ['ot_ang_loss_norm']:
                        t_score_map = torch.abs(t_score_map) / np.pi
                        s_score_map = torch.abs(s_score_map) / np.pi
                    for img_idx in range(t_score_map.shape[0]):
                        t_score, score_cls = torch.max(t_score_map[img_idx][mask[img_idx]], dim=-1)
                        s_score = s_score_map[img_idx][mask[img_idx]][
                            torch.arange(t_score.shape[0], device=gpu_device),
                            score_cls]
                        pts = mask[img_idx].nonzero()
                        if len(pts) <= 1:
                            continue
                        loss_gc += self.gc_loss(t_score, s_score, pts, cost_type=self.cost_type,
                                                clamp_ot=self.clamp_ot)
                    unsup_losses.update(loss_gc=self.ot_weight * loss_gc.sum() / t_score_map.shape[0])
                else:
                    raise RuntimeError(f"Not support {self.loss_type}")

        return unsup_losses

