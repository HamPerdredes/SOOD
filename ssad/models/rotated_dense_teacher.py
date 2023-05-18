#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 17:03
# @Author : WeiHua
import torch
import numpy as np

from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector


@ROTATED_DETECTORS.register_module()
class RotatedDenseTeacher(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss, train_cfg=None, test_cfg=None, symmetry_aware=False):
        super(RotatedDenseTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.logit_specific_weights = train_cfg.get("logit_specific_weights")
            self.region_ratio = train_cfg.get("region_ratio")
        self.symmetry_aware = symmetry_aware

    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedDenseTeacher, self).forward_train(imgs, img_metas, **kwargs)
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
            # print(f"{key}: {format_data[key]['img'].shape}")
        losses = dict()
        # supervised forward
        sup_losses = self.student.forward_train(**format_data['sup'])
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val
        if self.iter_count > self.burn_in_steps:
            # Train Logic
            # unsupervised forward
            unsup_weight = self.unsup_weight
            if self.weight_suppress == 'exp':
                target = self.burn_in_steps + 2000
                if self.iter_count <= target:
                    scale = np.exp((self.iter_count - target) / 1000)
                    unsup_weight *= scale
            elif self.weight_suppress == 'step':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= 0.25
            elif self.weight_suppress == 'linear':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps
            with torch.no_grad():
                # get teacher data
                teacher_rpn_logits, teacher_roi_logits = self.teacher.forward_train(get_data=True, use_roi_head=False,
                                                                                    **format_data['unsup_weak'])
                teacher_rpn_cls, teacher_rpn_box = teacher_rpn_logits
            # get student data
            student_rpn_logits, student_roi_logits = self.student.forward_train(get_data=True, use_roi_head=False,
                                                                                **format_data['unsup_strong'])
            student_rpn_cls, student_rpn_box = student_rpn_logits
            if self.symmetry_aware:
                unsup_losses = self.semi_loss(teacher_rpn_cls, teacher_rpn_box, teacher_roi_logits,
                                              student_rpn_cls, student_rpn_box, student_roi_logits,
                                              teacher_img_metas=format_data['unsup_weak']['img_metas'],
                                              student_img_metas=format_data['unsup_strong']['img_metas'],
                                              ratio=self.region_ratio)
            else:
                unsup_losses = self.semi_loss(teacher_rpn_cls, teacher_rpn_box, teacher_roi_logits,
                                              student_rpn_cls, student_rpn_box, student_roi_logits,
                                              ratio=self.region_ratio)
            for key, val in self.logit_specific_weights.items():
                if key in unsup_losses.keys():
                    unsup_losses[key] *= val
            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    losses[f"{key}_unsup"] = unsup_weight * val
                else:
                    losses[key] = val
        self.iter_count += 1

        return losses

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
