#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/30 0:14
# @Author : WeiHua

from mmrotate.models import RotatedFCOSHead, ROTATED_HEADS


@ROTATED_HEADS.register_module()
class SemiRotatedFCOSHead(RotatedFCOSHead):
    def __init__(self, num_classes, in_channels, **kwargs):
        super(SemiRotatedFCOSHead, self).__init__(
            num_classes,
            in_channels,
            **kwargs)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      get_data=False,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            get_data (Bool): If return logit only.

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        if get_data:
            return self(x)
        return super(SemiRotatedFCOSHead, self).forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs
        )
