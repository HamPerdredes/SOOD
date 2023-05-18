#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/4 1:03
# @Author : WeiHua

import numpy as np
import cv2
import torch
import random
from mmdet.core import multi_apply
from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma
import torch.nn.functional as F


def cal_dist(pt1, pt2):
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2


def xywha2mask_single(rotate_box):
    xc, yc, w, h, ag = rotate_box
    wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
    hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
    p1 = (xc - wx - hx, yc - wy - hy)
    p2 = (xc + wx - hx, yc + wy - hy)
    p3 = (xc + wx + hx, yc + wy + hy)
    p4 = (xc - wx + hx, yc - wy + hy)
    pts = np.array([p1, p2, p3, p4]).astype(np.int)
    # hard-code
    obj_mask = np.zeros((1024, 1024, 1), dtype=np.uint8)
    obj_mask = torch.from_numpy(cv2.fillPoly(obj_mask, [pts], 255)) > 100
    return obj_mask.squeeze(-1), None


def single_random_selection(obj_mask, ratio=0.25):
    pts_mask = torch.zeros_like(obj_mask)
    pts = obj_mask.nonzero().squeeze(-1)
    select_list = list(range(len(pts)))
    for _ in range(3):
        random.shuffle(select_list)
    select_num = int(ratio * len(pts))
    pts = pts[select_list[:select_num]]
    pts_mask[pts] = True
    # None here for avoiding multi_apply bug
    return pts_mask, select_num


def single_topk_selection(obj_mask, score_map, ratio=0.25):
    pts_mask = torch.zeros_like(obj_mask)
    pts = obj_mask.nonzero().squeeze(-1)
    count_num = int(pts.size(0) * ratio)
    max_vals = torch.max(score_map, 1)[0]
    sorted_vals, sorted_inds = torch.topk(max_vals, pts.size(0))
    pts_mask[pts[sorted_inds[:count_num]]] = True
    # None here for avoiding multi_apply bug
    return pts_mask, None


def xywha2rbox(rotate_boxes, gpu_device, h=1024, w=1024, img_meta=None,
               ret_instance_pts=False, ratio=0.25,
               ret_base_ang=False, score_map=None, topk=False):
    """Random Sampling within rotate boxes."""
    cls_labels = rotate_boxes[:, -1]
    obj_masks, _ = multi_apply(xywha2mask_single, rotate_boxes[:, :-2])
    num_obj = len(obj_masks)
    # N, H, W -> N, HW
    obj_masks = torch.stack(obj_masks, dim=0).reshape(num_obj, -1)
    if ret_instance_pts:
        instance_pts = list()
    with torch.no_grad():
        if num_obj > 200:
            if topk:
                score_map_1d = score_map.permute(0, 2, 3, 1).reshape(-1, score_map.shape[1])
                mask_list, _ = multi_apply(single_topk_selection, obj_masks,
                                           [score_map_1d[_] for _ in obj_masks],
                                           [ratio for _ in range(len(obj_masks))])
            else:
                mask_list, _ = multi_apply(single_random_selection, obj_masks,
                                           [ratio for _ in range(len(obj_masks))])
            mask = torch.stack(mask_list, dim=0).sum(dim=0, dtype=bool).reshape(1024, 1024).to(
                gpu_device)
        else:
            if topk:
                score_map_1d = score_map.permute(0, 2, 3, 1).reshape(-1, score_map.shape[1])
                mask_list, _ = multi_apply(single_topk_selection, obj_masks.to(gpu_device),
                                           [score_map_1d[_] for _ in obj_masks],
                                           [ratio for _ in range(len(obj_masks))])
            else:
                mask_list, _ = multi_apply(single_random_selection, obj_masks.to(gpu_device),
                                           [ratio for _ in range(len(obj_masks))])
            mask = torch.stack(mask_list, dim=0).sum(dim=0, dtype=bool).reshape(1024, 1024)
    if ret_instance_pts:
        assert cls_labels.shape[0] == len(mask_list)
        for idx, temp_mask in enumerate(mask_list):
            # hard code
            temp_mask = temp_mask.reshape(1, 1, h, w).float()
            instance_pts.append(
                [F.interpolate(temp_mask, (256, 256)).squeeze(0).squeeze(0).nonzero().cpu().numpy(),
                 cls_labels[idx]])
        return mask, instance_pts
    if ret_base_ang:
        base_angs = rotate_boxes[:, -3]
        assert len(base_angs) == len(mask_list)
        ang_list = [x*ang for x, ang in zip(mask_list, base_angs)]
        base_angs = torch.stack(ang_list, dim=0).sum(dim=0).reshape(1024, 1024).to(gpu_device)
        return mask, base_angs
    # # visualization
    # raw_img = img_meta['raw_img']
    # mask_array = mask.cpu().numpy()
    # raw_img[mask_array] = [0, 255, 0]
    # cv2.imwrite("/home/whua/vis.jpg", raw_img)
    return mask
