#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 0:27
# @Author : WeiHua
import os
import shutil

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter
import copy
import cv2

from mmcv import is_list_of
from mmdet.datasets.pipelines import Compose as BaseCompose

from mmrotate.datasets.builder import ROTATED_PIPELINES
from mmrotate.datasets.pipelines import PolyRandomRotate


class DTSingleOperation:
    def __init__(self):
        self.transform = None

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            results[key] = self.transform(results[key])
        return results


@ROTATED_PIPELINES.register_module()
class DTToPILImage(DTSingleOperation):
    def __init__(self):
        super(DTToPILImage, self).__init__()
        self.transform = transforms.ToPILImage()


# DT represents for Dense Teacher
@ROTATED_PIPELINES.register_module()
class DTRandomApply:
    def __init__(self, operations, p=0.5):
        self.p = p
        if is_list_of(operations, dict):
            self.operations = []
            for ope in operations:
                self.operations.append(build_dt_aug(**ope))
        else:
            self.operations = operations

    def __call__(self, results):
        if self.p < np.random.random():
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            for ope in self.operations:
                img = ope(img)
            results[key] = img
        return results


@ROTATED_PIPELINES.register_module()
class DTRandomGrayscale(DTSingleOperation):
    def __init__(self, p=0.2):
        super(DTRandomGrayscale, self).__init__()
        self.transform = transforms.RandomGrayscale(p=p)


@ROTATED_PIPELINES.register_module()
class DTRandCrop(DTSingleOperation):
    def __init__(self):
        super(DTRandCrop, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),
            transforms.ToPILImage(),
        ])


@ROTATED_PIPELINES.register_module()
class DTToNumpy:
    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            results[key] = np.asarray(results[key])
        return results


# ST represents Soft Teacher
@ROTATED_PIPELINES.register_module()
class STMultiBranch(object):
    def __init__(self, is_seq=False, **transform_group):
        self.is_seq = is_seq
        self.transform_group = {k: BaseCompose(v) for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = []
        if self.is_seq:
            weak_pipe = self.transform_group['unsup_weak']
            strong_pipe = self.transform_group['unsup_strong']
            res = weak_pipe(copy.deepcopy(results))
            multi_results.append(copy.deepcopy(res))
            res.pop('tag')
            multi_results.append(strong_pipe(res))
            for k, v in self.transform_group.items():
                if 'common' in k:
                    for idx in range(len(multi_results)):
                        multi_results[idx] = v(multi_results[idx])
        else:
            for k, v in self.transform_group.items():
                res = v(copy.deepcopy(results))
                if res is None:
                    return None
                multi_results.append(res)
        return multi_results


@ROTATED_PIPELINES.register_module()
class LoadEmptyAnnotations:
    def __init__(self, with_bbox=False, with_mask=False, with_seg=False, fill_value=255):
        """Load Empty Annotations for un-supervised pipeline"""
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.fill_value = fill_value

    def __call__(self, results):
        if self.with_bbox:
            results["gt_bboxes"] = np.zeros((0, 5))
            results["gt_labels"] = np.zeros((0,))
            if "bbox_fields" not in results:
                results["bbox_fields"] = []
            if "gt_bboxes" not in results["bbox_fields"]:
                results["bbox_fields"].append("gt_bboxes")
        if self.with_mask:
            raise NotImplementedError
        if self.with_seg:
            results["gt_semantic_seg"] = self.fill_value * np.ones(
                results["img"].shape[:2], dtype=np.uint8)
            if "seg_fields" not in results:
                results["seg_fields"] = []
            if "gt_semantic_seg" not in results["seg_fields"]:
                results["seg_fields"].append("gt_semantic_seg")
        return results


@ROTATED_PIPELINES.register_module()
class EmptyPolyRandomRotate(PolyRandomRotate):

    def __call__(self, results):
        """Call function of EmptyPolyRandomRotate."""
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            results['rotate'] = True
            if self.mode == 'range':
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

            class_labels = results['gt_labels']
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = \
            abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)

        return results


@ROTATED_PIPELINES.register_module()
class ExtraAttrs:
    def __init__(self, **attrs):
        self.keep_raw = attrs.pop('keep_raw', False)
        self.attrs = attrs

    def __call__(self, results):
        if self.keep_raw:
            results['raw_img'] = results['img'].copy()
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results


class DTGaussianBlur:
    def __init__(self, rad_range=[0.1, 2.0]):
        self.rad_range = rad_range

    def __call__(self, x):
        rad = np.random.uniform(*self.rad_range)
        x = x.filter(ImageFilter.GaussianBlur(radius=rad))
        return x


DT_LOCAL_AUGS = {
    'DTGaussianBlur': DTGaussianBlur
}


def build_dt_aug(type, **kwargs):
    return DT_LOCAL_AUGS[type](**kwargs)
