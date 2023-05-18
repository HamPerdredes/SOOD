#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/20 14:19
# @Author : WeiHua
import os
import shutil
import cv2
import glob
import numpy as np

from mmrotate.datasets.builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class CustomVisualize:
    def __init__(self, save_path='/home/whua/vis', vis_num=100):
        self.save_path = save_path
        self.vis_num = vis_num
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)

    def __call__(self, results):
        if len(glob.glob(os.path.join(self.save_path, '*'))) < self.vis_num:
            # visualize image
            img = results['img'].copy()
            filename = results['filename'].split('/')[-1].split('.')[0]
            tag = results['tag']
            cv2.imwrite(os.path.join(self.save_path, f"{tag}_{filename}.png"), img)
            rotate_boxes = results['gt_bboxes']
            show_img = img.copy()
            show_poly_img = img.copy()
            for rotate_box in rotate_boxes:
                # [cx, cy, w, h, a]
                xc, yc, w, h, ag = rotate_box
                wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
                hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
                p1 = (xc - wx - hx, yc - wy - hy)
                p2 = (xc + wx - hx, yc + wy - hy)
                p3 = (xc + wx + hx, yc + wy + hy)
                p4 = (xc - wx + hx, yc - wy + hy)
                poly = np.int0(np.array([p1, p2, p3, p4])).reshape(-1, 2)
                cv2.polylines(show_img, [poly], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.imwrite(os.path.join(self.save_path, f"{filename}_{tag}_label.png"), show_img)
            gt_polygons = results['ann_info']['polygons'].astype(np.int)
            num_poly = gt_polygons.shape[0]
            if num_poly > 0:
                cv2.polylines(show_poly_img, gt_polygons.reshape(num_poly, -1, 2), isClosed=True, color=(0, 255, 0), thickness=1)
                # get main center line
                cl_x0 = (gt_polygons[:, 0] + gt_polygons[:, 2]) / 2
                cl_y0 = (gt_polygons[:, 1] + gt_polygons[:, 3]) / 2
                cl_x1 = (gt_polygons[:, 4] + gt_polygons[:, 6]) / 2
                cl_y1 = (gt_polygons[:, 5] + gt_polygons[:, 7]) / 2
                cl_p0 = np.stack([cl_x0, cl_y0], axis=1).astype(np.int)
                cl_p1 = np.stack([cl_x1, cl_y1], axis=1).astype(np.int)
                for poly_idx in range(num_poly):
                    show_poly_img = cv2.line(show_poly_img, cl_p0[poly_idx], cl_p1[poly_idx], (0, 0, 255), 1)
                cv2.imwrite(os.path.join(self.save_path, f"{filename}_{tag}_poly.png"), show_poly_img)
        return results
