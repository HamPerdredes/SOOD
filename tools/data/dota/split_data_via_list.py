#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import json
import os
import shutil

def split_img_vis_list(list_file, src_dir, out_dir):
    with open(list_file, 'r', encoding='utf-8') as f:
        file_list = json.load(f)
    all_files = dict()
    for file_ in glob.glob(os.path.join(src_dir, '*.png')):
        all_files[file_.split('/')[-1]] = file_
    print(f"Total images: {len(all_files)}")
    labeled_out_dir = out_dir['labeled']
    unlabeled_out_dir = out_dir['unlabeled']
    if os.path.exists(labeled_out_dir):
        shutil.rmtree(labeled_out_dir)
    if os.path.exists(unlabeled_out_dir):
        shutil.rmtree(unlabeled_out_dir)
    os.mkdir(labeled_out_dir)
    os.mkdir(unlabeled_out_dir)
    labeled_num = 0
    for file_name, file_path in all_files.items():
        if file_name in file_list:
            shutil.copyfile(file_path, os.path.join(labeled_out_dir, file_name))
            labeled_num += 1
        else:
            shutil.copyfile(file_path, os.path.join(unlabeled_out_dir, file_name))
    assert labeled_num == len(file_list)
    print(f"Finish saving {labeled_num} labeled image.")


if __name__ == '__main__':
    # example
    list_file = 'PATH-TO-DATALIST-Json-FILE'
    src_dir = 'PATH-TO-DOTA-IMAGE-DIRECTORY'
    out_dir = dict(
        labeled='PATH-TO-SAVE-LABEL-PART',
        unlabeled='PATH-TO-SAVE-UNLABELED-PART'
    )
    split_img_vis_list(list_file, src_dir, out_dir)
