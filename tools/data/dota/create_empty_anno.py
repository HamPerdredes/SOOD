#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/22 10:53
# @Author : WeiHua

import os
import shutil
import glob


def create_empty_anno(src_dir, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    files = glob.glob(os.path.join(src_dir, '*.txt'))
    for file_ in files:
        file_name = file_.split('/')[-1]
        out_file_name = os.path.join(out_dir, file_name)
        with open(out_file_name, 'w', encoding='utf-8') as f:
            pass
    print(f"Finish creating {len(files)} empty annotation files.")


if __name__ == '__main__':
    src_dir = "PROJECT-DIR/data/split_ss_dota_v15/train_xx_unlabeled/annfiles"
    out_dir = "PROJECT-DIR/data/split_ss_dota_v15/train_xx_unlabeled/empty_annfiles"
    create_empty_anno(src_dir, out_dir)
