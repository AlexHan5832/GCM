#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: ufukefe and kutalmisince
@modified by Y Han
"""
import os
import argparse
from PIL import Image
import numpy as np
from typing import List

from DeepFeatureMatcher import DeepFeatureMatcher

import time
import yaml
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default="./configs/VGG_dstlR2D2_95.yml")
    parser.add_argument('--input_dir', type=str, default="Datasets/hpatches")
    parser.add_argument('--input_pairs', type=str, default="Datasets/hpatches/image_pairs.txt")
    parser.add_argument('--output_dir', type=str, required=True)

    args_util = parser.parse_args()

    if not os.path.exists(os.path.join(args_util.output_dir, 'original_outputs')):
        os.makedirs(os.path.join(args_util.output_dir, 'original_outputs'))

    with open(args_util.config, "rb") as f:
        cfg: dict = yaml.safe_load(f)

    log_file = args_util.output_dir
    out_dir = os.path.join(log_file, "original_outputs")

    fm = DeepFeatureMatcher(**cfg)

    with open(args_util.input_pairs) as f:

        total_time = 0
        for total_pair_number, line in enumerate(tqdm(f, total=540)):
            pairs = line.split(' ')
            p1_path = args_util.input_dir + '/' + pairs[0]
            p2_path = args_util.input_dir + '/' + pairs[1]

            img_A = np.array(Image.open(p1_path))
            img_B = np.array(Image.open(p2_path))
            start_time = time.time()
            H, H_init, points_A, points_B = fm.match(img_A, img_B)

            end_time = time.time()

            total_time += end_time - start_time

            keypoints0 = points_A.T
            keypoints1 = points_B.T

            mtchs = np.vstack([np.arange(0, keypoints0.shape[0])] * 2).T

            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]

            np.savez_compressed(out_dir + '/' + p1 + '_' + p2 + '_' + 'matches',
                                keypoints0=keypoints0, keypoints1=keypoints1, matches=mtchs)
            # if output_directory:
            #     cv2.imwrite(out_dir + '/' + p1 + '_' + p2 + '_' + 'matches' + '.png',
            #                 draw_matches(img_A, img_B, keypoints0, keypoints1))

        avg_time = total_time / (total_pair_number + 1)

        print(f'Total Execution Time is: {total_time}')
        print(f'Average Execution Time is: {avg_time}')
        data = dict()
        data['configuration'] = cfg
        tm = {'total time': total_time, 'avg_time': avg_time}
        data['time'] = tm

        with open(log_file + r"\config.yml", "w") as log:
            # use yaml save some hyperparameters
            yaml.dump(data)
