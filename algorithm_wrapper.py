#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@authors: ufukefe and kutalmisince
@modified by Y Han
"""
import os
import argparse
import numpy as np


#First, extract and save the original algorithm's output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_dir', type=str, default="./Datasets/hpatches")
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

 #Then, read saved outputs and transform to proper format (keypointsA, keypointsB, matches)

    root_path = os.getcwd()

    args.output_dir = root_path + args.output_dir

    args.dataset_dir = root_path + args.dataset_dir

    # args.alg_dir = root_path + args.alg_dir

    pairs_out = os.listdir(os.path.join(args.output_dir, 'original_outputs'))

    with open(args.dataset_dir + '/' + 'image_pairs.txt') as f:
        for line in f:
            pairs = line.split(' ')
            subset = pairs[0].split('/')[0]
            subsubset = pairs[0].split('/')[1]
            p1 = pairs[0].split('/')[2].split('.')[0]
            p2 = pairs[1].split('/')[2].split('.')[0]
            
            if not os.path.exists(args.output_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset):
                os.makedirs(args.output_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset)
            
            for k in pairs_out:
                if p1 in k and p2 in k:
                    
                    # Original Algorithm's Output
                    pair_out = np.load(args.output_dir + '/' + 'original_outputs' + '/' + k)
                    
                    keypoints0 = pair_out['keypoints0']
                    keypoints1 = pair_out['keypoints1']
                    mtchs = pair_out['matches']
                    
                    # Wrapper's Output
                    pointsA = keypoints0
                    pointsB = keypoints1
                    matches = np.vstack(((mtchs > -1).nonzero(), mtchs[mtchs > -1])).T                  
                    
                    np.savez_compressed(args.output_dir + '/' + 'outputs' + '/' + subset + '/' + subsubset + 
                                        '/' + k, pointsA=pointsA, pointsB=pointsB, matches=mtchs)
                    
          
