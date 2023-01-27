# Code written by Dmytro Zabolotnii, 2020/2021

"""
Quick script for massive generation of input images and ground truth values from main trainval nuscenes data set
Extremely demanding on memory because loading entire nuscenes dataset object takes 5-6 gb at minimum
"""

import gc
from nuscenes.nuscenes import NuScenes
import argparse
import os.path as osp

from stage1_input_generation import main

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate mass input for neural ' +
                                                 'network stage 1 and ground truth values.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start', default=0, type=int,
                        help='Starting index of scene processed in main nuscenes dataset')
    parser.add_argument('--end', default=100, type=int,
                        help='Ending index of scene processed in main nuscenes dataset')
    parser.add_argument('--dataroot', default=osp.join(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
                                                       'data', 'sets', 'nuscenes')), type=str,
                        help='Path to main nuscenes dataset')
    parser.add_argument('--out_dir', default='output', type=str, help='Output folder name for the whole dataset')
    parser.add_argument('--patch_gen', default=0, type=int, help='Amount of additional patches to generate per scene')
    parser.add_argument('--patch_gen_only', default=0, help='Generate patches only')

    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)
    dataroot = args.dataroot
    out_dir = args.out_dir
    patch_gen = args.patch_gen
    patch_gen_only = bool(args.patch_gen_only)
    # Enable garbage collector and load the main Nuscenes dataset
    gc.enable()
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    # Generate BEV from given Nuscenes scenes concurrently
    for scene in nusc.scene[start:end+1]:
        name = scene['name']

        main(nusc, use_as_function=True, scene_name=name, gt_gen=True, out_dir=osp.join(osp.dirname(
            osp.dirname(osp.abspath(__file__))), 'input', out_dir),
            patch_gen=patch_gen, patch_gen_only=patch_gen_only, min_box=1)
