from __future__ import division

import argparse
import os.path as osp
import shutil
import tempfile

import mmcv
from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, \
    HBBSeg2Comp4, OBBDet2Comp4, OBBDetComp4, \
    HBBOBB2Comp4, HBBDet2Comp4

import argparse
import copy

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch
import json
from mmcv import Config
import sys
from DOTA_devkit.ResultMerge_multi_process import *
from test_mRI import generate_pkl

# import pdb; pdb.set_trace()
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='../configs/DOTA/retinanet_obb_r50_fpn_2x_dota_mRI.py')
    parser.add_argument('--checkpoint', default='./work_dirs/retinanet_obb_r50_fpn_2x_dota/epoch_24.pth')
    parser.add_argument('--type', default=r'OBB',
                        help='parse type of detector')

    parser.add_argument('--out', default='./work_dirs/retinanet_obb_r50_fpn_2x_dota_mRI')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--log_dir', help='log the inference speed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def OBB2HBB(srcpath, dstpath):
    filenames = util.GetFileFromThisRootDir(srcpath)
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    for file in filenames:
        with open(file, 'r') as f_in:
            with open(os.path.join(dstpath, os.path.basename(os.path.splitext(file)[0]) + '.txt'), 'w') as f_out:
                lines = f_in.readlines()
                splitlines = [x.strip().split() for x in lines]
                for index, splitline in enumerate(splitlines):
                    imgname = splitline[0]
                    score = splitline[1]
                    poly = splitline[2:]
                    poly = list(map(float, poly))
                    xmin, xmax, ymin, ymax = min(poly[0::2]), max(poly[0::2]), min(poly[1::2]), max(poly[1::2])
                    rec_poly = [xmin, ymin, xmax, ymax]
                    outline = imgname + ' ' + score + ' ' + ' '.join(map(str, rec_poly))
                    if index != (len(splitlines) - 1):
                        outline = outline + '\n'
                    f_out.write(outline)

def parse_results(cfg, resultfile, dstpath, type):
    data_test = cfg.data['test']
    dataset = get_dataset(data_test)
    outputs = mmcv.load(resultfile)
    if type == 'OBB':
        #  dota1 has tested
        obb_results_dict = OBBDetComp4(dataset, outputs)
        current_thresh = 0.1
    elif type == 'HBB':
        # dota1 has tested
        hbb_results_dict = HBBDet2Comp4(dataset, outputs)
    elif type == 'HBBOBB':
        # dota1 has tested
        # dota2
        hbb_results_dict, obb_results_dict = HBBOBB2Comp4(dataset, outputs)
        current_thresh = 0.3
    elif type == 'Mask':
        # TODO: dota1 did not pass
        # dota2, hbb has passed, obb has passed
        hbb_results_dict, obb_results_dict = HBBSeg2Comp4(dataset, outputs)
        current_thresh = 0.3

    dataset_type = cfg.dataset_type

    if 'obb_results_dict' in vars():
        if not os.path.exists(os.path.join(dstpath, 'Task1_results')):
            os.makedirs(os.path.join(dstpath, 'Task1_results'))

        for cls in obb_results_dict:
            with open(os.path.join(dstpath, 'Task1_results', cls + '.txt'), 'w') as obb_f_out:
                for index, outline in enumerate(obb_results_dict[cls]):
                    if index != (len(obb_results_dict[cls]) - 1):
                        obb_f_out.write(outline + '\n')
                    else:
                        obb_f_out.write(outline)

        if not os.path.exists(os.path.join(dstpath, 'Task1_results_nms')):
            os.makedirs(os.path.join(dstpath, 'Task1_results_nms'))

        mergebypoly_multiprocess(os.path.join(dstpath, 'Task1_results'),
                                 os.path.join(dstpath, 'Task1_results_nms'), nms_type=r'py_cpu_nms_poly_fast', o_thresh=current_thresh)

        OBB2HBB(os.path.join(dstpath, 'Task1_results_nms'),
                         os.path.join(dstpath, 'Transed_Task2_results_nms'))

    if 'hbb_results_dict' in vars():
        if not os.path.exists(os.path.join(dstpath, 'Task2_results')):
            os.makedirs(os.path.join(dstpath, 'Task2_results'))
        for cls in hbb_results_dict:
            with open(os.path.join(dstpath, 'Task2_results', cls + '.txt'), 'w') as f_out:
                for index, outline in enumerate(hbb_results_dict[cls]):
                    if index != (len(hbb_results_dict[cls]) - 1):
                        f_out.write(outline + '\n')
                    else:
                        f_out.write(outline)

        if not os.path.exists(os.path.join(dstpath, 'Task2_results_nms')):
            os.makedirs(os.path.join(dstpath, 'Task2_results_nms'))
        mergebyrec(os.path.join(dstpath, 'Task2_results'),
            os.path.join(dstpath, 'Task2_results_nms'))

if __name__ == '__main__':
    args = parse_args()

    config_file = args.config
    config_name = os.path.splitext(os.path.basename(config_file))[0]
    cfg = mmcv.Config.fromfile(config_file)
    for angle in range(0, 360, 15):
        angle_config = copy.deepcopy(cfg)
        angle_args = copy.deepcopy(args)
        test_config = angle_config.data.test
        ann = test_config['ann_file'].split('/')
        ann.insert(-1, str(angle))
        img = test_config['img_prefix'].split('/')
        img.insert(-1, str(angle))
        out_pkl = angle_args.out + '/' + str(angle)
        os.makedirs(out_pkl, exist_ok=True)
        test_config['ann_file'] = '/'.join(ann)
        test_config['img_prefix'] = '/'.join(img)
        angle_args.out = out_pkl + '/results.pkl'
        angle_config.data.test = test_config
        generate_pkl(angle_args, angle_config)

        pkl_file = os.path.join('work_dirs', config_name, str(angle), 'results.pkl')
        output_path = os.path.join('work_dirs', config_name, str(angle))
        type = args.type
        parse_results(angle_config, pkl_file, output_path, type)

