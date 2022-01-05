# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import xml.etree.ElementTree as ET
import os
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
from DOTA_devkit import polyiou
import torch
import pandas as pd
from functools import partial

def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        #if i % 100 == 0:
         #   print ('Reading annotation for {:d}/{:d}'.format(
          #      i + 1, len(imagenames)) )
        # save
        #print ('Saving cached annotations to {:s}'.format(cachefile))
        #with open(cachefile, 'w') as f:
         #   cPickle.dump(recs, f)
    #else:
        # load
        #with open(cachefile, 'r') as f:
         #   recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)


    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def evaluation_task1(detpath, annopath, imagesetfile):


    # detpath = r'/media/csudky2020/zl/rotated_detection/AerialDetection/tools/work_dirs/htc_r50_fpn_1x_dota_mRI/0/Task1_results_nms/{:s}.txt'
    # annopath = r'/media/csudky2020/zl/dataset/dota/val/rotated_labelTxt/0/labelTxt/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'/media/csudky2020/zl/dataset/dota/val/imagefile.txt'

    # For DOTA-v1.0
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    classaps = []
    map = 0
    for classname in classnames:
        # print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        # print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
       # plt.show()
    map = map/len(classnames)
    print('map:', map)
    return 100*np.array(classaps), map
    # print('classaps: ', classaps)


if __name__ == '__main__':
    imagesetfile = r'/raid/detection/dota/val/imagefile.txt'
    work_dirs = '/home/dgx/workspace/cui/AerialDetection/tools/work_dirs/faster_rcnn_obb_r50_fpn_1x_dota_mRI/'
    clsAp = None
    mAp = None
    for angle in range(0, 360, 15):
        print(angle)
        detpath = work_dirs + str(angle) + '/Task1_results_nms/{:s}.txt'
        annopath = r'/raid/detection/dota/val/rotated_labelTxt/' + str(angle) + '/labelTxt_v1.0/{:s}.txt'

        angle_classaps, angle_map = evaluation_task1(detpath, annopath, imagesetfile)
        angle_map = torch.tensor(angle_map)[None]
        angle_classaps = torch.tensor(angle_classaps)[:, None]
        if clsAp is None:
            mAp = angle_map
            clsAp = angle_classaps
        else:
            mAp = torch.cat([mAp, angle_map], dim=-1)
            clsAp = torch.cat([clsAp, angle_classaps], dim=-1)
    mAp_res = pd.DataFrame(mAp.tolist())
    mAp_out = work_dirs + 'mAP.txt'
    clsAp_res = pd.DataFrame(clsAp.tolist())
    clsAp_out = work_dirs + 'clsAp.txt'
    mAp_res.to_csv(mAp_out, sep=" ", header=None, index=False, mode='a')
    clsAp_res.to_csv(clsAp_out, sep=" ", header=None, index=False, mode='a')