from detectron2.data import transforms as T
import os
import pandas as pd
import numpy as np
import cv2
from SplitOnlyImage import splitbase
import shapely.geometry as shgeo
import shutil

label_path = '/media/csudky2020/zl/dataset/dota/val/labelTxt-v1.5/DOTA-v1.5_val'
image_path = '/media/csudky2020/zl/dataset/dota/val/images'
files = os.listdir(label_path)
output_path = '/media/csudky2020/zl/dataset/dota/val/rotated_labelTxt'

for filename in files:
    file = os.path.join(label_path, filename)
    df = pd.read_csv(file)
    gound_truth = np.array(df)[1:]
    coords = np.array([coord[0].split(' ')[:-2] for coord in gound_truth]).astype(np.float64).reshape(-1, 2)
    source_area = np.asarray([shgeo.Polygon(coords[i:(i+4), :]).area for i in range(0, coords.shape[0], 4)])
    info_1 = np.asarray([coord[0].split(' ')[-2] for coord in gound_truth])
    info_2 = np.asarray([coord[0].split(' ')[-1] for coord in gound_truth])
    for angle in range(0, 360, 10):
        os.makedirs(os.path.join(output_path, str(angle), 'labelTxt_v1.5'), exist_ok=True)
        # os.makedirs(os.path.join(output_path, str(angle), 'images'), exist_ok=True)
        # split = splitbase(r'/media/csudky2020/zl/dataset/dota/val',
        #                   r'/media/csudky2020/zl/dataset/dota/val/rotated_labelTxt/' + str(angle) + '/images')
        img = cv2.imread(os.path.join(image_path, filename.split('.')[0] + '.png'))
        h, w = img.shape[:2]
        transform = T.RotationTransform(w=w, h=h, angle=angle, expand=False)
        # img = transform.apply_image(img)
        # split.splitdata(filename.split('.')[0], img, 1)
        angle_coords = np.minimum(transform.apply_coords(coords).clip(min=0).reshape(-1, 8), [w, h] * 4)
        angle_area_coords = angle_coords.reshape(-1, 2)
        rotated_area = np.asarray([shgeo.Polygon(angle_area_coords[i:(i+4), :]).area for i in range(0, angle_area_coords.shape[0], 4)])
        area_ind = (rotated_area / source_area) > 0.8
        angle_coords = angle_coords[area_ind]
        res = pd.DataFrame(angle_coords)
        info_angle_1 = info_1[area_ind].tolist()
        info_angle_2 = info_2[area_ind].tolist()
        res.insert(8, 8, info_angle_1)
        res.insert(9, 9, info_angle_2)
        angle_out = os.path.join(output_path, str(angle), 'labelTxt_v1.5', filename)
        res.to_csv(angle_out, sep=" ", header=None, index=False, mode='a')
    print(1111)
