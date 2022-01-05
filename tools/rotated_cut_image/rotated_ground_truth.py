from detectron2.data import transforms as T
import os
import pandas as pd
import numpy as np
import cv2
# from SplitOnlyImage import splitbase
import shapely.geometry as shgeo

label_path = '/raid/detection/dota/val/labelTxt-v1.5'
image_path = '/raid/detection/dota/val/images'
files = os.listdir(label_path)
output_path = '/raid/detection/dota/val/rotated_labelTxt'

for filename in files:
    file = os.path.join(label_path, filename)
    df = pd.read_csv(file)
    gound_truth = np.array(df)[1:]
    coords = np.array([coord[0].split(' ')[:-2] for coord in gound_truth]).astype(np.float64).reshape(-1, 2)
    source_area = np.asarray([shgeo.Polygon(coords[i:(i + 4), :]).area for i in range(0, coords.shape[0], 4)])
    info_1 = [coord[0].split(' ')[-2] for coord in gound_truth]
    info_2 = [coord[0].split(' ')[-1] for coord in gound_truth]
    for angle in range(0, 360, 15):
        print(angle)
        os.makedirs(os.path.join(output_path, str(angle), 'labelTxt_v1.5'), exist_ok=True)
        # os.makedirs(os.path.join(output_path, str(angle), 'images'), exist_ok=True)
        # split = splitbase(r'/raid/detection/dota/val',
        #                   r'/raid/detection/dota/val/rotated_labelTxt/' + str(angle) + '/images')
        img = cv2.imread(os.path.join(image_path, filename.split('.')[0] + '.png'))
        h, w = img.shape[:2]
        transform = T.RotationTransform(w=w, h=h, angle=angle, expand=True)
        img = transform.apply_image(img)
        nh, nw = img.shape[:2]
        # split.splitdata(filename.split('.')[0], img, 1)
        angle_coords = np.minimum(transform.apply_coords(coords).clip(min=0), [nw, nh])
        angle_area = np.asarray([shgeo.Polygon(angle_coords[i:(i + 4), :]).area for i in range(0, angle_coords.shape[0], 4)])
        angle_coords = angle_coords.reshape(-1, 8)
        print(angle_area / source_area)
        res = pd.DataFrame(angle_coords)
        res.insert(8, 8, info_1)
        res.insert(9, 9, info_2)
        angle_out = os.path.join(output_path, str(angle), 'labelTxt_v1.5', filename)
        res.to_csv(angle_out, sep=" ", header=None, index=False, mode='a')

