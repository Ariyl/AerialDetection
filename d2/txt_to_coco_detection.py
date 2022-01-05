import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from detectron2.data import transforms as T


path = "/media/csudky2020/zl/dataset/dota/val/rotated_labelTxt/60/box_to_image"
files = os.listdir(path)
image_num = len(files)
image_path = "/media/csudky2020/zl/dataset/dota/val/images"

categories = ["large-vehicle", "swimming-pool", "helicopter", "bridge", "plane", "ship", "soccer-ball-field",
              "basketball-court", "ground-track-field", "small-vehicle", "harbor", "baseball-diamond", "tennis-court",
              "roundabout", "storage-tank", "container-crane"]

with open("./train2017_v1_5_predict_angle_60.json", 'w') as f:
    test_dict = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "large-vehicle"},
                                                                {"id": 1, "name": "swimming-pool"}, {"id": 2, "name": "helicopter"}, {"id": 3, "name": "bridge"}, {"id": 4, "name": "plane"}, {"id": 5, "name": "ship"}, {"id": 6, "name": "soccer-ball-field"}, {"id": 7, "name": "basketball-court"}, {"id": 8, "name": "ground-track-field"},
                                                                {"id": 9, "name": "small-vehicle"}, {"id": 10, "name": "harbor"}, {"id": 11, "name": "baseball-diamond"}, {"id": 12, "name": "tennis-court"}, {"id": 13, "name": "roundabout"}, {"id": 14, "name": "storage-tank"}]}
    for ind, file in enumerate(files):
        imagename = file.split('.')[0] + '.png'
        image = cv2.imread(os.path.join(image_path, imagename))
        h, w = image.shape[:2]
        transform = T.RotationTransform(w=w, h=h, angle=60, expand=False)
        image = transform.apply_image(image)
        height, width = image.shape[:2]
        # width = 1024
        # height = 1024
        image_info = {"file_name": imagename, "id": ind, "width": width, "height": height}

        # if os.path.getsize(os.path.join(path, file)) == 0:
        #     os.remove(os.path.join(path, file))
        #     continue
        for f_line in open(os.path.join(path, file)):
            image_num += 1
            datas = f_line.split(" ")
            categ_id = categories.index(datas[-2])
            datas = datas[0:-2]
            length = len(datas)
            datas = [float(data) for data in datas]
            x = np.asarray(datas[0:length:2])
            y = np.asarray(datas[1:length + 1:2])

            min_px = np.min(x)
            min_py = np.min(y)
            max_px = np.max(x)
            max_py = np.max(y)
            x_extend = np.diff(np.concatenate((x, x[0][None]), axis=0))
            y_extend = np.diff(np.concatenate((y, y[0][None]), axis=0))
            length_side = np.power((np.power(x_extend, 2) + np.power(y_extend, 2)), 0.5)

            ind_max = np.argmax(length_side, -1)
            angle = np.arctan((x_extend / (y_extend + 1e-8))[ind_max]) * 180 / np.pi

            w = np.min(length_side)
            h = np.max(length_side)
            center_x = (max_px + min_px) / 2
            center_y = (max_py + min_py) / 2
            area = np.max(length_side) * np.min(length_side)
            box = [center_x, center_y, w, h, angle]
            # box.extend(datas)
            # x_len = max([abs(x[i] - x[i+1]) for i in range(len(x) - 1)])
            # y_len = max([abs(y[i] - y[i+1]) for i in range(len(y) - 1)])
            # area = x_len * y_len
            # box = data[1:]

            anno_info = {"category_id": categ_id, "image_id": ind, "bbox": box, "id": image_num, "area": area, "iscrowd": 0, "segmentation": []}
            test_dict['annotations'].append(anno_info)

        test_dict['images'].append(image_info)
    json.dump(test_dict, f)
    print("finish!!")
