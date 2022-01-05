from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data import transforms as T
import cv2
import random

train_name = 'rotated_dota'
register_coco_instances(train_name, {}, './train2017_v1_5_predict_angle_60.json', '/media/csudky2020/zl/dataset/dota/val/images')

airplane_metadata = MetadataCatalog.get(train_name)
dataset_dicts = DatasetCatalog.get(train_name)
for d in random.sample(dataset_dicts, 100):
    e = dict(d)
    name = e.get("file_name")
    img = cv2.imread(name)
    h, w = img.shape[:2]
    transformer = T.RotationTransform(w=w, h=h, angle=60, expand=False)
    img = transformer.apply_image(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=airplane_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(e, )
    cv2.namedWindow('img', 0)
    row, colume = img.shape[:2]
    cv2.resizeWindow('img', row, colume)
    cv2.imshow('img', vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
