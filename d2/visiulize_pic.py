from pycocotools.coco import COCO
import cv2
import os

path = '/media/csudky2020/zl/coco_data/dota_1024_rotated/train2017.json'
imaga_path = '/media/csudky2020/zl/coco_data/dota_1024_rotated/train'
coco = COCO(path)

idx = coco.getCatIds('container-crane')[0]
imgIds = coco.catToImgs[idx]
imgInfos = coco.loadImgs(imgIds)
for imgInfo in imgInfos:
    img = os.path.join(imaga_path, imgInfo['file_name'])
    annIds = coco.getAnnIds(imgInfo['id'])
    anns = coco.loadAnns(annIds)
    boxes = [ann['bbox'] for ann in anns if ann['category_id'] == idx]
    img = cv2.imread(img)
    for box in boxes:
        for i in range(4, 9, 2):
            cv2.line(img, (box[i], box[i+1]), (box[i+2], box[i+3]), (255, 0, 0))
    cv2.namedWindow('img', 0)
    row, colume = img.shape[:2]
    cv2.resizeWindow('img', row, colume)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





print(111)
