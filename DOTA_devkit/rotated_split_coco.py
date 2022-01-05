from DOTA2COCO import DOTA2COCOTest
import os

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

path = '/raid/detection/dota/val/rotated_labelTxt'
for angle in range(0, 360, 15):
    image_path = os.path.join(path, str(angle))
    DOTA2COCOTest(os.path.join(image_path), os.path.join(image_path, 'val_cut_dota1_5.json'),
                  wordname_16)
