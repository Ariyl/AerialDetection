import os
import shutil


path = '/raid/detection/dota/val/images'
remove_path = '/raid/detection/dota/train/images'
remove_path2 = '/raid/detection/dota/train/labelTxt-v1.0'
# remove_path3 = '/raid/detection/dota/train/labelTxt-v1.5'
files = os.listdir(path)
for file in files:
    imagname = file
    if os.path.exists(os.path.join(remove_path, imagname)):
        os.remove(os.path.join(remove_path, imagname))
    label_txt = imagname.split('.')[0] + '.txt'
    if os.path.exists(os.path.join(remove_path2, label_txt)):
        os.remove(os.path.join(remove_path2, label_txt))
    # os.remove(os.path.join(remove_path3, label_txt))
