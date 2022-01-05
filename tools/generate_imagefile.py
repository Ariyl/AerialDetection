import os

path = '/raid/detection/dota/val/images'
files = os.listdir(path)
output_path = '/raid/detection/dota/val/imagefile.txt'
with open(output_path, 'w+') as f:
    for file in files:
        imagefile = file.split('.')[0]
        f.write(imagefile + '\r')

f.close()
