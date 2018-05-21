import os
from os import listdir, getcwd
from os.path import join
from PIL import Image
from tqdm import tqdm
import pdb

TRAINVAL_NUM = 7481
# 8 class, Misc = DontCare
CLASS = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

IMAGE_PATH = "/home/snk/WindowsDisk/Download/KITTI/data_object_image_2/training/image_2/"
LABEL_PATH = "/home/snk/WindowsDisk/Download/KITTI/data_object_label_2/training/label_2/"

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    # if w < 0 or h < 0:
    #     print ('FUCK')
    return (x,y,w,h)


# /home/snk/WindowsDisk/Download/
if not os.path.exists('KITTI/labels/'):
    os.makedirs('KITTI/labels/')

list_file = open(join('KITTI/', 'trainval.txt'), 'w')
for idx in tqdm(range(0,TRAINVAL_NUM)):
    out_file = open(join('KITTI/labels/', '%06d.txt' % idx), 'w')
    list_file.write(join(IMAGE_PATH, '%06d.png' % idx) + '\n')
    # print ('%06d.txt' % idx)
    img = Image.open(join(IMAGE_PATH, '%06d.png' % idx))
    # img.show()
    w, h = img.size
    # pdb.set_trace()
    label = open(join(LABEL_PATH, '%06d.txt' % idx))
    for line in label.readlines():
        line = line.split()
        cls = line[0]
        if cls == 'DontCare':
            cls = 'Misc'
        if cls not in CLASS:
            print ('Invalid ClassName %s!' % cls)
        cls_id = CLASS.index(cls)
        bb = convert((w,h), tuple(map(float, line[4:8])))
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()
list_file.close()
        

