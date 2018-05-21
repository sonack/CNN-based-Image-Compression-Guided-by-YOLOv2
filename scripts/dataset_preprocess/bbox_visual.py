import sys
import os
from os.path import join
import time
import math
from PIL import Image, ImageDraw, ImageFont
import pdb
import numpy as np

IMAGE_PATH = "/home/snk/WindowsDisk/Download/KITTI/data_object_image_2/training/image_2/"
PROCESSED_LABEL_PATH = "/home/snk/WindowsDisk/Download/KITTI/labels/"

CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
def plot_bbox_on_image(img, boxes, savename):
    colors = [[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[1] - box[3]/2.0) * width
        y1 = (box[2] - box[4]/2.0) * height
        x2 = (box[1] + box[3]/2.0) * width
        y2 = (box[2] + box[4]/2.0) * height

        rgb = (255, 0, 0)

        classes = len(CLASS_NAMES)
        cls_id = int(box[0])
        offset = cls_id * 123457 % classes
        red   = get_color(2, offset, classes)
        green = get_color(1, offset, classes)
        blue  = get_color(0, offset, classes)
        rgb = (red, green, blue)
        draw.text((x1, y1), CLASS_NAMES[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    # pdb.set_trace()
    if savename:
        print("save img with bbox to %s" % savename)
        img.save(savename)
    return img

if __name__ == '__main__':
    if len(sys.argv) == 2:
        img_id = int(sys.argv[1])
        img = Image.open(join(IMAGE_PATH, '%06d.png' % img_id))
        label = np.loadtxt(join(PROCESSED_LABEL_PATH, '%06d.txt' % img_id))
        if label.ndim == 1:
            label = np.expand_dims(label, 0)
        # pdb.set_trace()
        plot_bbox_on_image(img, label, 'img_%06d_with_bbox.png' % img_id)
        # pdb.set_trace()
        # print (img.width)
        # print (img.height)
        
    else:
        print ('Usage: ')
        print ('  python bbox_visual.py img_id')