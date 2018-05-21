#coding: utf-8
import os
import shutil
import random
import pdb
'''
划分train和val set
'''

# 7481
TRAINVAL_FILE_PATH = '/home/snk/WindowsDisk/Download/KITTI/trainval.txt'
TRAIN_FILE_NAME = "/home/snk/WindowsDisk/Download/KITTI/train.txt"
VAL_FILE_NAME = "/home/snk/WindowsDisk/Download/KITTI/val.txt"
VAL_NUM = 481 # sample 481 val + 7000 train examples

imgs = open(TRAINVAL_FILE_PATH, 'r').readlines()
total_imgs = len(imgs)
random.shuffle(imgs)
# pdb.set_trace()
val_set = imgs[:VAL_NUM]
train_set = imgs[VAL_NUM:]
with open(VAL_FILE_NAME, 'w') as f:
    for img in val_set:
        f.write(img)

with open(TRAIN_FILE_NAME, 'w') as f:
    for img in train_set:
        f.write(img)
