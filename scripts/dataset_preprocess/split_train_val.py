#coding: utf-8
import os
import shutil
import random
import pdb
'''
划分train和val set
'''

# 7481 train+val (其实就是KITTI的所有train)
# TRAINVAL_FILE_PATH = '/home/snk/WindowsDisk/Download/KITTI/trainval.txt'
TRAINVAL_FILE_PATH = '/home/snk/WindowsDisk/Download/KITTI/val.txt'

# TRAIN_FILE_NAME = "/home/snk/WindowsDisk/Download/KITTI/train.txt"
TRAIN_FILE_NAME = "/home/snk/WindowsDisk/Download/KITTI/test_subset.txt"

VAL_FILE_NAME = "/home/snk/WindowsDisk/Download/KITTI/val_subset.txt"

# VAL_NUM = 481  # sample 481 val + 7000 train examples
VAL_NUM = 240 # sample 241 test + 240 val

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
