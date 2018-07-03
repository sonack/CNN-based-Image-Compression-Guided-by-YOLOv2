#coding:utf8
from __future__ import print_function
import os
from tqdm import tqdm
# make caffe filelist
IMGS_NUM = 240

WRITE_FILE = "val_imgs_caffe_fl"
with open(WRITE_FILE, 'w') as f:
    for i in tqdm(range(IMGS_NUM)):
        f.write('%d.png 0\n'%i)

