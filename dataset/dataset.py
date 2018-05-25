#coding: utf-8
from __future__ import print_function, division
import os
from os.path import join
import torch as th
import numpy as np
# from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from collections import Iterable, Iterator
import matplotlib.pyplot as plt

import random
from PIL import Image
import pdb
from config import opt
TEST_LABEL_PATH = '/home/snk/WindowsDisk/Download/KITTI/test_labels/' if not opt.GPU_HPC else "/share/Dataset/KITTI/test_labels/"
TRAIN_GT_PATH = '/home/snk/WindowsDisk/Download/KITTI/labels/' if not opt.GPU_HPC else "/share/Dataset/KITTI/labels/"


def generate_crop_mask(img_size, crop_region, full_mask, crop_size, original_crop = False):
    w, h = img_size
    left, upper, right, lower = crop_region

    left_relative = left / w
    right_relative = right / w
    upper_relative = upper / h
    lower_relative = lower / h
    mw = w
    mh = h
    m_crop_size = crop_size
    if not original_crop:
        mw = w // 8
        mh = h // 8
        m_crop_size = m_crop_size // 8

    mask_left = int(left_relative * mw)
    mask_right = mask_left + m_crop_size
    mask_upper = int(upper_relative * mh)
    mask_lower = mask_upper + m_crop_size
    crop_mask = full_mask[mask_upper:mask_lower,mask_left:mask_right]
    return crop_mask

def bbox_center_crop(img, full_mask, original_full_mask, crop_size, box):
    w, h = img.size
    _, cx, cy, bw, bh = box
    cx = int(cx * w)
    bw = int(bw * w)
    cy = int(cy * h)
    bh = int(bh * h)

    left = cx - crop_size // 2
    right = left + crop_size
    upper = cy - crop_size // 2
    lower = upper + crop_size
    if left < 0:
        right -= left
        left = 0
    if upper < 0:
        lower -= upper
        upper = 0
    if right > w:
        left -= (right - w)
        right = w
    if lower > h:
        upper -= (lower - h)
        lower = h

    crop_region = (left, upper, right, lower)

    crop = img.crop(crop_region)
    crop_mask = generate_crop_mask((w, h), crop_region, full_mask, crop_size)
    original_crop_mask = generate_crop_mask((w, h), crop_region, original_full_mask, crop_size, True)
    return crop, crop_mask, original_crop_mask

def random_pick_crop(img, full_mask, original_full_mask, crop_size):
    w, h = img.size
    max_x = w - crop_size
    max_y = h - crop_size

    left = random.randint(0, max_x - 1)
    upper = random.randint(0, max_y - 1)
    right = left + crop_size
    lower = upper + crop_size
    
    crop_region = (left, upper, right, lower)
    
    crop = img.crop(crop_region)
    crop_mask = generate_crop_mask((w, h), crop_region, full_mask, crop_size)
    original_crop_mask = generate_crop_mask((w, h), crop_region, original_full_mask, crop_size, True)
    return crop, crop_mask, original_crop_mask

    
# <object-class> <x> <y> <width> <height>
def fill_full_mask(mask, bs, R):
    mh, mw = mask.size()
    # print ('mh,mw', mh, mw)
    for obj in bs:
        cls_id = obj[0]
        cx, cy, bw, bh = obj[1:5]
        cx = cx - bw / 2    # move from center to top-left
        cy = cy - bh / 2
        d_cx = max(int(cx * mw), 0)
        d_bw = max(int(bw * mw), 0)
        d_cy = max(int(cy * mh), 0)
        d_bh = max(int(bh * mh), 0)
        # pdb.set_trace()
        # print (d_bh, d_bw)
        # print (d_cy,(d_cy+d_bh+1), d_cx,(d_cx+d_bw+1))
        mask[d_cy:(d_cy+d_bh+1), d_cx:(d_cx+d_bw+1)] = R
        
    return mask


class ImageCropWithBBoxMaskDataset(Dataset):
    def __init__(self, list_file, transform = None, crop_size = 320, contrastive_degree = 2, mse_bbox_weight = 2, train = True):
        self.imgs = open(list_file).readlines()
        self.transform = transform
        self.crop_size = crop_size
        self.R = contrastive_degree
        self.mse_bbox_weight = mse_bbox_weight
        self.train = train

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx].rstrip()
        # print (img_path)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        if 'testing' in img_path:
            label_path = join(TEST_LABEL_PATH, img_id) + '.txt'
        elif 'training' in img_path:
            label_path = join(TRAIN_GT_PATH, img_id) + '.txt'
        else:
            print ('Unknown source(train/test) of img!')
        img = Image.open(img_path)
        w, h = img.size
        if not self.train:
            if w % 8:
                w -= w % 8
            if h % 8:
                h -= h % 8
            img = img.crop((0,0,w,h))
        original_full_mask = th.ones((h,w))
        full_mask = th.ones((h//8, w//8))
        bs = np.array([])
        if os.path.getsize(label_path):
            bs = np.loadtxt(label_path)
        # else:
        #     print(label_path)
        if len(bs):
            bs = bs.reshape((-1, 5))
        num_objs = bs.shape[0]
        if num_objs:
            fill_full_mask(full_mask, bs, self.R)
            fill_full_mask(original_full_mask, bs, self.mse_bbox_weight)
        if not self.train:
            if self.transform:
                img = self.transform(img)
            return (img, full_mask, original_full_mask.unsqueeze(0))
        
        need_bbox_center_crop = num_objs and random.randint(0,10000) % 2 == 0
        # bbox center crop (must contain obj)
        if need_bbox_center_crop:
            # print ('center')
            crop, crop_mask, original_crop_mask = bbox_center_crop(img, full_mask, original_full_mask, self.crop_size, bs[random.randint(0, len(bs)-1)])
        # random crop (very possible background)
        else:
            # print ('random')
            crop, crop_mask, original_crop_mask = random_pick_crop(img, full_mask, original_full_mask, self.crop_size)
        if self.transform:
            crop = self.transform(crop)
        return (crop, crop_mask.unsqueeze(0), original_crop_mask.unsqueeze(0))
        
class ImageNet_200k(Dataset):

    def __init__(self, root_dir, transforms=None, train = True):
        '''
            root_dir: 图片文件夹
            transforms: 转换操作
            train:  True 训练集
                    False 验证集
        '''
        self.train = train
        self.imgs = [os.path.join(root_dir, img) for img in sorted(os.listdir(root_dir))
                        if os.path.isfile(os.path.join(root_dir, img)) and
                        has_file_allowed_extension(img, IMG_EXTENSIONS)]
        # print (self.imgs)
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # print (img_path)
        # sample = Image.open(img_path)
        sample = io.imread(img_path)
        if self.transforms:
            sample = self.transforms(sample)
        return (sample, img_path)


'''
class ImageCrops(Dataset):

    def __init__(self, root_dir, transforms=None, h, w = None):
            root_dir: 图片文件夹
            transforms: 转换操作
            train:  True 训练集
                    False 验证集
        if w is None:
            w = h
        self.imgs = [os.path.join(root_dir, img) for img in sorted(os.listdir(root_dir))
                        if os.path.isfile(os.path.join(root_dir, img)) and
                        has_file_allowed_extension(img, IMG_EXTENSIONS)]
        # print (self.imgs)
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # print (img_path)
        # sample = Image.open(img_path)
        sample = io.imread(img_path)
        if self.transforms:
            sample = self.transforms(sample)

        
        return (sample, img_path)
'''


def image_show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def test():
    val_dataset = ImageNet_200k("/share/Dataset/CLIC/val_test/valid/", None, False)
    val_loader = DataLoader(val_dataset,
                            batch_size = 4,
                            shuffle = True,
                            num_workers = 4
                            )
    print(type(val_loader))
    print("Iterable?", isinstance(val_loader, Iterable))
    print("Iterator?", isinstance(val_loader, Iterator))
    print("Total number of validation set is", len(val_dataset))
    image_show(val_dataset[0])
    print("Type of read image is",type(val_dataset[0]))
    I = np.asarray(val_dataset[0])
    print(I.shape)
    print(I[0,0,0])
    print(I.max())

def test2():
    test_dataset = CompressedFiles('/home/zhangwenqiang/jobs/pytorch_implement/image_codes_test')
    test_loader = DataLoader(test_dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 4
                            )
    
    print(test_dataset[0])

def test_fill_mask():
    img = Image.open('/home/snk/WindowsDisk/Download/KITTI/data_object_image_2/training/image_2/000000.png')
    w, h = img.size
    print (w, h)
    full_mask = th.ones((h//8, w//8))
    print (full_mask.size)
    original_full_mask = th.ones((h,w))
    bs = np.loadtxt('/home/snk/WindowsDisk/Download/KITTI/labels/000000.txt')
    if len(bs):
        bs = bs.reshape((-1, 5))
    num_objs = bs.shape[0]
    if num_objs:
        fill_full_mask(full_mask, bs, 233)
        fill_full_mask(original_full_mask, bs, 10)
    img.show()
    # plt.imshow(full_mask.numpy())
    plt.imshow(original_full_mask.numpy())
    plt.show()
    # pdb.set_trace()
    # print (full_mask)
    # plt.plot([1,2])
    # plt.imshow(full_mask.numpy())
    # crop, crop_mask = random_pick_crop(img, full_mask, 320)
    print (len(bs))
    rnd = random.randint(0, len(bs)-1)
    print (rnd)
    crop, crop_mask,original_crop_mask = bbox_center_crop(img, full_mask, original_full_mask, 320, bs[rnd])
    crop.show()
    plt.imshow(crop_mask.numpy())
    plt.show()

def test_crop_mask_dataset():
    ds = ImageCropWithBBoxMaskDataset('/home/snk/WindowsDisk/Download/KITTI/train.txt')
    # print (ds[0])
    img, crop, o_crop = ds[0]
    print (img)
    print (crop)
    img.show()
    plt.imshow(o_crop.squeeze(0))
    plt.show()
    
if __name__ == "__main__":
    # test_fill_mask()
    test_crop_mask_dataset()
