#coding: utf-8
from __future__ import print_function, division
import os
import torch as th
import numpy as np
# from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from collections import Iterable, Iterator
import matplotlib.pyplot as plt
import sys; sys.path.append('/home/zhangwenqiang/jobs/pytorch_implement')
from utils import Serializer

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.ras']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

class CompressedFiles(Dataset):
    def __init__(self,root_dir):
        self.imgs = [os.path.join(root_dir, img) for img in sorted(os.listdir(root_dir))
                    if os.path.isfile(os.path.join(root_dir, img)) and
                    img.endswith('cnn')]
        self.serializer = Serializer()
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        print ('parsing', img_path)
        data = self.serializer.parse(img_path)
        # print ("SIZE", data.size())
        return (data, img_path)
        
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

if __name__ == "__main__":
    test2()
