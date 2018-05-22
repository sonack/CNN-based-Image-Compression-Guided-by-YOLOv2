#coding: utf-8
import os
import shutil
import random

'''
划分train和val set
'''

# train_path = "/home/snk/WindowsDisk/DataSets/old-datasets/Data/flickr/flick/"
# val_path = "/home/snk/WindowsDisk/DataSets/old-datasets/Data/flickr/flickr_val/"

train_path = "/share/Dataset/flickr"
val_path= "/share/Dataset/flickr_val"


def random_pick(s, t, p):
    for img in os.listdir(s):
        print (img)
        if random.random() < p:
            print ('Move img %s of %s into val.' % (img,s))
            s_img = os.path.join(s, img)
            t_img = os.path.join(t, img)
            shutil.move(s_img, t_img)
    
for class_dir in os.listdir(train_path):
    origin_dir = os.path.join(train_path, class_dir)
    if not os.path.isdir(origin_dir):
        continue
    
    print ("Processing", origin_dir)
    target_dir = os.path.join(val_path, class_dir)
    if not os.path.exists(target_dir):
        print ('Mkdir', target_dir)
        os.mkdir(target_dir)
    random_pick(s=origin_dir, t=target_dir, p=0.2)
    


# flickr 9726 (include subdir and itself)
# 3837 large
# 3213 medium
# 2675 small


'''
flickr val 2023
787 large
664 medium
571 small
'''
