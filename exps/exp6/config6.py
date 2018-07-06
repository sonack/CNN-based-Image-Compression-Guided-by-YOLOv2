#coding:utf8
from __future__ import print_function
import warnings
import getpass
import os

# ----------------------------------------------
# exp_id = 1,   pretrain_wo_imp
# exp_id = 2,   'train_w_imp_gamma=0.2_r=0.19',
# exp_id = 3,   'train_w_imp_gamma=0.2_r=0.26',
# exp_id = 4,   'train_w_imp_gamma=0.2_r=0.43',
# exp_id = 5,   'train_w_imp_gamma=0.2_r=0.66'


# Imagenet 10K, reproduce limu's paper
# exp_id = 6,   'pretrain_wo_imp_imagenet_10k'

# ----------------------------------------------






class DefaultConfig(object):
# judge environment according username
    GPU_HPC = (getpass.getuser() == 'zhangwenqiang')

# exp_description lookup table
    exp_desc_LUT = [
                '',
                'pretrain_wo_imp',
                'train_w_imp_gamma=0.2_r=0.19',
                'train_w_imp_gamma=0.2_r=0.26',
                'train_w_imp_gamma=0.2_r=0.43',
                'train_w_imp_gamma=0.2_r=0.66',
                'pretrain_wo_imp_imagenet_10k'
    ]
        
# 批处理
    use_batch_process = True
    # rate_loss_threshold
    r_s = [0.19, 0.26, 0.43, 0.66]
    # r_s = [0.26, 0.43, 0.66]

    # exp_ids = [2,3,4,5]
    exp_ids = [6]

    # max_epochs = [200*3] * 4  
    max_epochs = [300*3]  # 注意lr_anneal_epochs是单独设置的!!!


    ####################


# make caffe dataset(imgs) to cmpr with caffe model
    make_caffe_data = False
    caffe_data_save_dir = "test_imgs_caffe"

# Init val for val_subset or test_subset
    init_val = True
    only_init_val = False # not train, only eval
    test_test = False # 用val的方式来test_test  use run_val
                      # 暂时对HPC无效， 因为HPC还没有Kodak数据集，且eval不需要用到HPC
     
    # exp_desc = "pretrain_wo_impmap_128"
    # yolo rate loss and weighted mse loss
    # exp_desc = "yrl2_and_wml_r=0.2_gm=0.2"    
    # exp_desc = 'pretrain_w_impmap_64_r=0.2_gm=0.2'
    # exp_desc = "yrl2_nml_12"
    # exp_desc = "wml_w=500_no_imp_4ch_pn0.5"
    
    exp_id = 1
    exp_desc = exp_desc_LUT[exp_id]


# KITTI model options
    input_4_ch = False
# KITTI detection crop options
    dataset_enable_bbox_center_crop = False


# model
    model = "ContentWeightedCNN"
    use_imp = False
    feat_num = 64  # defaut is 64

    # contrastive_degree = 0  # yrlv2 requires
    # 4-ch input requires
    input_original_bbox_inner = 25
    input_original_bbox_outer = 1

    # weighted mse
    mse_bbox_weight = 5

    # rate loss
    rate_loss_weight = 0.2
    rate_loss_threshold = 0.12      # 0.12 | 0.17 | 0.32 | 0.49 |


# save path
    # test_imgs_save_path = ("/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_" if not GPU_HPC else "/home/zhangwenqiang/jobs/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_") + exp_desc
    save_test_img = False
    test_imgs_save_path = "test_imgs_saved/"
    

# datasets
    # local and HPC dataset root

    use_imagenet = True
    local_ds_root = "/home/snk/WindowsDisk/DataSets/ImageNet/ImageNet_10K/" if use_imagenet else "/home/snk/WindowsDisk/Download/KITTI/cmpr_datasets/"
    hpc_ds_root = "/share/Dataset/ILSVRC12/ImageNet_10K/" if use_imagenet else "/share/Dataset/KITTI/cmpr_datasets/"

    # train, val and test data filelist
    train_data_list = os.path.join(local_ds_root,"traintest.txt") if not GPU_HPC else os.path.join(hpc_ds_root, "traintest.txt")
    # val_data_list = os.path.join(local_ds_root,"val_subset.txt") if not GPU_HPC else os.path.join(hpc_ds_root, "val.txt")
    
    # KITTI test
    # test_data_list = os.path.join(local_ds_root,"test_subset.txt") if not GPU_HPC else os.path.join(hpc_ds_root,"test.txt") 
    
    # Kodak test
    test_data_list = "/home/snk/Desktop/总结/codes/CNN-based-Image-Compression-Guided-by-YOLOv2/caffe_model_cmp/ctifl.txt" 
    
    val_data_list = (test_data_list if test_test else os.path.join(local_ds_root,"val_subset.txt")) if not GPU_HPC else os.path.join(hpc_ds_root, "val_subset.txt") # 利用InitVal来测试Val集和Test集

# training

    # base
    use_gpu = True
    num_workers = 8

    # basic
    batch_size = 32 # for train and val
    max_epoch = 200*3

    # lr
    lr = 1e-4
    lr_decay = 0.1
    lr_anneal_epochs = 300

    # lr decay controlled by file created
    use_file_decay_lr = True
    lr_decay_file = "signal/lr_decay_%d" % exp_id # 不适用batch process

    # auto early adjust lr
    use_early_adjust = False
    tolerant_max = 3

    # regularization
    weight_decay = 0

# display
    print_freq = 1 # by iteration
    print_smooth = True
    plot_path = 'plot'
    log_path = 'log'

    # interval
    eval_interval = 1 # by epoch
    save_interval = 15 # by epoch
   
# debug
    debug_file = "debug/info"

# finetune
    # resume = "/home/snk/Desktop/总结/codes/CNN-based-Image-Compression-Guided-by-YOLOv2/exps/exp1/checkpoints/pretrain_wo_imp_no_imp/06-21/pretrain_wo_imp_no_imp_600_06-21_03:30:17.pth" \
    #             if not GPU_HPC else "/home/zhangwenqiang/jobs/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/exp1/pretrain_wo_imp_no_imp_600_06-21_03_30_17.pth"
    resume = "/home/zhangwenqiang/jobs/CNN-based-Image-Compression-Guided-by-YOLOv2/exps/exp6/checkpoints/pretrain_wo_imp_imagenet_10k/07-06/pretrain_wo_imp_imagenet_10k_195_07-06_20:53:45.pth"
    finetune = False  # continue training or finetune when given a resume file

# ---------------------------------------------------------
    def __getattr__(self, attr_name):
        return None
    
    def parse(self, kwargs={}):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
                print ("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        
        print ('\n')
        print ('*' * 30)
        print('User Config:\n')
        # print('-' * 30)
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse' and k != 'make_new_dirs':
                print(k,":",getattr(self, k))

        print('Good Luck!')

    

    def make_new_dirs(self):
        if not os.path.exists(self.plot_path):
            print ('mkdir', self.plot_path)
            os.makedirs(self.plot_path)

        if not os.path.exists(self.log_path):
            print ('mkdir', self.log_path)
            os.makedirs(self.log_path)    
        
opt = DefaultConfig()

if __name__ == '__main__':
    opt.parse()
