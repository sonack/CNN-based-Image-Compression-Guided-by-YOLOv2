#coding:utf8
from __future__ import print_function
import warnings
import getpass
import os

# ----------------------------------------------
# exp_id = 1,   pretrain_wo_imp
# LUT = Lookup Table





class DefaultConfig(object):
    GPU_HPC = (getpass.getuser() == 'zhangwenqiang')

    exp_desc_LUT = [
                '',
                'pretrain_wo_imp',
                'train_w_imp_gamma=0.2_r=0.19',
                'train_w_imp_gamma=0.2_r=0.26',
                'train_w_imp_gamma=0.2_r=0.43',
                'train_w_imp_gamma=0.2_r=0.66'
    ]
        
    # 批处理
    use_batch_process = True
    r_s = [0.19, 0.26, 0.43, 0.66]
    exp_ids = [2,3,4,5]
    ####################

    make_caffe_data = False
    caffe_data_save_dir = "test_imgs_caffe"


    input_4_ch = False
    only_init_val = False # not train, only eval
    init_val = True

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
    # resume = ""
    # exp_desc = "yrl_noimp_w=50"


    dataset_enable_bbox_center_crop = False
# lr decay controlled by file created
    use_file_decay_lr = True
    lr_decay_file = "signal/lr_decay_" + exp_desc

# model
    model = "ContentWeightedCNN"
    use_imp = True
    feat_num = 64  # defaut is 64

    # contrastive_degree = 0  # yrlv2 required
    input_original_bbox_inner = 25
    input_original_bbox_outer = 1

    mse_bbox_weight = 5
    rate_loss_weight = 0.2
    rate_loss_threshold = 0.12      # 0.12 | 0.17 | 0.32 | 0.49 |


# save path
    # test_imgs_save_path = ("/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_" if not GPU_HPC else "/home/zhangwenqiang/jobs/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_") + exp_desc
    test_imgs_save_path = "test_imgs_saved/"
    save_test_img = False

# datasets
    local_ds_root = "/home/snk/WindowsDisk/Download/KITTI/cmpr_datasets/"
    hpc_ds_root = "/share/Dataset/KITTI/cmpr_datasets/"
    train_data_list = os.path.join(local_ds_root,"traintest.txt") if not GPU_HPC else os.path.join(hpc_ds_root, "traintest.txt")
    # val_data_list = os.path.join(local_ds_root,"val_subset.txt") if not GPU_HPC else os.path.join(hpc_ds_root, "val.txt")
    # test_data_list = os.path.join(local_ds_root,"test_subset.txt") if not GPU_HPC else os.path.join(hpc_ds_root,"test.txt") 
    test_data_list = "/home/snk/Desktop/总结/codes/CNN-based-Image-Compression-Guided-by-YOLOv2/caffe_model_cmp/ctifl.txt" # Kodak
    val_data_list = (test_data_list if test_test else os.path.join(local_ds_root,"val_subset.txt")) if not GPU_HPC else os.path.join(hpc_ds_root, "val_subset.txt") # 利用InitVal来测试Val集和Test集

# training
    batch_size = 32 # for train and val
    use_gpu = True
    num_workers = 8
    max_epoch = 200*3
    lr = 1e-4
    lr_decay = 0.1
    lr_anneal_epochs = 200
    use_early_adjust = False
    tolerant_max = 3
    weight_decay = 0
# display
    print_freq = 1 # by iteration
    eval_interval = 1 # by epoch
    save_interval = 10 # by epoch
    print_smooth = True

    plot_path = 'plot'
    log_path = 'log'
# debug
    debug_file = "debug/info"
# finetune
    resume = "/home/snk/Desktop/总结/codes/CNN-based-Image-Compression-Guided-by-YOLOv2/exps/exp1/checkpoints/pretrain_wo_imp_no_imp/06-21/pretrain_wo_imp_no_imp_600_06-21_03:30:17.pth" \
                if not GPU_HPC else "/home/zhangwenqiang/jobs/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/exp1/pretrain_wo_imp_no_imp_600_06-21_03_30_17.pth"

    finetune = True  # continue training or finetune when given a resume file

# ---------------------------------------------------------
    def __getattr__(self, attr_name):
        return None
    
    def parse(self, kwargs={}):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
                print ("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        
        print('User Config:\n')
        print('-' * 30)
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse' and k != 'make_new_dirs':
                print(k,":",getattr(self, k))
        print('-' * 30)
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
