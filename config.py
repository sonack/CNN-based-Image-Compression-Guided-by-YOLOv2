#coding:utf8
from __future__ import print_function
import warnings
import getpass
import os

class DefaultConfig(object):
    GPU_HPC = (getpass.getuser() == 'zhangwenqiang')

    input_4_ch = False
    only_init_val = False # not train
    init_val = True
    # exp_desc = "pretrain_wo_impmap_128"
    # yolo rate loss and weighted mse loss
    # exp_desc = "yrl2_and_wml_r=0.2_gm=0.2"    
    # exp_desc = 'pretrain_w_impmap_64_r=0.2_gm=0.2'
    # exp_desc = "yrl2_nml_12"
    exp_desc = "wml_w=500_no_imp_4ch_pn0.5"
    # resume = ""
    # exp_desc = "yrl_noimp_w=50"


    dataset_enable_bbox_center_crop = False
# lr decay controlled by file created
    use_file_decay_lr = True
    lr_decay_file = "signal/lr_decay_" + exp_desc

# model
    model = "ContentWeightedCNN_YOLO"
    use_imp = False
    feat_num = 64  # defaut is 64

    contrastive_degree = 0  # yrlv2 required
    input_original_bbox_inner = 0.5
    input_original_bbox_outer = -0.5
    mse_bbox_weight = 250  # 25  1=original mse loss
    rate_loss_weight = 0.2
    rate_loss_threshold = 0.2      # 0.643  
# save path
    test_imgs_save_path = ("/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_" if not GPU_HPC else "/home/zhangwenqiang/jobs/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_") + exp_desc
    save_test_img = True
# datasets
    train_data_list = "/home/snk/WindowsDisk/Download/KITTI/traintest.txt" if not GPU_HPC else "/share/Dataset/KITTI/traintest.txt"
    val_data_list = "/home/snk/WindowsDisk/Download/KITTI/val.txt" if not GPU_HPC else "/share/Dataset/KITTI/val.txt"
# training
    batch_size = 32  # 128-16, 64-32 
    use_gpu = True
    num_workers = 8
    max_epoch = 150*3
    lr = 1e-6
    lr_decay = 0.1
    lr_anneal_epochs = 150
    use_early_adjust = False
    tolerant_max = 3
    weight_decay = 0
# display
    print_freq = 1
    eval_interval = 1
    save_interval = 1
    print_smooth = True

    plot_path = 'logs/plot/' + exp_desc
    log_path = 'logs/log/' 
# debug
    debug_file = "debug/info"
# finetune
    # resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/no_imp/Cmpr_yolo_no_imp_pretrain_wo_impmap_180_05-25_21:34:37.pth" 
    # resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/pretrained_w_imp_0.2/Cmpr_yolo_imp_pretrain_w_impmap_64_r=0.2_gm=0.2_cont_from_22_r=0.2_gama=0.2_94_05-26_18:11:40.pth"
    resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/Cmpr_yolo_no_imp_pretrain_wo_impmap_300_05-26_02:29:21.pth"
    # resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/Cmpr_yolo_imp_yrl2_and_wml_r=0.2_gm=0.2_r=0.2_gama=0.2_130_05-26_18:46:47.pth"
    # resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/Cmpr_yolo_imp_yrl2_and_wml_r=0.2_gm=0.2_wml=25_r=0.2_gama=0.2_87_05-26_19:10:29.pth"
    # resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/Cmpr_yolo_no_imp_yrl_and_wml_no_imp_weight=50_416_05-26_18:54:01.pth"
    # resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/Cmpr_yolo_no_imp_yrl_and_wml_no_imp_350_05-26_17:59:13.pth"
    # 64 no imp pretrain
    # resume = ""

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
            if not k.startswith('__') and k != 'parse':
                print(k,":",getattr(self, k))
        print('-' * 30)
        print('Good Luck!')
        
opt = DefaultConfig()
if not os.path.exists(opt.plot_path):
    print ('mkdir', opt.plot_path)
    os.makedirs(opt.plot_path)


if __name__ == '__main__':
    opt.parse()
