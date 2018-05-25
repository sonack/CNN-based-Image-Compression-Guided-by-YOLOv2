#coding:utf8
from __future__ import print_function
import warnings

class DefaultConfig(object):
    GPU_HPC = True
    init_val = False
    exp_desc = "pretrain_wo_impmap"
# lr decay controlled by file created
    use_file_decay_lr = True
    lr_decay_file = "signal/lr_decay"

# model
    model = "ContentWeightedCNN_YOLO"
    use_imp = False

    contrastive_degree = 4
    mse_bbox_weight = 5
    rate_loss_weight = 0.1
    rate_loss_threshold = 0.2      # 0.643  
# save path
    test_imgs_save_path = ("/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_" if not GPU_HPC else "/home/zhangwenqiang/jobs/CNN-based-Image-Compression-Guided-by-YOLOv2/logs/test_imgs_") + exp_desc
    save_test_img = True
# datasets
    train_data_list = "/home/snk/WindowsDisk/Download/KITTI/train.txt" if not GPU_HPC else "/share/Dataset/KITTI/train.txt"
    val_data_list = "/home/snk/WindowsDisk/Download/KITTI/test.txt" if not GPU_HPC else "/share/Dataset/KITTI/test.txt"
# training
    batch_size = 32
    use_gpu = True
    num_workers = 8
    max_epoch = 30*3
    lr = 1e-4
    lr_decay = 0.1
    lr_anneal_epochs = 30
    use_early_adjust = False
    tolerant_max = 3
    weight_decay = 0
# display
    print_freq = 10
    eval_interval = 1
    print_smooth = True

    plot_path = 'logs/plot/'
    log_path = 'logs/log/'
# debug
    debug_file = "debug/info"
# finetune
    # resume = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/Cmpr_yolo_imp__r=0.01_gama=0.02/05-24/Cmpr_yolo_imp__r=0.01_gama=0.02_31_05-24_16:12:27.pth"
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


if __name__ == '__main__':
    opt.parse()
