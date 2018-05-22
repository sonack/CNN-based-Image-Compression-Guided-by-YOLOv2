#coding:utf8
from __future__ import print_function
import warnings
import os

class DefaultConfig(object):

# patch params
    patch_size = 32

# freeze decoder
    init_val = False
    freeze_decoder = False
# lr decay controlled by file created
    use_file_decay_lr = True
    lr_decay_file = "signal/lr_decay"
# inf config
    inf_imgs_save_path = "./inf/imgs"
    save_inf_img = False
    # inf_data_root = "/share/Dataset/CLIC/val/valid/"
    # MyImageFolder
    inf_data_root = "/share/Dataset/CLIC/test/all"
    # inf_data_root = "./test_code_images"
# model
    model = "ContentWeightedCNN"
    use_imp = False

    rate_loss_weight = 0.2
    rate_loss_threshold = 0.122      # 0.643  
# save path
    test_imgs_save_path = "./test/imgs"
    save_test_img = False
# datasets
    #train_data_root = "/share/Dataset/CLIC/val"


    # train_data_root = "/share/Dataset/CLIC/val"
    # val_data_root = "/share/Dataset/CLIC/val"
    
    # ImageFolder
    # train_data_root = "/share/Dataset/CLIC/test"
    # val_data_root = "/share/Dataset/CLIC/test"

    train_data_root = "/share/Dataset/flickr/flickr_train"
    val_data_root = "/share/Dataset/flickr/flickr_val"
    # train_data_root = "/share/Dataset/ILSVRC12/train"
    # val_data_root = "/share/Dataset/ILSVRC12/val"
    # train_data_root = "/share/Dataset/ILSVRC12/test_wrapper/val"
    # val_data_root = "/share/Dataset/ILSVRC12/test_wrapper/val"
# training
    batch_size = 96
    use_gpu = True
    num_workers = 8
    max_epoch = 30 * 3
    lr = 1e-4
    lr_decay = 0.1
    lr_anneal_epochs = 3000   # don't use it, so I set it as 3000
    use_early_adjust = False  # don't use it, because its cost is too expensive!!!
    tolerant_max = 3
    weight_decay = 0
# display
    # env = 'debug_pytorch_issue' # visdom环境  test for imp, main for no imp
    print_freq = 1000 # 157 -> 30 -> 1 -> 1000
    print_smooth = True
    plot_interval = 1000
    path_suffix = "Context_no_imp_p32_noci_1/"  # noci  no_context_information
    plot_path = '/home/zhangwenqiang/jobs/pytorch_implement/logs/plots/' + path_suffix
    log_path = '/home/zhangwenqiang/jobs/pytorch_implement/logs/texts/' + path_suffix
# debug
    debug_file = "debug/info"
# finetune
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/ContentWeightedCNN_ImageNet/03-21/ContentWeightedCNN_ImageNet_6_03-21_01:36:35.pth"
    # resume from epoch 18, lr from 1e-4 -> 1e-5
    # resume ="/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/ContentWeightedCNN_ImageNet/03-22/ContentWeightedCNN_ImageNet_18_03-22_17:07:34.pth"
    # resume from epoch 22, lr from 1e-5 -> 1e-6
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/ContentWeightedCNN_ImageNet/03-23/ContentWeightedCNN_ImageNet_22_03-23_04:54:24.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/ContentWeightedCNN/03-20/ContentWeightedCNN_2_03-20_12:34:08.pth"
    # resume = "/home/snk/Desktop/workspace/pytorch_implement/checkpoints/ContentWeightedCNN/03-11/ContentWeightedCNN_30_03-11_03:50:28.pth" 
    # resume = "/home/snk/Desktop/workspace/pytorch_implement/checkpoints/CWCNN_imp_r=0.5_γ=0.2/03-11/CWCNN_imp_r=0.5_γ=0.2_17_03-11_23:59:47.pth"
    
    # CLIC fine-tune from epoch 42
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/ContentWeightedCNN_ImageNet/03-24/ContentWeightedCNN_ImageNet_42_03-24_14:45:27.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/CLIC_imp_r=0.1_γ=0.2/04-03/CLIC_imp_r=0.1_γ=0.2_25_04-03_08:52:03.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/CLIC_imp_ft_r=0.13_gama=0.2/04-11/CLIC_imp_ft_r=0.13_gama=0.2_21_04-11_15:27:04.pth"

    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/CLIC_imp_ft_r=0.13_gama=0.2/04-11/CLIC_imp_ft_r=0.13_gama=0.2_18_04-11_12:28:21.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/CLIC_imp_ft_train_r=0.13_gama=0.2/04-13/CLIC_imp_ft_train_r=0.13_gama=0.2_16_04-13_15:34:05.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/CLIC_imp_ft_all_r=0.125_gama=0.2/04-13/CLIC_imp_ft_all_r=0.125_gama=0.2_11_04-13_19:08:14.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/CLIC_imp_ft_all_2_r=0.125_gama=0.2/04-13/CLIC_imp_ft_all_2_r=0.125_gama=0.2_16_04-13_19:33:28.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/CLIC_imp_ft_all_3_r=0.125_gama=0.2/04-13/CLIC_imp_ft_all_3_r=0.125_gama=0.2_23_04-13_19:57:57.pth"
    finetune = True  # continue training or finetune when given a resume file

# run val
    val_ckpt = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/ContentWeightedCNN_ImageNet/03-23/ContentWeightedCNN_ImageNet_22_03-23_04:54:24.pth"
    run_val_data_root = "/share/Dataset/ILSVRC12/debug_data/val/"






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
    os.mkdir(opt.plot_path)

if not os.path.exists(opt.log_path):
    print ('mkdir', opt.log_path)
    os.mkdir(opt.log_path) 

  


if __name__ == '__main__':
    opt.parse()
