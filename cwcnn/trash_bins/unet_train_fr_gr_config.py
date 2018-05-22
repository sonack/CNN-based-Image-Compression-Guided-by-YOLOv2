#coding:utf8
from __future__ import print_function
import warnings
import os

class DefaultConfig(object):

# patch params
    patch_size = 128  # python main.py train --patch_size=16
    init_val = True

# freeze decoder
    freeze_decoder = False
    freeze_encoder = True
# gradual schema
    use_grad_mode = True
    alpha_decay_factor = 0.1
    alpha_decay_epochs = 15

# lr decay controlled by file created
    use_file_decay_lr = False
    lr_decay_file = "signal/lr_decay"

# inf config
    save_inf_img = False
    inf_imgs_save_path = "./inf/imgs"
    inf_data_root = "/share/Dataset/CLIC/test/all"  # MyImageFolder

# model
    model = "ContentWeightedCNN_UNET"  # change from ContentWeightedCNN to ContentWeightedCNN_UNET
    t_model = "ContentWeightedCNN_UNET"
    # epoch = 120
    t_ckpt = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/UNet_Teacher_Network_p128/05-04/UNet_Teacher_Network_p128_120_05-04_19:00:41.pth"
    
    use_imp = False
    use_unet = False

    rate_loss_weight = 0.2
    rate_loss_threshold = 0.122      # 0.643

    unet_loss_weight = 0.02 * (0.1 ** 6)   # alpha init const = 0.02
# E->D save path
    save_test_img = False
    test_imgs_save_path = "./test/imgs"

# datasets
    # ImageFolder
    train_data_root = "/share/Dataset/flickr/flickr_train"
    val_data_root = "/share/Dataset/flickr/flickr_val"
# training
    batch_size = 8   # 7805 / 64 = 122   # change from 64 to 32 to 8
    use_gpu = True
    num_workers = 8
    max_epoch = 30 * 3 + 90  # 90 is the continue point
    lr = 1e-6
    lr_decay = 0.1
    lr_anneal_epochs = 450  # use 1e-5 -> 1e-6 (change when half)

    use_early_adjust = False
    tolerant_max = 3

    weight_decay = 0

# display
    # env = 'debug_pytorch_issue' # visdom环境  test for imp, main for no imp
    print_freq = 15 # 157 -> 30 -> 1 -> 1000
    print_smooth = True
    plot_interval = 15
    path_suffix = "UNet_Fr_Gr_p{ps}_w{w}/".format(ps=patch_size, w=unet_loss_weight)  # noci  no_context_information
    exp_id = "exp_unet_fr_gr"
    plot_path = os.path.join('/home/zhangwenqiang/jobs/pytorch_implement/logs/{exp_id}/plots/'.format(exp_id=exp_id), path_suffix)
    log_path = os.path.join('/home/zhangwenqiang/jobs/pytorch_implement/logs/{exp_id}/texts/'.format(exp_id=exp_id), path_suffix)
# debug
    debug_file = "debug/info"
# finetune
# because we want to start a no unet loss trained model ckpt
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/TestPatchSizeNoImpP32_Plain/04-28/TestPatchSizeNoImpP32_Plain_6_04-28_16:13:12.pth"
    # resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/UNet_BaseStudent_Network_btsz=8_p128/05-06/UNet_BaseStudent_Network_btsz=8_p128_90_05-06_20:42:16.pth"
    # finetune = True  # continue training or finetune when given a resume file

    resume = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/UNet_Fr_Gr_p128_w0.02/05-08/UNet_Fr_Gr_p128_w0.02_90_05-08_18:52:57.pth"
    finetune = False

# run val
    val_ckpt = "/home/zhangwenqiang/jobs/pytorch_implement/checkpoints/ContentWeightedCNN_ImageNet/03-23/ContentWeightedCNN_ImageNet_22_03-23_04:54:24.pth"
    run_val_data_root = "/share/Dataset/ILSVRC12/debug_data/val/"
# show inf image
    show_inf_imgs_T = 1
    save_inf_imgs_path = os.path.join(plot_path, 'inf_imgs')






# ---------------------------------------------------------
    def __getattr__(self, attr_name):
        return None
    
    def parse(self, kwargs={}):
        # print ('parse kwargs', kwargs)
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
                print ("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
            if k == "patch_size" and v != self.patch_size:
                setattr(self, "path_suffix", "TPSE_p{ps}/".format(ps=v))
                setattr(self, "plot_path", os.path.join('/home/zhangwenqiang/jobs/pytorch_implement/logs/{exp_id}/plots/'.format(exp_id=self.exp_id), self.path_suffix))
                setattr(self, "log_path", os.path.join('/home/zhangwenqiang/jobs/pytorch_implement/logs/{exp_id}/texts/'.format(exp_id=self.exp_id), self.path_suffix))
                setattr(self, "save_inf_imgs_path", os.path.join(self.plot_path, 'inf_imgs'))
        print('\nUser Config:\n')
        print('*' * 30)
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse':
                print(k,":",getattr(self, k))
        print('*' * 30)
        print('Good Luck!')

        # create plot and text path
        if not os.path.exists(opt.plot_path):
            print ('mkdir', opt.plot_path)
            os.makedirs(opt.plot_path)

        if not os.path.exists(opt.log_path):
            print ('mkdir', opt.log_path)
            os.makedirs(opt.log_path)

        if not os.path.exists(opt.save_inf_imgs_path):
            print ('mkdir', opt.save_inf_imgs_path)
            os.makedirs(opt.save_inf_imgs_path)
        
 
opt = DefaultConfig()




if __name__ == '__main__':
    opt.parse()
