#coding:utf8
from __future__ import print_function
from config import opt
import os
import numpy as np
import math
import torch as t
import models

from dataset.dataset import ImageCropWithBBoxMaskDataset
from torch.utils.data import DataLoader

from torch.autograd import Variable

from utils import AverageValueMeter
from torchvision import transforms, datasets
import pdb
# import cv2

from utils.visualize import Visualizer, PlotSaver
from extend import RateLoss, LimuRateLoss
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import time
# from tqdm import tqdm

use_data_parallel = False
def multiple_gpu_process(model):
    global use_data_parallel
    print ('Process multiple GPUs...')
    if t.cuda.is_available():
        print ('Cuda enabled, use GPU instead of CPU.')
    else:
        print ('Cuda not enabled!')
        return model
    if t.cuda.device_count() > 1:
        print ('Use', t.cuda.device_count(),'GPUs.')
        model = t.nn.DataParallel(model).cuda()
        use_data_parallel = True
    else:
        print ('Use single GPU.')
        model.cuda()
    return model

def train(**kwargs):
    opt.parse(kwargs)
    # log file
    logfile_name = "Cmpr_with_YOLOv2_"+time.strftime("_%m_%d_%H:%M:%S")+".log.txt"
    ps = PlotSaver(logfile_name)


    # step1: Model
    model = getattr(models, opt.model)(use_imp = opt.use_imp, model_name="Cmpr_yolo_imp__r={r}_gama={w}".format(
                                                                r=opt.rate_loss_threshold, 
                                                                w=opt.rate_loss_weight)
                                                                if opt.use_imp else "Cmpr_yolo_no_imp")
    if opt.use_gpu:
        model = multiple_gpu_process(model)
    
    cudnn.benchmark = True

    # step2: Data
    normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
                )
    
    train_data_transforms = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),  TODO: try to reimplement by myself to simultaneous operate on label and data
            transforms.ToTensor(),
            normalize
        ]
    )
    val_data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize
        ]
    )
    train_data = ImageCropWithBBoxMaskDataset(
        opt.train_data_list,
        train_data_transforms,
        contrastive_degree = opt.contrastive_degree,
        mse_bbox_weight = opt.mse_bbox_weight
    )
    val_data = ImageCropWithBBoxMaskDataset(
        opt.val_data_list,
        val_data_transforms,
        contrastive_degree = opt.contrastive_degree,
        mse_bbox_weight = opt.mse_bbox_weight
    )
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    # step3: criterion and optimizer

    mse_loss = t.nn.MSELoss(size_average = False)
    def weighted_mse_loss(input, target, weight):
        return t.sum(weight * (input - target) ** 2)

    if opt.use_imp:
        # TODO: new rate loss
        rate_loss = RateLoss(opt.rate_loss_threshold, opt.rate_loss_weight)        
        # rate_loss = LimuRateLoss(opt.rate_loss_threshold, opt.rate_loss_weight)

    lr = opt.lr


    
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    start_epoch = 0
    decay_file_create_time = -1 # 为了避免同一个文件反复衰减学习率, 所以判断修改时间

    if opt.resume:
        if use_data_parallel:
            start_epoch = model.module.load(None if opt.finetune else optimizer, opt.resume, opt.finetune)
        else:
            start_epoch = model.load(None if opt.finetune else optimizer, opt.resume, opt.finetune)
            
        if opt.finetune:
            print ('Finetune from model checkpoint file', opt.resume)
        else:
            print ('Resume training from checkpoint file', opt.resume)
            print ('Continue training at epoch %d.' %  start_epoch)
    
    # step4: meters
    mse_loss_meter = AverageValueMeter()
    if opt.use_imp:
        rate_loss_meter = AverageValueMeter()
        rate_display_meter = AverageValueMeter()    
        total_loss_meter = AverageValueMeter()


    previous_loss = 1e100
    tolerant_now = 0
    same_lr_epoch = 0
    

    # ps init

    ps.new_plot('train mse loss', opt.print_freq, xlabel="iteration", ylabel="train_mse_loss")
    ps.new_plot('val mse loss', 1, xlabel="epoch", ylabel="val_mse_loss")
    if opt.use_imp:
        ps.new_plot('train rate value', opt.print_freq, xlabel="iteration", ylabel="train_rate_value")
        ps.new_plot('train rate loss',  opt.print_freq, xlabel="iteration", ylabel="train_rate_loss")
        ps.new_plot('train total loss', opt.print_freq, xlabel="iteration", ylabel="train_total_loss")
        ps.new_plot('val rate value', 1, xlabel="iteration", ylabel="val_rate_value")
        ps.new_plot('val rate loss', 1, xlabel="iteration", ylabel="val_rate_loss")
        ps.new_plot('val total loss', 1, xlabel="iteration", ylabel="val_total_loss")
   


    for epoch in range(start_epoch+1, opt.max_epoch+1):

        same_lr_epoch += 1
        # per epoch avg loss meter
        mse_loss_meter.reset()
        if opt.use_imp:
            rate_display_meter.reset()
            rate_loss_meter.reset()
            total_loss_meter.reset()
        else:
            total_loss_meter = mse_loss_meter
        # cur_epoch_loss refresh every epoch
        # vis.refresh_plot('cur epoch train mse loss')
        ps.new_plot("cur epoch train mse loss", opt.print_freq, xlabel="iteration in cur epoch", ylabel="train_mse_loss")
        # progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
        # progress_bar.set_description('epoch %d/%d, loss = 0.00' % (epoch, opt.max_epoch))
    

        # Init val
        if (epoch == start_epoch + 1) and opt.init_val:
            print ('Init validation ... ')
            if opt.use_imp:
                mse_val_loss, rate_val_loss, total_val_loss, rate_val_display = val(model, val_dataloader, weighted_mse_loss, rate_loss, ps)
            else:
                mse_val_loss = val(model, val_dataloader, weighted_mse_loss, None, ps)
        
            ps.add_point('val mse loss', mse_val_loss)
            if opt.use_imp:
                ps.add_point('val rate value', rate_val_display)
                ps.add_point('val rate loss', rate_val_loss)
                ps.add_point('val total loss', total_val_loss)
            
            ps.make_plot('val mse loss')

            if opt.use_imp:
                ps.make_plot('val rate value')
                ps.make_plot('val rate loss')
                ps.make_plot('val total loss')

            # log sth.
            if opt.use_imp:
                ps.log('Init Val @ Epoch:{epoch}, lr:{lr}, val_mse_loss: {val_mse_loss}, val_rate_loss: {val_rate_loss}, val_total_loss: {val_total_loss}, val_rate_display: {val_rate_display} '.format(
                    epoch = epoch,
                    lr = lr,
                    val_mse_loss = mse_val_loss,
                    val_rate_loss = rate_val_loss,
                    val_total_loss = total_val_loss,
                    val_rate_display = rate_val_display
                ))
            else:
                ps.log('Init Val @ Epoch:{epoch}, lr:{lr}, val_mse_loss:{val_mse_loss}'.format(
                    epoch = epoch,
                    lr = lr,
                    val_mse_loss = mse_val_loss
                ))

        model.train()
        # mask is the detection bounding box mask
        for idx, (data, mask, o_mask) in enumerate(train_dataloader):

            # pdb.set_trace()

            data = Variable(data)
            mask = Variable(mask)
            o_mask = Variable(o_mask)
            
            
            if opt.use_gpu:
                data = data.cuda(async = True)
                mask = mask.cuda(async = True)
                o_mask = o_mask.cuda(async = True)
            
            # pdb.set_trace()

            optimizer.zero_grad()
            reconstructed, imp_mask_sigmoid = model(data, o_mask, mask)

            # print ('imp_mask_height', model.imp_mask_height)
            # pdb.set_trace()

            # print ('type recons', type(reconstructed.data))
            loss = weighted_mse_loss(reconstructed, data, o_mask)
            # loss = mse_loss(reconstructed, data)
            caffe_loss = loss / (2 * opt.batch_size)
            
            if opt.use_imp:
                rate_loss_display = imp_mask_sigmoid
                rate_loss_ =  rate_loss(rate_loss_display)
                total_loss = caffe_loss + rate_loss_
            else:
                total_loss = caffe_loss
    
            total_loss.backward()
            optimizer.step()

            mse_loss_meter.add(caffe_loss.data[0])

            if opt.use_imp:
                rate_loss_meter.add(rate_loss_.data[0])
                rate_display_meter.add(rate_loss_display.data.mean())
                total_loss_meter.add(total_loss.data[0])


            if idx % opt.print_freq == opt.print_freq - 1:
                ps.add_point('train mse loss', mse_loss_meter.value()[0] if opt.print_smooth else caffe_loss.data[0])
                ps.add_point('cur epoch train mse loss', mse_loss_meter.value()[0] if opt.print_smooth else caffe_loss.data[0])
                if opt.use_imp:
                    ps.add_point('train rate value', rate_display_meter.value()[0] if opt.print_smooth else rate_loss_display.data.mean())
                    ps.add_point('train rate loss', rate_loss_meter.value()[0] if opt.print_smooth else rate_loss_.data[0])
                    ps.add_point('train total loss', total_loss_meter.value()[0] if opt.print_smooth else total_loss.data[0])
            
                if not opt.use_imp:
                    ps.log('Epoch %d/%d, Iter %d/%d, loss = %.2f, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader), total_loss_meter.value()[0], lr))
                else:
                    ps.log('Epoch %d/%d, Iter %d/%d, loss = %.2f, mse_loss = %.2f, rate_loss = %.2f, rate_display = %.2f, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader), total_loss_meter.value()[0], mse_loss_meter.value()[0], rate_loss_meter.value()[0], rate_display_meter.value()[0], lr))
                
                # 进入debug模式                
                if os.path.exists(opt.debug_file):
                    pdb.set_trace()
        
        if use_data_parallel:
            model.module.save(optimizer, epoch)
        else:
            model.save(optimizer, epoch)

        ps.make_plot('train mse loss')
        ps.make_plot('cur epoch train mse loss',epoch)
        if opt.use_imp:
            ps.make_plot("train rate value")
            ps.make_plot("train rate loss")
            ps.make_plot("train total loss")

        print ('Validating ...')

        # val 
        if opt.use_imp:
            mse_val_loss, rate_val_loss, total_val_loss, rate_val_display = val(model, val_dataloader, weighted_mse_loss, rate_loss, ps)
        else:
            mse_val_loss = val(model, val_dataloader, weighted_mse_loss, None, ps)
    
        ps.add_point('val mse loss', mse_val_loss)
        if opt.use_imp:
            ps.add_point('val rate value', rate_val_display)
            ps.add_point('val rate loss', rate_val_loss)
            ps.add_point('val total loss', total_val_loss)
        

        ps.make_plot('val mse loss')

        if opt.use_imp:
            ps.make_plot('val rate value')
            ps.make_plot('val rate loss')
            ps.make_plot('val total loss')

        # log sth.
        if opt.use_imp:
            ps.log('Epoch:{epoch}, lr:{lr}, train_mse_loss: {train_mse_loss}, train_rate_loss: {train_rate_loss}, train_total_loss: {train_total_loss}, train_rate_display: {train_rate_display} \n\
val_mse_loss: {val_mse_loss}, val_rate_loss: {val_rate_loss}, val_total_loss: {val_total_loss}, val_rate_display: {val_rate_display} '.format(
                epoch = epoch,
                lr = lr,
                train_mse_loss = mse_loss_meter.value()[0],
                train_rate_loss = rate_loss_meter.value()[0],
                train_total_loss = total_loss_meter.value()[0],
                train_rate_display = rate_display_meter.value()[0],

                val_mse_loss = mse_val_loss,
                val_rate_loss = rate_val_loss,
                val_total_loss = total_val_loss,
                val_rate_display = rate_val_display
            ))
        else:
            ps.log('Epoch:{epoch}, lr:{lr}, train_mse_loss:{train_mse_loss}, val_mse_loss:{val_mse_loss}'.format(
                epoch = epoch,
                lr = lr,
                train_mse_loss = mse_loss_meter.value()[0],
                val_mse_loss = mse_val_loss
            ))

        # Adaptive adjust lr
        # 每个lr，如果有opt.tolerant_max次比上次的val_loss还高， 
        # update learning rate
        # if loss_meter.value()[0] > previous_loss:
        if opt.use_early_adjust:
            if total_loss_meter.value()[0] > previous_loss:
                tolerant_now += 1
                if tolerant_now == opt.tolerant_max:
                    tolerant_now = 0
                    same_lr_epoch = 0
                    lr = lr * opt.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Due to early stop anneal lr to %.10f at epoch %d' % (lr, epoch))
                    ps.log ('Due to early stop anneal lr to %.10f at epoch %d' % (lr, epoch))

            else:
                tolerant_now -= 1
            
        if same_lr_epoch and same_lr_epoch % opt.lr_anneal_epochs == 0:
            same_lr_epoch = 0
            tolerant_now = 0
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print ('Anneal lr to %.10f at epoch %d due to full epochs.' % (lr, epoch))
            ps.log ('Anneal lr to %.10f at epoch %d due to full epochs.' % (lr, epoch))
        
        if opt.use_file_decay_lr and os.path.exists(opt.lr_decay_file):
            cur_mtime = os.path.getmtime(opt.lr_decay_file)
            if cur_mtime > decay_file_create_time:
                decay_file_create_time = cur_mtime
                lr = lr * opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print ('Anneal lr to %.10f at epoch %d due to decay-file indicator.' % (lr, epoch))
                ps.log ('Anneal lr to %.10f at epoch %d due to decay-file indicator.' % (lr, epoch))

        if previous_loss == 1e100:
            print ('Start training, please inspect log file %s!' % logfile_name)
        previous_loss = total_loss_meter.value()[0]

# TenCrop + Lambda
# ValueError: Expected 4D tensor as input, got 5D tensor instead.
# [torch.FloatTensor of size 96x10x3x224x224]
def val(model, dataloader, mse_loss, rate_loss, ps):
    if ps:
        ps.log ('Validating ... ')
    else:
        print ('Validating ... ')
    model.eval()    

    mse_loss_meter = AverageValueMeter()
    if opt.use_imp:
        rate_loss_meter = AverageValueMeter()
        rate_display_meter = AverageValueMeter()    
        total_loss_meter = AverageValueMeter()

    mse_loss_meter.reset()
    if opt.use_imp:
        rate_display_meter.reset()
        rate_loss_meter.reset()
        total_loss_meter.reset()

    for idx, (data, mask, o_mask) in enumerate(dataloader):
        # ps.log('%.0f%%' % (idx*100.0/len(dataloader)))
        val_data = Variable(data, volatile=True)
        val_mask = Variable(mask, volatile=True)
        val_o_mask = Variable(o_mask, volatile=True)

        if opt.use_gpu:
            val_data = val_data.cuda(async=True)
            val_mask = val_mask.cuda(async=True)
            val_o_mask = val_o_mask.cuda(async=True)

        reconstructed, imp_mask_sigmoid = model(val_data, val_mask)

        batch_loss = mse_loss(reconstructed, val_data, val_o_mask)
        batch_caffe_loss = batch_loss / (2 * opt.batch_size)

        if opt.use_imp and rate_loss:
            rate_loss_display = imp_mask_sigmoid
            rate_loss_value =  rate_loss(rate_loss_display)
            total_loss = batch_caffe_loss + rate_loss_value
    
        mse_loss_meter.add(batch_caffe_loss.data[0])

        if opt.use_imp:
            rate_loss_meter.add(rate_loss_value.data[0])
            rate_display_meter.add(rate_loss_display.data.mean())
            total_loss_meter.add(total_loss.data[0])

    if opt.use_imp:
        return mse_loss_meter.value()[0], rate_loss_meter.value()[0], total_loss_meter.value()[0], rate_display_meter.value()[0]
    else:
        return mse_loss_meter.value()[0]

# ''' used for just inference
def test(model, dataloader):
    model.eval()
    avg_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))


    revert_transforms = transforms.Compose([
        transforms.Normalize((-1,-1,-1),(2,2,2)),
        transforms.ToPILImage()
    ])

    mse = lambda x,y: (np.sum(np.square(y - x))) / float(x.size)
    psnr = lambda x,y: 10*math.log10(255. ** 2 / mse(x,y))


    evaluate = False
    mmse = 0
    mpsnr = 0
    mrate = 0
    for idx, (data, mask, o_mask) in progress_bar:
        test_data = Variable(data, volatile=True)
        test_mask = Variable(mask, volatile=True)
        test_o_mask = Variable(o_mask, volatile=True)
        # pdb.set_trace()
        if opt.use_gpu:
            test_data = test_data.cuda(async=True)
            test_mask = test_mask.cuda(async=True)
            test_o_mask = test_o_mask.cuda(async=True)
        
        reconstructed, imp_mask_sigmoid = model(test_data, test_mask)
        # clamp to [0.0,1.0]
        # print('min,max', reconstructed.min(), reconstructed.max())
        reconstructed = t.clamp(reconstructed, -1.0, 1.0)
        # print('after clamped, min,max', reconstructed.min(), reconstructed.max())


        if opt.use_imp:
            mrate += imp_mask_sigmoid.data.mean()
            imp_map = transforms.ToPILImage()(imp_mask_sigmoid.data.cpu()[0])
            imp_map = imp_map.resize((imp_map.size[0]*8, imp_map.size[1]*8))
            if opt.save_test_img:
                imp_map.save(os.path.join(opt.test_imgs_save_path, "%d_impMap.png" % idx))

        img_origin = revert_transforms(test_data.data.cpu()[0])
        # pdb.set_trace()
        if opt.save_test_img:
            img_origin.save(os.path.join(opt.test_imgs_save_path, "%d_origin.png" % idx))

        img_reconstructed = revert_transforms(reconstructed.data.cpu()[0])
        if opt.save_test_img:
            img_reconstructed.save(os.path.join(opt.test_imgs_save_path, "%d_reconst.png" % idx))
        if evaluate:
            origin_arr = np.array(img_origin)
            reconst_arr = np.array(img_reconstructed)
            
            mmse += mse(origin_arr, reconst_arr)
            mpsnr += psnr(origin_arr, reconst_arr)

    mmse /= len(progress_bar) * 1
    mpsnr /= len(progress_bar) * 1
    mrate /= len(progress_bar) * 1
    
    print ('avg mse = {mse}, avg psnr = {psnr}, avg rate = {rate}'.format(mse = mmse, psnr = mpsnr, rate = mrate))

def run_test():
    model = getattr(models, opt.model)(use_imp = opt.use_imp)
    if opt.use_gpu:
        model.cuda()  # ???? model.cuda() or model = model.cuda() all is OK
    

    test_ckpt = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/checkpoints/Cmpr_yolo_imp__r=0.122_gama=0.2/05-24/Cmpr_yolo_imp__r=0.122_gama=0.2_61_05-24_14:35:40.pth"


    model.load(None, test_ckpt)

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )

    run_test_data_list = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/my_test.txt"
    test_data = ImageCropWithBBoxMaskDataset(run_test_data_list, test_transforms, contrastive_degree=opt.contrastive_degree, train=False)
    test_dataloader = DataLoader(test_data, 1, shuffle = False)
    test(model, test_dataloader)
# '''



''' used for explicitly run val()
def run_val():
    model = getattr(models, opt.model)(use_imp = opt.use_imp)
    if opt.use_gpu:
        model.cuda()  # ???? model.cuda() or model = model.cuda() all is OK
    model.load('checkpoints/ContentWeightedCNN_0309_08:57:47.pth')
    # print (list(model.parameters())[0])

    val_transforms = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )
    val_data = ImageNet_200k(opt.val_data_root, train = False, transforms=val_transforms)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    val(model, val_dataloader)
'''




if __name__ == '__main__':
    import fire
    fire.Fire() 
    # import sys
    # assert len(sys.argv) == 2, 'one for filename, one for function'
    # if sys.argv[1] == 'train':
    #     train()
