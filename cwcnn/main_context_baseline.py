#coding:utf8
from __future__ import print_function
from context_config import opt
import os
import numpy as np
import math
import torch as t
import models

# from data.dataset import ImageNet_200k
from torch.utils.data import DataLoader

from torch.autograd import Variable


from utils import AverageValueMeter


import torchvision as tv
from torchvision import transforms, datasets
import pdb

from utils.visualize import Visualizer, PlotSaver
from extend import RateLoss, LimuRateLoss
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F

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
    ps = PlotSaver("Context_Baseline_p32_no_imp_plain_1_"+time.strftime("%m_%d_%H:%M:%S")+".log.txt")


    # step1: Model
    model = getattr(models, opt.model)(use_imp = opt.use_imp, model_name="Context_Baseline_p32_imp_r={r}_gama={w}_plain".format(
                                                                r=opt.rate_loss_threshold, 
                                                                w=opt.rate_loss_weight)
                                                                if opt.use_imp else "ContextBaseNoImpP32_Plain")

    if opt.use_gpu:
        model = multiple_gpu_process(model)
    
    cudnn.benchmark = True


    normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
                )

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize
        ]
    )
    train_data = datasets.ImageFolder(
        opt.train_data_root,
        data_transforms
    )
    val_data = datasets.ImageFolder(
        opt.val_data_root,
        data_transforms
    )
    # opt.batch_size  --> 1
    train_dataloader = DataLoader(train_data, 1, shuffle=True, num_workers=opt.num_workers, pin_memory=True)    
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # step3: criterion and optimizer

    mse_loss = t.nn.MSELoss(size_average = False)

    if opt.use_imp:    
        rate_loss = LimuRateLoss(opt.rate_loss_threshold, opt.rate_loss_weight)

    lr = opt.lr

   
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


    start_epoch = 0

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
        
        ps.new_plot("cur epoch train mse loss", opt.print_freq, xlabel="iteration in cur epoch", ylabel="train_mse_loss")
      
        # Init val
        if (epoch == start_epoch + 1) and opt.init_val:
            print ('Init validation ... ')
            if opt.use_imp:
                mse_val_loss, rate_val_loss, total_val_loss, rate_val_display = val(model, val_dataloader, mse_loss, rate_loss, ps)
            else:
                mse_val_loss = val(model, val_dataloader, mse_loss, None, ps)
        
            ps.add_point('val mse loss', mse_val_loss)
            if opt.use_imp:
                ps.add_point('val rate value', rate_val_display)
                ps.add_point('val rate loss', rate_val_loss)
                ps.add_point('val total loss', total_val_loss)
            

            # make plot
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
                    # train_mse_loss = mse_loss_meter.value()[0],
                    val_mse_loss = mse_val_loss
                ))

        model.train()

        

        def to_patches(x, patch_size):
            print (type(x))
            _, h, w = x.size()
            print ('original h,w', h, w)
            dw = (patch_size - w % patch_size) if w % patch_size else 0
            dh = (patch_size - h % patch_size) if h % patch_size else 0

            x = F.pad(x, (dw//2, dw-dw//2, dh//2, dh-dh//2))
            

            _, h, w = x.size()
            print (h, w)
            pdb.set_trace()

            tv.utils.save_image(x.data, 'test_images/padded_original_img.png')


            num_patch_x = w // patch_size
            num_patch_y = h // patch_size
            print (num_patch_x, num_patch_y)
            patches = []
            for i in range(num_patch_y):
                for j in range(num_patch_x):
                    patch = x[:, i*patch_size : (i+1)*patch_size, j*patch_size : (j+1)*patch_size]
                    patches.append(patch.contiguous())
                if (j+1)*patch_size < w:
                    extra_patch = x[:, i*patch_size : (i+1)*patch_size, (j+1)*patch_size : w]
                    extra_patch

            return patches


        # _ is corresponding Label， compression doesn't use it.
        for idx, (data, _) in enumerate(train_dataloader):

            if idx == 0:
                print ('skip idx =', idx)
                continue
            # ipt = Variable(data[0])
            ipt = data[0]
            # if opt.use_gpu:
            #     # if not use_data_parallel:  # because ipt is also target, so we still need to transfer it to CUDA
            #         ipt = ipt.cuda(async = True)


            # ipt is a full image, so I need to split it into crops
            # so set batch_size = 1 for simplicity at first

            # pdb.set_trace()

            tv.utils.save_image(ipt, "test_imgs/original.png")
            patches = to_patches(ipt, opt.patch_size)


            for (i,p) in enumerate(patches):
                tv.utils.save_image(p, "test_imgs/%s.png" % i)
                
            pdb.set_trace()


            optimizer.zero_grad()
            # reconstructed, imp_mask_sigmoid = model(ipt)
            reconstructed = model(ipt)
            

            # print ('imp_mask_height', model.imp_mask_height)
            # pdb.set_trace()

            # print ('type recons', type(reconstructed.data))
            loss = mse_loss(reconstructed, ipt)
            caffe_loss = loss / (2 * opt.batch_size)
            
            if opt.use_imp:
                rate_loss_display = imp_mask_sigmoid
                rate_loss_ =  rate_loss(rate_loss_display)
    
                total_loss = caffe_loss + rate_loss_
            else:
                total_loss = caffe_loss
    
            # 1.
            total_loss.backward()
            # caffe_loss.backward()
            optimizer.step()

            mse_loss_meter.add(caffe_loss.data[0])

            if opt.use_imp:
                rate_loss_meter.add(rate_loss_.data[0])
                rate_display_meter.add(rate_loss_display.data.mean())
                total_loss_meter.add(total_loss.data[0])

            if idx % opt.print_freq == opt.print_freq - 1:
                ps.add_point('train mse loss', mse_loss_meter.value()[0] if opt.print_smooth else caffe_loss.data[0])
                ps.add_point('cur epoch train mse loss', mse_loss_meter.value()[0] if opt.print_smooth else caffe_loss.data[0])
                # print (rate_loss_display.data.mean())
                if opt.use_imp:
                    ps.add_point('train rate value', rate_display_meter.value()[0] if opt.print_smooth else rate_loss_display.data.mean())
                    ps.add_point('train rate loss', rate_loss_meter.value()[0] if opt.print_smooth else rate_loss_.data[0])
                    ps.add_point('train total loss', total_loss_meter.value()[0] if opt.print_smooth else total_loss.data[0])
                # pdb.set_trace()
                # progress_bar.set_description('epoch %d/%d, loss = %.2f' % (epoch, opt.max_epoch, total_loss.data[0]))

                #  2. 
                # ps.log('Epoch %d/%d, Iter %d/%d, loss = %.2f, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader), total_loss_meter.value()[0], lr))
                # ps.log('Epoch %d/%d, Iter %d/%d, loss = %.2f, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader), mse_loss_meter.value()[0], lr))
                
                # ps.log('loss = %f' % caffe_loss.data[0])
                # print(total_loss.data[0])                                
                # input('waiting......')

                if not opt.use_imp:
                    ps.log('Epoch %d/%d, Iter %d/%d, loss = %.2f, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader), total_loss_meter.value()[0], lr))
                else:
                    ps.log('Epoch %d/%d, Iter %d/%d, loss = %.2f, mse_loss = %.2f, rate_loss = %.2f, rate_display = %.2f, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader), total_loss_meter.value()[0], mse_loss_meter.value()[0], rate_loss_meter.value()[0], rate_display_meter.value()[0], lr))
                # 进入debug模式                
                if os.path.exists(opt.debug_file):
                    # import pdb
                    pdb.set_trace()
        
        if use_data_parallel:
            # print (type(model.module))
            # print (model)
            # print (type(model))
            model.module.save(optimizer, epoch)
        else:
            model.save(optimizer, epoch)
        
        # print ('case error', total_loss.data[0])
        # print ('smoothed error', total_loss_meter.value()[0])


        # plot before val can ease me 
        ps.make_plot('train mse loss')   # all epoch share a same img, so give "" to epoch
        ps.make_plot('cur epoch train mse loss',epoch)
        if opt.use_imp:
            ps.make_plot("train rate value")
            ps.make_plot("train rate loss")
            ps.make_plot("train total loss")

        # val 
        if opt.use_imp:
            mse_val_loss, rate_val_loss, total_val_loss, rate_val_display = val(model, val_dataloader, mse_loss, rate_loss, ps)
        else:
            mse_val_loss = val(model, val_dataloader, mse_loss, None, ps)
    
        ps.add_point('val mse loss', mse_val_loss)
        if opt.use_imp:
            ps.add_point('val rate value', rate_val_display)
            ps.add_point('val rate loss', rate_val_loss)
            ps.add_point('val total loss', total_val_loss)
        

        # make plot
        # ps.make_plot('train mse loss', "")   # all epoch share a same img, so give "" to epoch
        # ps.make_plot('cur epoch train mse loss',epoch)
        ps.make_plot('val mse loss')

        if opt.use_imp:
            # ps.make_plot("train rate value","")
            # ps.make_plot("train rate loss","")
            # ps.make_plot("train total loss","")
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
                    print ('Due to early stop anneal lr to',lr,'at epoch',epoch)
                    ps.log ('Due to early stop anneal lr to %.10f at epoch %d' % (lr, epoch))

            else:
                tolerant_now -= 1
    
        # if same_lr_epoch and same_lr_epoch % opt.lr_anneal_epochs == 0: 
        #     same_lr_epoch = 0
        #     tolerant_now = 0
        #     lr = lr * opt.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print ('Due to full epochs anneal lr to',lr,'at epoch',epoch)
        #     ps.log ('Due to full epochs anneal lr to %.10f at epoch %d' % (lr, epoch))

        if opt.use_file_decay_lr and os.path.exists(opt.lr_decay_file):
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # previous_loss = total_loss_meter.value()[0] if opt.use_imp else mse_loss_meter.value()[0]
        previous_loss = total_loss_meter.value()[0]


def test(model, dataloader):
    model.eval()
    avg_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    mse_loss = t.nn.MSELoss(size_average = False)

    revert_transforms = transforms.Compose([
        transforms.Normalize((-1,-1,-1),(2,2,2)),
        transforms.ToPILImage()
    ])

    mse = lambda x,y: (np.sum(np.square(y - x))) / float(x.size)
    psnr = lambda x,y: 10*math.log10(255. ** 2 / mse(x,y))


    mmse = 0
    mpsnr = 0
    mrate = 0
    for idx, data in progress_bar:
        val_input = Variable(data, volatile=True)

        if opt.use_gpu:
            val_input = val_input.cuda()

        reconstructed = model(val_input)
        # clamp to [0.0,1.0]
        # print('min,max', reconstructed.min(), reconstructed.max())
        reconstructed = t.clamp(reconstructed, -1.0, 1.0)
        # print('after clamped, min,max', reconstructed.min(), reconstructed.max())


        if opt.use_imp:
            mrate += model.imp_mask_sigmoid.data.mean()
            # print(model.imp_mask_sigmoid.data)
            # imp_map_data =model.imp_mask_sigmoid.data.cpu()[0].numpy()
            # imp_map_data = imp_map_data.reshape(imp_map_data.shape[1],imp_map_data.shape[2])
            # plt.imshow(imp_map_data, cmap='gray')
            # plt.show()
            imp_map = transforms.ToPILImage()(model.imp_mask_sigmoid.data.cpu()[0])
            # imp_map.show()
            # pdb.set_trace()
            imp_map = imp_map.resize((imp_map.size[0]*8, imp_map.size[1]*8))
            if opt.save_test_img:
                imp_map.save(os.path.join(opt.test_imgs_save_path, "%d_impMap.png" % idx))

        img_origin = revert_transforms(val_input.data.cpu()[0])
        # pdb.set_trace()
        if opt.save_test_img:
            img_origin.save(os.path.join(opt.test_imgs_save_path, "%d_origin.png" % idx))
        
        # img_origin.show()
        # input('origin img is above...')

        img_reconstructed = revert_transforms(reconstructed.data.cpu()[0])
        if opt.save_test_img:
            img_reconstructed.save(os.path.join(opt.test_imgs_save_path, "%d_reconst.png" % idx))

        origin_arr = np.array(img_origin)
        reconst_arr = np.array(img_reconstructed)
        
        mmse += mse(origin_arr, reconst_arr)
        mpsnr += psnr(origin_arr, reconst_arr)

        # img_reconstructed.show()
        # input('reconstructed img is above...')
    mmse /= len(progress_bar) * 1
    mpsnr /= len(progress_bar) * 1
    mrate /= len(progress_bar) * 1
    
    print ('avg mse = {mse}, avg psnr = {psnr}, avg rate = {rate}'.format(mse = mmse, psnr = mpsnr, rate = mrate))



def run_test():
    model = getattr(models, opt.model)(use_imp = opt.use_imp)
    if opt.use_gpu:
        model.cuda()  # ???? model.cuda() or model = model.cuda() all is OK
    
    # test_ckpt = "/home/snk/Desktop/workspace/pytorch_implement/checkpoints/ContentWeightedCNN/03-10/ContentWeightedCNN_3_03-10_19:37:54.pth"
    # test_ckpt = "/home/snk/Desktop/workspace/pytorch_implement/checkpoints/ContentWeightedCNN/03-11/ContentWeightedCNN_30_03-11_03:50:28.pth"
    # test_ckpt = "/home/snk/Desktop/workspace/pytorch_implement/checkpoints/CWCNN_imp_r=0.5_γ=0.2/03-12/CWCNN_imp_r=0.5_γ=0.2_42_03-12_20:38:48.pth"
    test_ckpt = "/home/snk/Desktop/workspace/pytorch_implement/checkpoints/CWCNN_imp_r=0.5_γ=0.2/03-14/CWCNN_imp_r=0.5_γ=0.2_90_03-14_03:00:50.pth"


    model.load(None, test_ckpt)

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )

    kodak_dataset = "/home/snk/WindowsDisk/DataSets/old-datasets/Data/Kodak/"
    test_data = ImageNet_200k(kodak_dataset, train = False, transforms=test_transforms)
    test_dataloader = DataLoader(test_data, 1, shuffle = False)
    test(model, test_dataloader)


# TenCrop + Lambda
# ValueError: Expected 4D tensor as input, got 5D tensor instead.
# [torch.FloatTensor of size 96x10x3x224x224]
def val(model, dataloader, mse_loss, rate_loss, ps):
    if ps:
        ps.log ('validating ... ')
    else:
        print ('run val ... ')
    model.eval()    
    # avg_loss = 0
    # progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), ascii=True)
    # print(type(next(iter(dataloader))))
    # print(next(iter(dataloader)))
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

    for idx, (data, _) in enumerate(dataloader):
        # ps.log('%.0f%%' % (idx*100.0/len(dataloader)))
        val_input = Variable(data, volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda(async=True)

        reconstructed, imp_mask_sigmoid = model(val_input)

        batch_loss = mse_loss(reconstructed, val_input)
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
        # progress_bar.set_description('val_iter %d: loss = %.2f' % (idx+1, batch_caffe_loss.data[0]))
        # progress_bar.set_description('val_iter %d: loss = %.2f' % (idx+1, total_loss_meter.value()[0] if opt.use_imp else mse_loss_meter.value()[0]))
        
        # avg_loss += batch_caffe_loss.data[0]
    
    # avg_loss /= len(dataloader)
    # print('avg_loss =', avg_loss)
    # print('meter loss =', loss_meter.value[0])
    # print ('Total avg loss = {}'.format(avg_loss))
    if opt.use_imp:
        return mse_loss_meter.value()[0], rate_loss_meter.value()[0], total_loss_meter.value()[0], rate_display_meter.value()[0]
    else:
        return mse_loss_meter.value()[0]


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





if __name__ == '__main__':
    import fire
    fire.Fire() 
    # import sys
    # assert len(sys.argv) == 2, 'one for filename, one for function'
    # if sys.argv[1] == 'train':
    #     train()
