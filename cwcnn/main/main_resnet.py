#coding:utf8
from __future__ import print_function
from config_resnet import opt
import os
import numpy as np
import math
import torch as t
import models
from models import ResNet50
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
import fire


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



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(**kwargs):
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)
    # log file
    ps = PlotSaver("FrozenCNN_ResNet50_RGB_"+time.strftime("%m_%d_%H:%M:%S")+".log.txt")


    # step1: Model
    compression_model = getattr(models, opt.model)(use_imp = opt.use_imp, model_name="CWCNN_limu_ImageNet_imp_r={r}_γ={w}_for_resnet50".format(
                                                                r=opt.rate_loss_threshold,
                                                                w=opt.rate_loss_weight)
                                                                if opt.use_imp else None)
                                                # if opt.use_imp else "test_pytorch")
    
    compression_model.eval()
    compression_model.load(None, opt.compression_model_ckpt)

    
    resnet_50 = ResNet50()
    if opt.use_gpu:
        # model = multiple_gpu_process(model)
        compression_model.cuda()
        resnet_50.cuda()

    # freeze the compression network
    for param in compression_model.parameters():
        param.requires_grad = False

#    pdb.set_trace()

    cudnn.benchmark = True

    # step2: Data
    normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )

    train_data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            #transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    )
    val_data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            #transforms.Scale(256),
            transforms.CenterCrop(224),
            # transforms.TenCrop(224),
            # transforms.Lambda(lambda crops: t.stack(([normalize(transforms.ToTensor()(crop)) for crop in crops]))),
            transforms.ToTensor(),
            normalize
        ]
    )
    # train_data = ImageNet_200k(opt.train_data_root, train=True, transforms=data_transforms)
    # val_data = ImageNet_200k(opt.val_data_root, train = False, transforms=data_transforms)
    train_data = datasets.ImageFolder(
        opt.train_data_root,
        train_data_transforms
    )
    val_data = datasets.ImageFolder(
        opt.val_data_root,
        val_data_transforms
    )
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
 
    # step3: criterion and optimizer

    # mse_loss = t.nn.MSELoss(size_average = False)

    class_loss = t.nn.CrossEntropyLoss()

    # if opt.use_imp:
        # rate_loss = RateLoss(opt.rate_loss_threshold, opt.rate_loss_weight)
        # rate_loss = LimuRateLoss(opt.rate_loss_threshold, opt.rate_loss_weight)

    lr = opt.lr

    optimizer = t.optim.Adam(resnet_50.parameters(), lr=lr, betas=(0.9, 0.999))

    start_epoch = 0

    if opt.resume:
        # if hasattr(model, 'module'):
            # start_epoch = model.module.load(None if opt.finetune else optimizer, opt.resume, opt.finetune)
        # else:
        start_epoch = resnet_50.load(None if opt.finetune else optimizer, opt.resume, opt.finetune)

        if opt.finetune:
            print ('Finetune from model checkpoint file', opt.resume)
        else:
            print ('Resume training from checkpoint file', opt.resume)
            print ('Continue training at epoch %d.' %  start_epoch)

    # step4: meters
    class_loss_meter = AverageValueMeter()

    class_acc_top5_meter = AverageValueMeter()
    class_acc_top1_meter = AverageValueMeter()

    # ps init

    ps.new_plot('train class loss', opt.print_freq, xlabel="iteration", ylabel="train_CE_loss")
    ps.new_plot('val class loss', 1, xlabel="epoch", ylabel="val_CE_loss")


    ps.new_plot('train top_5 acc', opt.print_freq, xlabel="iteration", ylabel="train_top5_acc")
    ps.new_plot('train top_1 loss',  opt.print_freq, xlabel="iteration", ylabel="train_top1_acc")
     
    ps.new_plot('val top_5 acc', 1, xlabel="iteration", ylabel="val_top_5_acc")
    ps.new_plot('val top_1 acc', 1, xlabel="iteration", ylabel="val_top_1_acc")


    
    for epoch in range(start_epoch+1, opt.max_epoch+1):
        # per epoch avg loss meter
        class_loss_meter.reset()
        
        class_acc_top1_meter.reset()
        class_acc_top5_meter.reset()
       
        # cur_epoch_loss refresh every epoch

        ps.new_plot("cur epoch train class loss", opt.print_freq, xlabel="iteration in cur epoch", ylabel="cur_train_CE_loss")
  
        resnet_50.train()
        # _ is corresponding Label， compression doesn't use it.
        for idx, (data, label) in enumerate(train_dataloader):
            ipt = Variable(data)
            label = Variable(label)
            # print('label: ',label)

            # pdb.set_trace()
            if opt.use_gpu:
                ipt = ipt.cuda()
                label = label.cuda()

            optimizer.zero_grad()   # fuck it! Don't forget to clear grad!
            compressed_RGB = compression_model(ipt)

            predicted = resnet_50(compressed_RGB)
            
            # print ('reconstructed tensor size :', reconstructed.size())
            # loss = mse_loss(reconstructed, ipt)
            # caffe_loss = loss / (2 * opt.batch_size)
            class_loss_ = class_loss(predicted, label)
            # if opt.use_imp:
            #     # print ('use data_parallel?',use_data_parallel)
            #     # pdb.set_trace()
            #     rate_loss_display = (model.module if use_data_parallel else model).imp_mask_sigmoid
            #     rate_loss_ =  rate_loss(rate_loss_display)
            #     total_loss = caffe_loss + rate_loss_
            # else:
            #     total_loss = caffe_loss

            class_loss_.backward()
            
            optimizer.step()

            class_loss_meter.add(class_loss_.data[0])

            acc1, acc5 = accuracy(predicted.data, label.data, topk=(1, 5))


            pdb.set_trace()

            
            class_acc_top1_meter.add(acc1[0])
            class_acc_top5_meter.add(acc5[0])

            # if opt.use_imp:
            # rate_loss_meter.add(rate_loss_.data[0])
            # rate_display_meter.add(rate_loss_display.data.mean())
            # total_loss_meter.add(total_loss.data[0])

            if idx % opt.print_freq == opt.print_freq - 1:
                ps.add_point('train class loss', class_loss_meter.value()[0] if opt.print_smooth else class_loss_.data[0])
                ps.add_point('cur epoch train class loss', class_loss_meter.value()[0] if opt.print_smooth else class_loss_.data[0])
                # if opt.use_imp:
                ps.add_point('train top_5 acc', class_acc_top5_meter.value()[0] if opt.print_smooth else acc5[0)
                ps.add_point('train top_1 acc', class_acc_top1.value()[0] if opt.print_smooth else acc1[0])
               

                # if not opt.use_imp:
                    # ps.log('Epoch %d/%d, Iter %d/%d, loss = %.2f, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader), total_loss_meter.value()[0], lr))
                # else:
                    ps.log('Epoch %d/%d, Iter %d/%d, class loss = %.2f, top 5 acc = %.2f %%, top 1 acc  = %.2f %%, lr = %.8f' % (epoch, opt.max_epoch, idx, len(train_dataloader),  class_loss_meter.value()[0], class_acc_top5_meter.value()[0], class_acc_top1_meter.value()[0], lr))
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    pdb.set_trace()

        # data parallel
        # if hasattr(model, 'module'):
        # if use_data_parallel:
        #     model.module.save(optimizer, epoch)
        # else:
        resnet_50.save(optimizer, epoch)

        # plot before val can ease me
        ps.make_plot('train class loss')   # all epoch share a same img, so give ""(default) to epoch
        ps.make_plot('cur epoch train class loss',epoch)
        # if opt.use_imp:
        ps.make_plot("train top_5 acc")
        ps.make_plot("train top_1 acc")
           

        # val
        # if opt.use_imp:
        #     mse_val_loss, rate_val_loss, total_val_loss, rate_val_display = val(model, val_dataloader, mse_loss, rate_loss, ps)
        # else:
        val_class_loss, val_top5_acc, val_top1_acc = val(model, val_dataloader, class_loss, None, ps)

        ps.add_point('val class loss', val_class_loss)
        # if opt.use_imp:
        ps.add_point('val top_5 acc', val_top5_acc)
        ps.add_point('val top_1 acc', val_top1_acc)
        # ps.add_point('val total loss', total_val_loss)



        ps.make_plot('val class loss')

        # if opt.use_imp:
        ps.make_plot('val top_5 acc')
        ps.make_plot('val top_1 acc')
        # ps.make_plot('val total loss')

        # log sth.
#         if opt.use_imp:
#             ps.log('Epoch:{epoch}, lr:{lr}, train_mse_loss: {train_mse_loss}, train_rate_loss: {train_rate_loss}, train_total_loss: {train_total_loss}, train_rate_display: {train_rate_display} \n\
# val_mse_loss: {val_mse_loss}, val_rate_loss: {val_rate_loss}, val_total_loss: {val_total_loss}, val_rate_display: {val_rate_display} '.format(
#                 epoch = epoch,
#                 lr = lr,
#                 train_mse_loss = mse_loss_meter.value()[0],
#                 train_rate_loss = rate_loss_meter.value()[0],
#                 train_total_loss = total_loss_meter.value()[0],
#                 train_rate_display = rate_display_meter.value()[0],

#                 val_mse_loss = mse_val_loss,
#                 val_rate_loss = rate_val_loss,
#                 val_total_loss = total_val_loss,
#                 val_rate_display = rate_val_display
#             ))
#         else:
        ps.log('Epoch:{epoch}, lr:{lr}, train_class_loss: {train_class_loss}, train_top5_acc: {train_top5_acc}%, train_top1_acc: {train_top1_acc}%,\n\
val_class_loss: {val_class_loss}, val_top5_acc: {val_top5_acc}%, val_top1_acc: {val_top1_acc}'.format(
            epoch = epoch,
            lr = lr,
            train_class_loss = train_class_loss.value()[0],
            train_top5_acc =  class_acc_top5_meter.value()[0],
            train_top1_acc = class_acc_top1_meter.value()[0],
            val_class_loss = val_class_loss,
            val_top5_acc = val_top5_acc,
            val_top1_acc = val_top1_acc

        ))

        # Adaptive adjust lr
        # 每个lr，如果有opt.tolerant_max次比上次的val_loss还高，
        # if opt.use_early_adjust:
        #     if total_loss_meter.value()[0] > previous_loss:
        #         tolerant_now += 1
        #         if tolerant_now == opt.tolerant_max:
        #             tolerant_now = 0
        #             same_lr_epoch = 0
        #             lr = lr * opt.lr_decay
        #             for param_group in optimizer.param_groups:
        #                 param_group['lr'] = lr
        #             print ('Anneal lr to',lr,'at epoch',epoch,'due to early stop.')
        #             ps.log ('Anneal lr to %.10f at epoch %d due to early stop.' % (lr, epoch))

        #     else:
        #         tolerant_now -= 1

        # if same_lr_epoch and same_lr_epoch % opt.lr_anneal_epochs == 0:
        #     same_lr_epoch = 0
        #     tolerant_now = 0
        #     lr = lr * opt.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print ('Anneal lr to',lr,'at epoch',epoch,'due to full epochs.')
        #     ps.log ('Anneal lr to %.10f at epoch %d due to full epochs.' % (lr, epoch))

        # previous_loss = total_loss_meter.value()[0]


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
            
        ]
    )

    kodak_dataset = "/home/snk/WindowsDisk/DataSets/old-datasets/Data/Kodak/"
    test_data = ImageNet_200k(kodak_dataset, train = False, transforms=test_transforms)
    test_dataloader = DataLoader(test_data, 1, shuffle = False)
    test(model, test_dataloader)


# TenCrop + Lambda
# ValueError: Expected 4D tensor as input, got 5D tensor instead.
# [torch.FloatTensor of size 96x10x3x224x224]
def val(model, dataloader, mse_loss, rate_loss=None, ps=None):
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
        #ps.log('%.0f%%' % (idx*100.0/len(dataloader)))
        val_input = Variable(data, volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()

        reconstructed = model(val_input)

        batch_loss = mse_loss(reconstructed, val_input)
        batch_caffe_loss = batch_loss / (2 * opt.batch_size)

        if opt.use_imp and rate_loss:
            rate_loss_display = model.imp_mask_sigmoid
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
    model.load(None, opt.val_ckpt)
    # print (list(model.parameters())[0])

    val_transforms = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        ]
    )
    # val_data = ImageNet_200k(opt.val_data_root, train = False, transforms=val_transforms)
    val_data = datasets.ImageFolder(
        opt.run_val_data_root,
        val_transforms
    )
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    mse_loss = t.nn.MSELoss(size_average = False)
    ret = val(model, val_dataloader, mse_loss)
    if not opt.use_imp:
        print ('mse loss is',ret)




if __name__ == '__main__':
    import fire
    fire.Fire()
    # import sys
    # assert len(sys.argv) == 2, 'one for filename, one for function'
    # if sys.argv[1] == 'train':
    #     train()
