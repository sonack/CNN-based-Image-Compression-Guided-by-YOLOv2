#coding:utf-8
import torch as t
import time
import os


CHECKPOINTS_DIR = "checkpoints"
class BasicModule(t.nn.Module):
    '''
    封装了nn.Module，主要提供save和load两个方法
    '''
    def __init__(self,opt=None):
       super(BasicModule,self).__init__()
       self.model_name = str(type(self)) # 模型的默认名字

    # return epoch
    # load model and optimizer state_dict
    def load(self, optimizer, path, finetune=False):
        '''
        可加载指定路径的模型
        finetune: 使用之前的网络权重来finetune当前不同的网络结构的权重，为True
                  只是单纯的继续之前停掉的训练，而网络结构并未改变，为False
        '''
        checkpoint = t.load(path)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optim'])
        if finetune:
            state = self.state_dict()
            state.update(checkpoint['model'])
            self.load_state_dict(state)
        else:
            self.load_state_dict(checkpoint['model'])
        return checkpoint['epoch'] if not finetune else 0

    def save(self, optimizer, epoch, path=None, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        '''
        if path is None:
            # model save dir MODEL_NAME/DATE/checkpoints_file
            path = os.path.join(CHECKPOINTS_DIR, self.model_name, time.strftime('%m-%d'))
        if name is None:
            name = self.model_name + ("_%d" % epoch) + time.strftime('_%m-%d_%H:%M:%S.pth')
        save_path = os.path.join(path, name)
        if not os.path.exists(path):
            os.makedirs(path)
        t.save({
            'model' : self.state_dict(),
            'optim' : optimizer.state_dict(),
            'epoch' : epoch
        }, save_path)
        return name