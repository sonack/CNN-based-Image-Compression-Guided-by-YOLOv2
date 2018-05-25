#coding: utf-8
from __future__ import print_function
# import visdom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import numpy as np
import os
import pdb
import sys
from config import opt
# sys.path.append('/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/')

class Visualizer(object):
    '''
        封装了visdom的基本操作，但是仍然可以通过self.vis.function调用原生的visdom接口
    '''

    def __init__(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        self.index = {}
        self.log_text = ''
    
    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        return self
    
    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        for k,v in d.items():
            self.plot(k, v)
    
    def img_many(self, d):
        for k, v in d.items():
            self.img(k,v)
    
    def refresh_plot(self, name):
        if name in self.index:
            del self.index[name]
        
    def plot(self, name, y, print_step = 1, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),X=np.array([x]),win=name,opts=dict(title=name, markers=True, markersize=5, markercolor=np.array([[10, 10, 233]]), **kwargs),update=None if x == 0 else 'append')
        self.index[name] = x + print_step
    
    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        !!! don't ~~ self.img('input_imgs', t.Tensor(100, 64, 64), nrows=10)~~ !!!
        '''
        self.vis.images(img_.cpu().numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs
                       )
    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m月%d日 %H:%M:%S'),\
                            info=info)) 
        self.vis.text(self.log_text, win)
    
    def __getattr__(self, name):
        '''
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        '''
        return getattr(self.vis, name)


class PlotSaver(object):
    '''
        Visualizer的退化版, 为了在不能使用visdom的GPU HPC上记录log和绘图
    '''
    def __init__(self, log_name):
        self.log_name = log_name
        self.index = {}
        self.steps = {}
        self.x_labels = {}
        self.y_labels = {}
        self.log_text = ''
    
    def new_plot(self, title_key, step, xlabel, ylabel):
        if title_key in self.index:
            del self.index[title_key]
            del self.steps[title_key]
            del self.x_labels[title_key]
            del self.y_labels[title_key]
        
        self.index[title_key] = []
        self.steps[title_key] = step
        self.x_labels[title_key] = xlabel
        self.y_labels[title_key] = ylabel
    
    def add_point(self, title_key, value):
        assert title_key in self.index, "You must new this plot before adding points to it!"
        self.index[title_key].append(value)
    
    def log(self, text, append = True):
        log_file = os.path.join(opt.log_path, self.log_name)
        with open(log_file, "a" if append else "w") as f:
            rec = time.strftime('%m月%d日 %H:%M:%S : ') + text + '\n'
            f.write(rec)

    def make_plot(self, title_key, epoch="*"):
        assert title_key in self.index, "You must new this plot before making plot to it!"         
        plot_name = 'epoch_{epoch}_{title_key}.png'.format(title_key=title_key.replace(' ','_'), epoch=epoch)
        plot_file = os.path.join(opt.plot_path, plot_name)
        plt.ioff()
        plt.title(title_key)
        plt.xlabel(self.x_labels[title_key])
        plt.ylabel(self.y_labels[title_key])
        # pdb.set_trace()
        len_ = len(self.index[title_key])
        step = self.steps[title_key]
        len_ = len_ * step
        plt.plot(
            list(range(0, len_, step)),
            self.index[title_key],
            marker='o',
            ms=5
        )
        plt.savefig(plot_file, dpi=300)
        plt.close()


def test():
    ps = PlotSaver("train_imagenet_without_imp.log")
    ps.new_plot('train mse loss', 1, xlabel="iteration", ylabel="train_mse_loss")
    for i in range(10):
        ps.add_point('train mse loss', i * 3)
    ps.make_plot('train mse loss', 1)

    ps.log("test log", False)
    ps.log("loss = 1")
    


if __name__ == '__main__':
    test()

