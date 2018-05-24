#coding:utf-8
'''
ResNet
'''

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import time


# used for 18 and 34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_mid, stride=1):
        super(BasicBlock, self).__init__()
        ch_out = ch_mid * self.expansion
        self.conv1 = nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_mid)
        self.conv2 = nn.Conv2d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.shortcut = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# used for 50 101 152 ...
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_mid, stride=1):
        super(Bottleneck,self).__init__()
        ch_out = ch_mid * self.expansion
        self.conv1 = nn.Conv2d(ch_in, ch_mid, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_mid)
        self.conv2 = nn.Conv2d(ch_mid, ch_mid, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_mid)
        self.conv3 = nn.Conv2d(ch_mid, ch_out, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch_out)

        self.shortcut = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
CHECKPOINTS_DIR = "checkpoints"

class ResNet(nn.Module):
    def __init__(self, model_name, block, num_blocks, num_classes = 1000, is_cResNet=False):
        super(ResNet, self).__init__()
        self.model_name = model_name
        self.is_cResNet = is_cResNet
        self.ch_in = 64
    
        if not is_cResNet:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        if not is_cResNet:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2 if not is_cResNet else 1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, ch_mid, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ch_in, ch_mid, stride))
            self.ch_in = ch_mid * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if not self.is_cResNet:
            out = F.relu(self.bn1(self.conv1(x)))
            # print ('1',out.size())
            out = F.max_pool2d(out, 3, 2)  ## mark1
            # print ('2',out.size())
            out = self.layer1(out)
        else:
            out = x
        
        # print ('3',out.size())                
        out = self.layer2(out)
        # print ('4',out.size())                
        out = self.layer3(out)
        # print ('5',out.size())                
        out = self.layer4(out)
        # print ('6',out.size())                
        out = F.avg_pool2d(out, 7, 1)
        # print ('7',out.size())                
        out = out.view(out.size(0), -1)
        # print ('8',out.size())
        out = self.linear(out)
        return out
    
    # just like the BasicModule
    def load(self, optimizer, path, finetune):
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



def ResNet18():
    return ResNet('ResNet18', BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet('ResNet34', BasicBlock, [3,4,6,3])

# bs = 128
def ResNet50():
    return ResNet('ResNet50_bs=256', Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet('ResNet101', Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet('ResNet152', Bottleneck, [3,8,36,3])

# cResNet

def cResNet51():
    return ResNet('cResNet51', Bottleneck, [-1, 4, 10, 3], is_cResNet=True)


def test():
    net = ResNet50()
    y = net(Variable(t.randn(1,3,224,224)))
    # print (y.size())
    # print (net)

if __name__ == '__main__':
    test()
