#coding:utf-8
from .BasicModule import BasicModule
import torch as th
from torch import nn
from torch.nn import functional as F
import sys
import math

from extend import RoundCuda, ImpMapCuda, LimuRound, Round

from torch.nn.init import xavier_uniform

import pdb
from config import opt

class ResidualBlock(nn.Module):
    '''
    残差模块
        input(nchw) -> Conv(3x3x128, pad 1) -> ReLU -> Conv(3x3xc, pad 1) -> ReLU
        input(nchw)                                                                +  = output(nchw)
    '''
    def __init__(self, ch_in, ch_out, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(ch_in, 128, 3, 1, 1),  # ch_in, ch_out, kernel_size, stride, pad
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, ch_out, 3, 1, 1),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False),
        )
        self.right = shortcut
    
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out, inplace=False)


def xavier_init(data):
    fan_in = data.numel() / data.size(0)
    scale = math.sqrt(3.0 / fan_in)  # (out_channels , in_channels , kernel_size)
    data.uniform_(-scale, scale)


def weights_initialization(m):
    classname = m.__class__.__name__
    # print(type(m))
    if classname.find('Conv') != -1:
        # print (m) total 21 conv layer    correct!
        # print(type(m.weight.data))
        # print(m.weight.data.size())
        # pdb.set_trace()
        xavier_init(m.weight.data)
        # xavier_uniform(m.weight)
        m.bias.data.fill_(0)



# YOLOv2 combined
class ContentWeightedCNN_YOLO(BasicModule):
    '''
    Learning Convolutional Networks for Content-weighted Image Compression 
    '''
    def __init__(self, use_imp = True, n = 64, input_4_ch = True, model_name = None):
        super(ContentWeightedCNN_YOLO, self).__init__()
        self.model_name = model_name if model_name else 'CWCNN_with_YOLOv2'
        self.use_imp = use_imp
        self.n = n
        self.input_4_ch = input_4_ch
        self.encoder = self.make_encoder()
        # self.round = RoundCuda()
        # print ('self.n', self.n)
        # pdb.set_trace()
        # if input_4_ch:
        #     self.mask_parser = self.make_mask_parser()
        if use_imp:
            self.impmap_sigmoid = self.make_impmap()
            self.impmap_expand = ImpMapCuda(L = 16, n = n)
        self.decoder = self.make_decoder()
        self.reset_parameters()
    
    def reset_parameters(self):
        self.apply(weights_initialization)
    
    def forward(self, x, m, o_m, need_decode = True):
        # pdb.set_trace()
        if self.input_4_ch:
            mgdata = self.encoder(th.cat((x,o_m), 1))
            # mgdata = self.encoder(x)
            # parsed_mask = self.mask_parser(o_m)
            # pdb.set_trace()
            # mgdata = self.round(mgdata * parsed_mask)

        else:
            mgdata = self.encoder(x)
            # mgdata = self.round(mgdata)
       
        # print('mgdata size',mgdata.shape)
        if self.use_imp:
            # m = m.unsqueeze(1)
            # print (m.size)
            # ex_mgdata = th.cat((mgdata, m), 1)
            ex_mgdata = mgdata
            # pdb.set_trace()            
            self.imp_mask_sigmoid = self.impmap_sigmoid(ex_mgdata)
            # pdb.set_trace()
            # masked_imp_map = (self.imp_mask_sigmoid * m).clamp(max=0.999999)
            masked_imp_map = self.imp_mask_sigmoid
            self.imp_mask, self.imp_mask_height = self.impmap_expand(masked_imp_map)
            # pdb.set_trace()
            enc_data = mgdata * self.imp_mask
        else:
            enc_data = mgdata
        if need_decode:
            dec_data = self.decoder(enc_data)
        # print ('dec_data size', dec_data.size())
        if self.use_imp:
            return (dec_data, self.imp_mask_sigmoid) if need_decode else (enc_data, self.imp_mask_height)
            # return (dec_data, masked_imp_map) if need_decode else (enc_data, self.imp_mask_height)
        else:
            return (dec_data, None)
        # return dec_data  # no_imp



    def make_impmap(self):
        layers = [
            # 64 + 1 mask channel
            nn.Conv2d(self.n, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 1, 1, 1, 0),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

    # def make_mask_parser(self):
    #     layers = [
    #         # nn.Conv2d(4 if self.input_4_ch else 3, 32, 8, 4, 2),
    #         nn.Conv2d(1, 32, 8, 4, 2),
    #         nn.BatchNorm2d(32),            
    #         nn.ReLU(inplace=False), # 54   # 128 -> 32

    #         nn.Conv2d(32, 64, 4, 2, 1), # 115  # 32 -> 16
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=False),

    #         nn.Conv2d(64, 64, 3, 1, 1), # 115  # 32 -> 16
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=False),
    #         nn.Conv2d(64, self.n, 1, 1, 0),    # conv 4  64 is n  # 16 -> 16
    #         nn.BatchNorm2d(self.n),
    #         nn.Sigmoid()     
    #     ]
    #     return nn.Sequential(*layers)              


    def make_encoder(self):
        layers = [
            # changed to 4
            nn.Conv2d(4 if self.input_4_ch else 3, 128, 8, 4, 2),
            # nn.Conv2d(3, 128, 8, 4, 2), 
            # nn.BatchNorm2d(128),          
            nn.ReLU(inplace=False), # 54   # 128 -> 32

            ResidualBlock(128, 128),

            nn.Conv2d(128, 256, 4, 2, 1), # 115  # 32 -> 16
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            ResidualBlock(256, 256),

            nn.Conv2d(256, 256, 3, 1, 1), #192  # 16 -> 16
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            ResidualBlock(256, 256),

            nn.Conv2d(256, self.n, 1, 1, 0),    # conv 4  64 is n  # 16 -> 16
            # nn.BatchNorm2d(self.n),
            nn.Sigmoid(),                    
            # Round()                         # mgdata
            RoundCuda()
            # LimuRound(1-1e-10, 0.01)  # ratio, scale to fix grad
        ]
        return nn.Sequential(*layers)



    def make_decoder(self):
        layers = [
            nn.Conv2d(self.n, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),

            ResidualBlock(512, 512),

            nn.Conv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),

            ResidualBlock(512, 512),

            nn.PixelShuffle(2),  # 128 x 32 x 32

            nn.Conv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            ResidualBlock(256, 256),

            nn.PixelShuffle(4),  # 256 x 128 x 128

            nn.Conv2d(16, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            
            nn.Conv2d(32, 3, 1, 1, 0) # 3 x 128 x 128
        ]
        return nn.Sequential(*layers)


# Original Model
class ContentWeightedCNN(BasicModule):
    '''
    Learning Convolutional Networks for Content-weighted Image Compression 
    '''
    def __init__(self, use_imp = True, n = 64, model_name = None):
        super(ContentWeightedCNN, self).__init__()
        self.model_name = model_name if model_name else 'CWCNN_default_name'
        self.use_imp = use_imp
        self.n = n
        self.shared_encoder = self.make_shared_encoder()
        self.encoder = self.make_encoder()
        if use_imp:
            self.impmap_sigmoid = self.make_impmap()
            self.impmap_expand = ImpMapCuda(L = 16, n = n)   # ImpMap 没有检查
        self.decoder = self.make_decoder()
        self.reset_parameters()
    
    def reset_parameters(self):
        self.apply(weights_initialization)
    

    def forward(self, x, need_decode = True):
        last_residual_out = self.shared_encoder(x)
        mgdata = self.encoder(last_residual_out)
        if self.use_imp:
            self.imp_mask_sigmoid = self.impmap_sigmoid(last_residual_out)
            self.imp_mask, self.imp_mask_height = self.impmap_expand(self.imp_mask_sigmoid)
            enc_data = mgdata * self.imp_mask
        else:
            enc_data = mgdata
        if need_decode:
            dec_data = self.decoder(enc_data)

        # print ('dec_data size', dec_data.size())
        
        if self.use_imp:
            return (dec_data, self.imp_mask_sigmoid) if need_decode else (enc_data, self.imp_mask_height)
        else:
            return dec_data  # no_imp



    def make_impmap(self):
        layers = [
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 1, 1, 1, 0),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)


    def make_shared_encoder(self):
        layers = [
            nn.Conv2d(3, 128, 8, 4, 2),
            nn.ReLU(inplace=False), # 54   # 128 -> 32

            ResidualBlock(128, 128),

            nn.Conv2d(128, 256, 4, 2, 1), # 115  # 32 -> 16
            nn.ReLU(inplace=False),

            ResidualBlock(256, 256),

            nn.Conv2d(256, 256, 3, 1, 1), #192  # 16 -> 16
            nn.ReLU(inplace=False),

            ResidualBlock(256, 256),
        ]
        return nn.Sequential(*layers)


    def make_encoder(self):
        layers = [
            nn.Conv2d(256, self.n, 1, 1, 0),    # conv 4  64 is n  # 16 -> 16
            nn.Sigmoid(),                    
            # Round()                         # mgdata
            RoundCuda()
            # LimuRound(1-(1e-10), 0.01)  # ratio, scale to fix grad
        ]
        return nn.Sequential(*layers)



    def make_decoder(self):
        layers = [
            nn.Conv2d(self.n, 512, 3, 1, 1),
            nn.ReLU(inplace=False),

            ResidualBlock(512, 512),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=False),

            ResidualBlock(512, 512),

            nn.PixelShuffle(2),  # 128 x 32 x 32

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=False),

            ResidualBlock(256, 256),

            nn.PixelShuffle(4),  # 256 x 128 x 128

            nn.Conv2d(16, 32, 3, 1, 1),  # 论文补充资料A table 2 给出的参数为depth2space(4)=128???
            nn.ReLU(inplace=False),
            
            nn.Conv2d(32, 3, 1, 1, 0) # 3 x 128 x 128
        ]
        return nn.Sequential(*layers)



# UNet-ContentWeighedCNN 
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class ContentWeightedCNN_UNET(BasicModule):
    def __init__(self, use_unet = True, use_imp = False, model_name = None):
        super(ContentWeightedCNN_UNET, self).__init__()
        self.use_unet = use_unet
        self.model_name = "UNet_teacher_CWCNN" if model_name is None else model_name
        encoder_list = []
        for i in range(4): # 0_(output_size = 64) 1_(output_size = 32) 2_(output_size = 16) 3_(output: code)
            encoder_list.append(self.make_encoder(i))
        self.encoder = ListModule(*encoder_list)
        decoder_list = []
        for j in range(4): # 0_(input: code) 1_(input_size = 16, +++ e2) 2_(input_size = 32, +++ e1 ) 3_(input_size = 64)
            decoder_list.append(self.make_decoder(j))
        self.decoder = ListModule(*decoder_list)

    def make_encoder(self, idx):
        es = [
                # idx = 0
                [
                    nn.Conv2d(3, 32, 4, 2, 1),  # 3x128x128 -> 32x64x64
                    nn.ReLU(inplace=False),

                    ResidualBlock(32, 32),  # 32x64x64 -> 32x64x64
                ],

                # idx = 1
                [
                    nn.Conv2d(32, 128, 4, 2, 1),  # 32x64x64 -> 128x32x32
                    nn.ReLU(inplace=False),

                    ResidualBlock(128, 128),
                ],

                # idx = 2
                [
                    nn.Conv2d(128, 512, 4, 2, 1), # 128x32x32 -> 512x16x16
                    nn.ReLU(inplace=False),

                    ResidualBlock(512, 512),
                ],

                # idx = 3
                [
                    nn.Conv2d(512, 64, 1, 1, 0),  # 512x16x16 -> 64x16x16
                    nn.Sigmoid(),
                    LimuRound(1-1e-10, 0.01)
                ]
        ]
        return nn.Sequential(*es[idx])

    def make_decoder(self, idx):
        ds = [
            # idx = 0, decode code, input: code
            [
                nn.Conv2d(64, 512, 3, 1, 1),
                nn.ReLU(inplace=False),

                ResidualBlock(512, 512),
            ],

            # idx = 1, input += e2
            [
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=False),

                ResidualBlock(512, 512),

                nn.PixelShuffle(2),   # 128x32x32
            ],

            # idx = 2, input += e1
            [
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU(inplace=False),

                ResidualBlock(128, 128),

                nn.PixelShuffle(2),  # 32x64x64
            ],

            # idx = 3, input += e0
            [
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=False),

                ResidualBlock(32, 32),

                nn.PixelShuffle(2), # 8 * 128 * 128

                nn.Conv2d(8, 32, 3, 1, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 3, 1, 1, 0)
            ]
        ]
        return nn.Sequential(*ds[idx])

    def forward(self, x, expose_skip = False):
        # encoder 
        e = [0, 0, 0, 0]
        e_in = x
        for i in range(4):
            e[i] = self.encoder[i](e_in)
            e_in = e[i]
        # code is e_in

        # decoder
        d = [0, 0, 0, 0]
        d_in = e_in
        for j in range(4):
            d[j] = self.decoder[j](d_in)
            if j < 3:  # skip-connection UNet here
                d_in = d[j] + e[2-j] if self.use_unet else d[j]
        # decode res is d[3]
        if not expose_skip:
            return d[3]
        else:
            return d[3], d[0], d[1], d[2]
