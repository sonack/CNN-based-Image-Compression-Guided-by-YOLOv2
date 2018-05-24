#!/usr/bin/env python3
#coding: utf-8

import struct
import torch as t
import pdb

'''
16 level : 4bits
n = 64 : 64 * r

little endian: <
3 bytes
[w, h, dw, dh]  imp_map:[w/8 * h/8 * 4bits] +  [w/8 * h/8 * 64 * r]
'''

# 4bits for impmap
def to_4bits(x1, x2, bits):
    # print (x1, x2)
    x = (x1 << 4) + x2
    # print (x)
    b = struct.pack('<B',x)
    bits.append(b)


def from_4bits(b):
    x = struct.unpack('<B', b)[0]
    x2 = x & 0x0f
    x1 = x >> 4
    return x1, x2

def test_write_binary():
    l = []
    with open('test.txt','wb') as f:
        # to_4bits(15,15,l)
        write_a_byte(['0','1'])
        bs = b''.join(l)
        f.write(bs)

def test_read_binary():
    with open('test.txt', 'rb') as f:
        x1, x2 = from_4bits(f.read(1))
        print (x1,x2)

# 01 str to bytes
# every 8 bits to a byte
# l = ['0','1','1','0' ... ]
def write_a_byte(l, bits):
    x = int(''.join(l),2)
    b = struct.pack('<B',x)
    bits.append(b)

def read_a_byte(b):
    x = struct.unpack('<B', b)[0]
    l = []
    cnt = 8
    while cnt:
        l.append(x%2)
        x >>= 1
        cnt -= 1
    return l[::-1]


class Serializer:
    def __init__(self):
        # self.s_f = s_filename
        self.bits_imp_map = []
        self.bits_codes = []
        self.bits_size = []
    
    def set_filename(self, s_filename):
        self.s_f = s_filename
    
    def get_filename(self):
        return self.s_f
    
    # [torch.cuda.FloatTensor of size 96x1x28x28 (GPU 0)]
    '''
        (0 ,0 ,.,.) = 
        20   8   8  ...   20   8  16
        12   4   8  ...   12   8   8
        12   8   8  ...   12   8   8
            ...       ⋱       ...    
        8   4   4  ...    8   4   4
        8   4   8  ...    8  28   4
        8   4   8  ...    8  20  20
            ⋮ 

        (1 ,0 ,.,.) = 
        8   4   8  ...   16   8   8
        8   4   4  ...    8   8   4
        12   8   8  ...    8   8   4
            ...       ⋱       ...    
        8  12  24  ...    8   8   4
        16   4   8  ...   12  12   4
        12   8   8  ...    8   8   8
            ⋮ 

        (2 ,0 ,.,.) = 
        4   4   4  ...    4   4   8
        4   8   4  ...    8   4   4
        4   4   4  ...    4   4   4
            ...       ⋱       ...    
        8   8   8  ...    4   4   4
        4   8   4  ...    4   4   4
        8  12   4  ...    8   8   4
        ...   
            ⋮ 

        (93,0 ,.,.) = 
        4   4   4  ...    4   4   4
        4   4   4  ...    4   4   4
        4   4   4  ...    4   4   4
            ...       ⋱       ...    
        4   4   4  ...    8   4   8
        4   4   4  ...    8   8   4
        4   4   4  ...    4   4   4
            ⋮ 

        (94,0 ,.,.) = 
        4   4   4  ...    8   8   4
        4   4   4  ...    4   4   4
        8   8   8  ...    8   8   4
            ...       ⋱       ...    
        4   8   8  ...    8   4   4
        4  12   8  ...    4   4   4
        8  16  24  ...    8   8   8
            ⋮ 

        (95,0 ,.,.) = 
        16   8   4  ...   16  16  16
        8   8   8  ...    8  12   8
        8   8   8  ...    8  12   8
            ...       ⋱       ...    
        12   8  12  ...   24   4   4
        8   8   4  ...    8   4   8
        16  12   4  ...    8   4   4

'''
    def _serialize_imp(self, imp_tensor):
        last_h = -1
        for x in range(imp_tensor.size(0)):
            for y in range(imp_tensor.size(1)):
                h = int(imp_tensor[x][y] // 4 + 0.00001)
                if last_h == -1:
                    last_h = h
                else:
                    # has 2
                    # print ('last _h', last_h)
                    # {1, 2, ... , 16} no 0
                    to_4bits(last_h, h, self.bits_imp_map)
                    # to_4bits(last_h - 1, h - 1, self.bits_imp_map)
                    last_h = -1
        
        # process the last single 4 bits
        if last_h != -1:
            to_4bits(last_h, 0, self.bits_imp_map)
            # to_4bits(last_h - 1, 0, self.bits_imp_map)
    
    # 64x28x28  {0, 1}
    def _serialize_codes(self, codes_tensor, imp_tensor):

        # (x,y) the first 
        l = []
        for x in range(codes_tensor.size(1)):
            for y in range(codes_tensor.size(2)):
                for c in range(int(imp_tensor[x][y] + 0.00001)):
                    l.append('1' if codes_tensor[c][x][y] > 0.5 else '0')
                    if len(l) == 8:
                                    
                        write_a_byte(l, self.bits_codes)
                        # pdb.set_trace()           
                        del l[:]

        if len(l):
            print ('remaining', len(l))
            l.extend(['0']*(8 - len(l)))
            write_a_byte(l, self.bits_codes)

    
    def _serialize_size(self):
        # change from B to H, 1 bits to 2 bits, because one image has tall of 2048, then 2048/8 = 256
        self.bits_size.append(struct.pack('<H',self.w))
        self.bits_size.append(struct.pack('<H',self.h))
        to_4bits(self.dw, self.dh, self.bits_size)

    def _serialize_model_id(self):
        self.model_id = struct.pack('<B',0)


    def _write_to_file(self, f, l):
        bs = b''.join(l)
        f.write(bs)
    
    # N,C,H,W
    # Participants will need to submit a file for each test image. 
    def serialize(self, batch_imp_tensor, batch_codes_tensor, dw = 0, dh = 0):
        assert batch_imp_tensor.size(0) == batch_codes_tensor.size(0)
        batch_size = batch_imp_tensor.size(0)
        assert batch_size == 1  # 暂时这么做，因为各个图片的size不一致
        f = open(self.s_f, 'wb')
        
        # serialize model_id : off
        # self._serialize_model_id()
        # self._write_to_file(f, [self.model_id])
        
        
        del self.bits_size[:]
        self.w, self.h = batch_imp_tensor[0][0].size()
        # print('self.w, self.h = ', self.w, self.h)
        self.dw, self.dh = dw, dh
        self._serialize_size()
        self._write_to_file(f, self.bits_size)

        del self.bits_codes[:]
        del self.bits_imp_map[:]
        imp_tensor = batch_imp_tensor[0][0]
        codes_tensor = batch_codes_tensor[0]

        self._serialize_imp(imp_tensor)
        self._serialize_codes(codes_tensor, imp_tensor)
        
        # print ('len of imp =', self.bits_imp_map)
        # print('len of codes =', self.bits_codes)

        self._write_to_file(f, self.bits_imp_map)
        self._write_to_file(f, self.bits_codes)

        f.close()

    def parse(self, file_path):
        f = open(file_path, 'rb')

        try:
            # parse model_id : off
            # mode_id = f.read(1)
            # assert model_id == 0
            # print ('model id is', model_id)

            w = f.read(2)
            h = f.read(2)
            d = f.read(1)

          
            self.w = struct.unpack('<H', w)[0]
            self.h = struct.unpack('<H', h)[0]
            self.dw, self.dh = from_4bits(d)

            # print (self.w, self.h, self.dw, self.dh)

            spatial_size = self.w * self.h
       
            target_imp_tensor = t.Tensor(spatial_size)
            target_codes_tensor = t.zeros(1, 64, self.w, self.h) # 64 is the channel

            cnt = 0
    
            while True:
                if cnt >= spatial_size:
                    break
    
                byte = f.read(1)
                cnt += 2
                x1, x2 = from_4bits(byte)

                if cnt > spatial_size:
                    # print ('x2 is redundance')
                    print (x1)
                    target_imp_tensor[cnt - 2] = x1 + 1
                else:
                    target_imp_tensor[cnt - 2] = x1 + 1
                    target_imp_tensor[cnt - 1] = x2 + 1
      
            
            target_imp_tensor = target_imp_tensor.view(1, 1, self.w, self.h)
            # print (target_imp_tensor.sum())
            
            codes = f.read()
            z_o_list = []
            for i in range(len(codes)):
                z_o_list.extend(read_a_byte(codes[i:i+1]))                        
            
            cur_pos = 0
            # print("len =", len(z_o_list))

            for x in range(target_codes_tensor.size(2)):
                for y in range(target_codes_tensor.size(3)):
                    height = int(target_imp_tensor[0][0][x][y] * 4 + 0.00001)
                    for c in range(height):
                        try:
                            target_codes_tensor[0][c][x][y] = z_o_list[cur_pos]
                            cur_pos += 1
                        except Exception as e:
                            # print (cur_pos)
                            pdb.set_trace()
                            # print (e)
            # print (target_codes_tensor)

        finally:
            f.close()
        return target_codes_tensor




def test_Serializer():
    serializer = Serializer('tmp_file')
    # imp map
    imp_map_tensor = t.Tensor(
    [
        [
            [
                [8, 8, 8],
                [8, 4, 4],
                [4, 0, 4]
            ],
            # [
            #     [0, 8],
            #     [4, 16]
            # ]
        ]
    ]
    )

    t.manual_seed(58)
    code_tensor = t.randn(1,8,3,3)
    code_tensor[code_tensor < 0] = 0
    code_tensor[code_tensor > 0] = 1

    # print (imp_map_tensor)
    print (code_tensor)
    # serializer.serialize_imp(imp_map_tensor)
    # serializer.serialize_codes(code_tensor, imp_map_tensor)

    # print (serializer.bits_imp_map)
    # print (serializer.bits_codes)
    # test = serializer.bits_codes

    serializer.serialize(imp_map_tensor, code_tensor)

    serializer.parse('tmp_file')
    # pdb.set_trace()

    # print (from_4bits(serializer.bits_imp_map[0]))
    # print (from_4bits(serializer.bits_imp_map[1]))    

if __name__ == '__main__':
    test_Serializer()
            





        



