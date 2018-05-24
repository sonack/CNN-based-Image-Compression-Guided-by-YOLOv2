#coding:utf-8

import torch as th
from torch.autograd import Variable, Function

include_path = '/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/extend'
import sys
if include_path not in sys.path:
    sys.path.append(include_path)

from impmap import impmap_forward_wrapper, impmap_backward_wrapper

class ImpMapCudaFunction(Function):
    
    def __init__(self, L, n):
        super(ImpMapCudaFunction, self).__init__()
        self.L = L
        self.n = n
    
    def forward(self, x):
        height_map = th.zeros_like(x)
        shape = list(x.size())
        self.shape = shape[:]
        assert shape[1] == 1, "The channel must be 1!"

        shape[1] = self.n
        y = th.Tensor(*shape).cuda()
        a,b,c,d = x.size()
        impmap_forward_wrapper(x, y, height_map, a, b, c, d, self.n , self.L)
        self.height_map = height_map
        return y, height_map
    
    def backward(self, grad_y, grad_h):
        grad_x = th.cuda.FloatTensor(*self.shape)
        a,b,c,d = self.shape
        impmap_backward_wrapper(grad_x, grad_y, self.height_map, a, b, c, d, self.n, self.L)
        return grad_x


class ImpMapCuda(th.nn.Module):
    def __init__(self, L, n):
        super(ImpMapCuda, self).__init__()
        self.L = L
        self.n = n
    
    def forward(self, x):
        return ImpMapCudaFunction(self.L, self.n)(x)

def test():
    my_imp_map = ImpMapCuda(L=2,n=4)
    rand_input = th.sigmoid(th.randn(1,1,5,5)).cuda()
    x = Variable(rand_input, requires_grad=True)
    y = my_imp_map(x)
    print (x)
    print (y)
    grad_y = th.ones_like(y)
    y.backward(grad_y)
    print (x.grad)

if __name__ == '__main__':
    test()
