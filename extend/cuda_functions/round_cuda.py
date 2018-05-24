import torch as th
from torch.autograd import Variable, Function

include_path = "/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/extend"
import sys
if include_path not in sys.path:
    sys.path.append(include_path)

from round import round_forward_wrapper, round_backward_wrapper

class RoundCudaFunction(Function):
    ''' 
        Pytorch Function wrapper of cuda implementation of round layer
    '''
    def forward(self, x):
        y = th.zeros_like(x)
        round_forward_wrapper(x, y, x.numel())
        return y
    
    def backward(self, grad_y):
        grad_x = th.zeros_like(grad_y)
        round_backward_wrapper(grad_x, grad_y, grad_y.numel())
        return grad_x

class RoundCuda(th.nn.Module):
    def forward(self, x):
        return RoundCudaFunction()(x)

def test():
    inp_tensor = th.Tensor([[1,2],[-1,0]]).cuda()
    inp = th.sigmoid(inp_tensor)
    x = Variable(inp, requires_grad = True)
    round_= RoundCuda()
    y = round_(x)
    print (x)
    print (y)
    
    y.backward(th.cuda.FloatTensor([[1,2],[3,4]]))
    print (x.grad)

if __name__ == '__main__':
    test()
