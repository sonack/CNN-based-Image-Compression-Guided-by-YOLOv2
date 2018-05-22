import torch as th
from torch.autograd import Variable

class MyImpMap(th.autograd.Function):
    # @staticmethod
    def forward(ctx, input, L, n):
        '''
         input: channels is 1, range is in (0,1)
         L: levels  16/32
         n: 64/128

         assert:
         1. channel == 1
         2. n % L == 0
        '''
        L = int(L)
        n = int(n)
        shape = input.size()
        assert shape[1] == 1
        assert n % L == 0

        input_plane = input.view(shape[0],shape[2],shape[3])
        height_plane = (1 + (input_plane * L + 0.00001).type(th.IntTensor)) * n / L
        output = th.zeros(shape[0],n,shape[2],shape[3])
        for n_ in range(shape[0]):
            for h_ in range(shape[2]):
                for w_ in range(shape[3]):
                    for c_ in range(int(height_plane[n_, h_, w_])):
                        output[n_, c_, h_, w_] = 1.0
        return output

    # @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.size()
        grad_input = th.zeros(shape[0],1,shape[2],shape[3])
        for n_ in range(shape[0]):
            for h_ in range(shape[2]):
                for w_ in range(shape[3]):
                    for c_ in range(shape[1]):
                        grad_input[n_, 0, h_, w_] += grad_output[n_, c_, h_, w_]
        return grad_input, None, None

class ImpMap(th.nn.Module):
    def __init__(self, L=16, n=64):
        super(ImpMap, self).__init__()
        self.L = L
        self.n = n
    def forward(self, input):
        return MyImpMap()(input, Variable(th.Tensor([self.L])), Variable(th.Tensor([self.n])))

def test():
    th.manual_seed(233)
    rand_input = th.sigmoid(th.randn(2,1,5,5))
    x = Variable(rand_input, requires_grad=True)
    print('input.x =', x)
    IMM = ImpMap(L=4, n=4)
    y = IMM(x)
    print(type(y))
    print(y.size())
    print(y)
    print(y.grad_fn)
    rand_grad = th.randn(2,4,5,5)
    print('rand_grad is', rand_grad)
    y.backward(rand_grad)
    print('x.grad is')
    print(x.grad)


if __name__ == '__main__':
    test()
     