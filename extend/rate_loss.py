import torch as th
from torch.autograd import Variable

class RateLossFunction(th.autograd.Function):
    def __init__(self, r, weight):
        super(RateLossFunction, self).__init__()
        self.r = r
        self.weight = weight
    
    def forward(ctx, x):
        # print ('x.type', type(x))
        assert x.size(1) == 1, 'channel must be 1!'
        ctx.shape = x.size()
        ctx.mask = [0 for i in range(ctx.shape[0])]
        loss = th.cuda.FloatTensor([0]) if type(x) == th.cuda.FloatTensor else th.FloatTensor([0])
        for i in range(ctx.shape[0]):
            mean_pij = x[i].mean()
            # print('mean pij', mean_pij)
            if mean_pij > ctx.r:
                ctx.mask[i] = 1
                loss = loss + ctx.weight * (mean_pij - ctx.r)
        return loss


    def backward(ctx, grad_y):
        # print ('grad_y.type', type(grad_y))
        grad_x = th.zeros(*ctx.shape)
        if type(grad_y) == th.cuda.FloatTensor:
            grad_x = grad_x.cuda()
        for i in range(ctx.shape[0]):
            if ctx.mask[i]:
                grad_x[i,:,:,:] = ctx.weight
        return grad_x

class RateLoss(th.nn.Module):
    def __init__(self, r, weight):
        super(RateLoss, self).__init__()
        self.r = r
        self.weight = weight
    def forward(self, x):
        return RateLossFunction(self.r, self.weight)(x)




def test():
    rate_loss = RateLoss(r=0.5,weight=0.2)
    rand_input = th.sigmoid(th.randn(2,1,5,5))
    x = Variable(rand_input, requires_grad=True)
    y = rate_loss(x)
    print (x)
    print (y)
    grad_y = th.ones_like(y)
    y.backward(grad_y)
    print (x.grad)

if __name__ == '__main__':
    test()