import torch as th
from torch.autograd import Variable

class LimuRateLossFunction(th.autograd.Function):
    def __init__(self, r, weight):
        super(LimuRateLossFunction, self).__init__()
        self.r = r
        self.weight = weight
    
    def forward(ctx, x):
        # print ('x.type', type(x))
        # assert x.size(1) == 1, 'channel must be 1!'
        ctx.shape = x.size()
        # print('input_x forward is',x.size())
        # ctx.mask = [0 for i in range(ctx.shape[0])]
        ctx.rate = x.mean()
        loss_value = ctx.weight * (ctx.rate - ctx.r) if ctx.rate > ctx.r else 0
        loss = th.cuda.FloatTensor([loss_value]) if type(x) == th.cuda.FloatTensor else th.FloatTensor([loss_value])
        # for i in range(ctx.shape[0]):
        #     mean_pij = x[i].mean()
        #     # print('mean pij', mean_pij)
        #     if mean_pij > ctx.r:
        #         ctx.mask[i] = 1
        #         loss = loss + ctx.weight * (mean_pij - ctx.r)
        return loss


    def backward(ctx, grad_y):
        # print ('grad_y.type', type(grad_y))
        grad_x = th.zeros(*ctx.shape)
        if type(grad_y) == th.cuda.FloatTensor:
            grad_x = grad_x.cuda()
        # for i in range(ctx.shape[0]):
        #     if ctx.mask[i]:
        #         grad_x[i,:,:,:] = ctx.weight
        if ctx.rate > ctx.r:
            grad_x.fill_(ctx.weight)
        return grad_x

class LimuRateLoss(th.nn.Module):
    def __init__(self, r, weight):
        super(LimuRateLoss, self).__init__()
        self.r = r
        self.weight = weight
    def forward(self, x):
        return LimuRateLossFunction(self.r, self.weight)(x)




def test():
    rate_loss = LimuRateLoss(r=0.5,weight=0.2)
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