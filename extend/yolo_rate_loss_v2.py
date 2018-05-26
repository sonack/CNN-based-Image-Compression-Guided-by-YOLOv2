import torch as th
from torch.autograd import Variable

class YoloRateLossFunctionV2(th.autograd.Function):
    # r is a tensor, the same size as imp map
    # r0 is the base threshold, is a scalar
    # r is inner 0, outer 1
    # weight is original scalar weight
    def __init__(self, r, r0, weight):
        super(YoloRateLossFunctionV2, self).__init__()
        self.r = r
        self.r0 = r0
        self.weight = weight
    
    def forward(ctx, x):
        # print ('x.type', type(x))
        # print ('r.type', type(ctx.r))
        assert x.size(1) == 1, 'channel must be 1!'
        
        # ctx.save_for_backward(x)
        # r = ctx.r0 * ctx.r
        ctx.shape = x.size()
        ctx.mask = [0 for i in range(ctx.shape[0])]
        loss = th.cuda.FloatTensor([0]) if type(x) == th.cuda.FloatTensor else th.FloatTensor([0])
        for i in range(ctx.shape[0]):
            # mean_pij = x[i].mean()
            # bg region mean pij
            if ctx.r[i].sum() == 0:
                mean_pij = 0
            else:
                mean_pij = (x[i]*ctx.r[i]).sum() / (ctx.r[i].sum())
            # print('mean pij', mean_pij)
            if mean_pij > ctx.r0:
                # print ('mask = 1, i =',i)
                ctx.mask[i] = 1
                # loss = loss + (mean_pij - ctx.r0)
                loss = loss + ctx.weight * mean_pij
        return loss


    def backward(ctx, grad_y):
        # print ('grad_y.type', type(grad_y))
        # input_x, = ctx.saved_tensors
        grad_x = th.zeros(*ctx.shape)
        if type(grad_y) == th.cuda.FloatTensor:
            grad_x = grad_x.cuda()
        for i in range(ctx.shape[0]):
            if ctx.mask[i]:
                # print ('x.shape =', input_x[i].shape)
                # print ('r.shape =', ctx.r[i].shape)
                # x_gt_r = input_x[i] > ctx.r[i]
                # print ('back i =',i)
                # print ('x>r',x_gt_r)
                # print ('x>r.shape =', x_gt_r.shape)
                grad_x[i][ctx.r[i] == 1] = ctx.weight
        return grad_x

class YoloRateLossV2(th.nn.Module):
    def __init__(self, r, r0, weight):
        super(YoloRateLossV2, self).__init__()
        # r must be a tensor, not Variable
        self.r = r.data if isinstance(r, th.autograd.variable.Variable) else r
        self.r0 = r0
        self.weight = weight
    def forward(self, x):
        return YoloRateLossFunctionV2(self.r, self.r0, self.weight)(x)




def test():

    mask_r = th.Tensor([[1,0],[0,0]]).unsqueeze(0)
    r = Variable(mask_r, requires_grad=False)
    rate_loss = YoloRateLossV2(r=r, r0=0.2, weight=0.233)
    rand_input = th.sigmoid(th.randn(1,1,2,2))
  
    x = Variable(rand_input, requires_grad=True)
    print('x0.mean',x[0].mean())
    # print('x1.mean',x[1].mean())
    y = rate_loss(x)
    print ('mask_r', mask_r)
    print ('x',x)
    print ('y',y)
    grad_y = th.ones_like(y)
    y.backward(grad_y)
    print (x.grad)

if __name__ == '__main__':
    test()