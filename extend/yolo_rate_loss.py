import torch as th
from torch.autograd import Variable

class YoloRateLossFunction(th.autograd.Function):
    # r is a tensor, the same size as imp map
    # r0 is the base threshold, is a scalar
    def __init__(self, r, r0, weight):
        super(YoloRateLossFunction, self).__init__()
        self.r = r * r0
        self.r0 = r0
        self.weight = weight
        self.weight_plane = weight / r
    
    def forward(ctx, x):
        # print ('x.type', type(x))
        # print ('r.type', type(ctx.r))
        assert x.size(1) == 1, 'channel must be 1!'
        
        ctx.save_for_backward(x)
        # r = ctx.r0 * ctx.r
        ctx.shape = x.size()
        ctx.mask = [0 for i in range(ctx.shape[0])]
        loss = th.cuda.FloatTensor([0]) if type(x) == th.cuda.FloatTensor else th.FloatTensor([0])
        for i in range(ctx.shape[0]):
            # mean_pij = x[i].mean()
            sub_thresh_pij = (x[i] - ctx.r[i]).clamp(min=0)
            mean_pij = x[i].mean()
            # print('mean pij', mean_pij)
            if mean_pij > ctx.r0:
                # print ('mask = 1, i =',i)
                ctx.mask[i] = 1
                # loss = loss + (mean_pij - ctx.r0)
                loss = loss + (ctx.weight_plane * sub_thresh_pij).mean()
        return loss


    def backward(ctx, grad_y):
        # print ('grad_y.type', type(grad_y))
        input_x, = ctx.saved_tensors
        grad_x = th.zeros(*ctx.shape)
        if type(grad_y) == th.cuda.FloatTensor:
            grad_x = grad_x.cuda()
        for i in range(ctx.shape[0]):
            if ctx.mask[i]:
                # print ('x.shape =', input_x[i].shape)
                # print ('r.shape =', ctx.r[i].shape)
                x_gt_r = input_x[i] > ctx.r[i]
                # print ('back i =',i)
                # print ('x>r',x_gt_r)
                # print ('x>r.shape =', x_gt_r.shape)
                grad_x[i][x_gt_r] = ctx.weight_plane[i][x_gt_r]
        return grad_x

class YoloRateLoss(th.nn.Module):
    def __init__(self, r, r0, weight):
        super(YoloRateLoss, self).__init__()
        # r must be a tensor, not Variable
        self.r = r.data if hasattr(r,'data') else r
        self.r0 = r0
        self.weight = weight
    def forward(self, x):
        return YoloRateLossFunction(self.r, self.r0, self.weight)(x)




def test():

    mask_r = th.Tensor([[2,1],[1,1]]).unsqueeze(0)
    r = Variable(mask_r, requires_grad=False)
    rate_loss = YoloRateLoss(r=mask_r, r0=0.5, weight=1)
    rand_input = th.sigmoid(th.randn(2,1,2,2))
  
    x = Variable(rand_input, requires_grad=True)
    print('x0.mean',x[0].mean())
    print('x1.mean',x[1].mean())
    y = rate_loss(x)
    print ('mask_r', mask_r)
    print ('x',x)
    print ('y',y)
    grad_y = th.ones_like(y)
    y.backward(grad_y)
    print (x.grad)

if __name__ == '__main__':
    test()