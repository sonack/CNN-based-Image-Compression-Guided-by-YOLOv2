import torch as th
from torch.autograd import Variable

class MyRound(th.autograd.Function):
    def forward(ctx, input):
        output = th.zeros_like(input)
        output[input >= 0.5] = 1
        return output
    
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Round(th.nn.Module):
    def forward(self, input):
        return MyRound()(input)

def test():
    rand_input = th.sigmoid(th.randn(5,5))
    print('rand_input ',rand_input)
    x = Variable(rand_input, requires_grad=True)
    # RM = Round()
    test_func = MyRound()
    y = test_func(x)
    print('y')
    print(y)
    # print('x')
    # print(x)
    print(y.grad_fn)
    z = test_func(x)
    print('z')
    print(z)
    print ("??",z.grad_fn == test_func)
    print(z.grad_fn)

    rand_grad = th.randn(5,5)
    print('rand_grad ', rand_grad)
    y.backward(rand_grad)
    print ('back for 1st time.')
    print ('x.grad',x.grad)

    z.backward(rand_grad)
    print ('back for 2nd time.')
    print('x.grad is')
    print(x.grad)

if __name__ == "__main__":
    test()