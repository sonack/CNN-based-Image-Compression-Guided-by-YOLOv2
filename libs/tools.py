#coding:utf8
from __future__ import print_function
import torch as t

# return:
# 0: use 1 gpu, no data parallel
# 1: use data parallel
# -1: use cpu
def multiple_gpu_process(model):
    use_data_parallel = 0
    print ('Process multiple GPUs...')
    if t.cuda.is_available():
        print ('Cuda enabled, use GPU instead of CPU.')
    else:
        print ('Cuda not enabled!')
        use_data_parallel = -1
        return model, use_data_parallel
    if t.cuda.device_count() > 1:
        print ('Use', t.cuda.device_count(),'GPUs.')
        model = t.nn.DataParallel(model).cuda()
        use_data_parallel = 1
    else:
        print ('Use single GPU.')
        model.cuda()
    return model, use_data_parallel