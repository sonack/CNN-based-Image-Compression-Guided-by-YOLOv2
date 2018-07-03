from __future__ import division  # py2
from __future__ import print_function
import caffe
from math import ceil
caffe.set_device(0)
caffe.set_mode_gpu()

solver_path = ""

solver = caffe.get_solver(solver_path)

weights = ""

solver.net.copy_from(weights)

test_imgs_num = 241 # prime
test_batch_size = 1
test_iter = ceil(test_imgs_num / test_batch_size)
for it in range(test_iter)
    solver.test_nets[0].forward()
    _test_mse_loss += solver.test_nets[0].blobs['loss']
    _test_rate_disp += solver.test_nets[0].params['imap'][0].data[0,0,0,0]

test_mse_loss = _test_mse_loss / test_iter
test_rate_disp = _test_rate_disp / test_iter
print ('mse loss = %f, rate_disp = %f' % (test_mse_loss, test_rate_disp))

