#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "caffe_cuda_macro.h"
#include "impmap_cuda_kernel.h"

__global__ void impmap_forward_cuda_kernel(const int n_threads, float *bottom_data, float *top_data, float *height_map, int H, int W, int n, int L) {
    CUDA_KERNEL_LOOP(i, n_threads) {
        // + 1 because lquantize option is false
        int fill_bits = (1 + int(bottom_data[i] * L + 0.00001)) * n / L;
        int space = H * W;
        int _n = i / space;
        int _pos = i % space;
        height_map[i] = fill_bits;
        for (int _c = 0; _c < n; _c++) {
            top_data[(_n * n + _c) * space + _pos] = (_c < fill_bits) ? 1.0 : 0.0;
        }
    }
}


__global__ void impmap_backward_cuda_kernel(const int n_threads, float *top_diff, float *bottom_diff, int H, int W, int n) {
    CUDA_KERNEL_LOOP(i, n_threads) {
        int space = H * W;
        int _n = i / space;
        int _pos = i % space;
        bottom_diff[i] = 0;
        for (int _k = 0; _k < n; _k++)
        {
            bottom_diff[i] += top_diff[(_n * n + _k) * space + _pos];
        }
    }
}


void impmap_forward_cuda(float *bottom_data, float *top_data, float *height_map, int N, int C, int H, int W, int n, int L, cudaStream_t stream)
{
    int count = N * C * H * W;    
    impmap_forward_cuda_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data, height_map, H, W, n, L);
    CUDA_POST_KERNEL_CHECK;
}


void impmap_backward_cuda(float *top_diff, float *bottom_diff, float *height_map, int N, int C, int H, int W, int n, int L, cudaStream_t stream)
{
    int count = N * C * H * W;
    impmap_backward_cuda_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_diff, H, W, n);
    CUDA_POST_KERNEL_CHECK;
}