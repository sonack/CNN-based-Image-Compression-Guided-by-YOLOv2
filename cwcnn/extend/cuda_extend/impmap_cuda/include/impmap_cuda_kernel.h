#ifndef _IMPMAP_CUDA_KERNEL
#define _IMPMAP_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void impmap_forward_cuda(float *bottom_data, float *top_data, float *height_map, int N, int C, int H, int W, int n, int L, cudaStream_t stream);
void impmap_backward_cuda(float *top_diff, float *bottom_diff, float *height_map, int N, int C, int H, int W, int n, int L, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#endif