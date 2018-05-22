#include <THC/THC.h>
#include "impmap_cuda_kernel.h"

extern THCState *state;

int impmap_forward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor, THCudaTensor *height_map_tensor, int N, int C, int H, int W, int n, int L) {
    float *bottom_data = THCudaTensor_data(state, bottom_tensor);
    float *top_data = THCudaTensor_data(state, top_tensor);
    float *height_map = THCudaTensor_data(state, height_map_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);
    impmap_forward_cuda(bottom_data, top_data, height_map, N, C, H, W, n, L, stream);
    return 1;
}

int impmap_backward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor, THCudaTensor *height_map_tensor, int N, int C, int H, int W, int n, int L) {
    float *bottom_diff = THCudaTensor_data(state, bottom_tensor);
    float *top_diff = THCudaTensor_data(state, top_tensor);
    float *height_map = THCudaTensor_data(state, height_map_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);
    impmap_backward_cuda(top_diff, bottom_diff, height_map, N, C, H, W, n, L, stream);
    return 1;
}
