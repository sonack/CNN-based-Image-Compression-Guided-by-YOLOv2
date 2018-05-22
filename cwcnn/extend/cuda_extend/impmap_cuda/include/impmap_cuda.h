int impmap_forward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor,THCudaTensor *height_map_tensor, int N, int C, int H, int W, int n, int L);
int impmap_backward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor,THCudaTensor *height_map_tensor, int N, int C, int H, int W, int n, int L);
