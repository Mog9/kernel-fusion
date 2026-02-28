# rawkernel fused custom implementation

import cupy as cp

fused_kernel_code = r'''
extern "C" __global__
void relu_layernorm_fused(const float* __restrict__ x,
                            float* __restrict__ out,
                            int N,
                            float eps
                            )
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ float shared[256];

    const float* row_ptr = x + row * N;
    float* out_ptr = out + row * N;

    float sum = 0.0f;

    for (int i = tid; i < N; i += block_size) {
        float val = row_ptr[i];
        val = val > 0.0f ? val : 0.0f; //relu
        sum += val;
    }

    shared[tid] = sum;
    __syncthreads();

    //parallel reduction (tree)
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    __syncthreads();
    }
    
    float mean = shared[0] / N;

    //compute variance
    float var_sum = 0.0f;

    for (int i = tid; i < N; i += block_size) {
        float val = row_ptr[i];
        val = val > 0.0f ? val : 0.0f; // relu again
        float diff = val - mean;
        var_sum += diff * diff;
    }   

    shared[tid] = var_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float var = shared[0] / N;
    float inv_std = rsqrtf(var + eps);

    //normalize and write
    for (int i = tid; i < N; i += block_size) {
        float val = row_ptr[i];
        val = val > 0.0f ? val : 0.0f;
        out_ptr[i] = (val - mean) * inv_std;
    }   
}
'''

fused_kernel = cp.RawKernel(fused_kernel_code, "relu_layernorm_fused")

def relu_layernorm_fused(x, eps=1e-5):
    B, N = x.shape
    out = cp.empty_like(x)

    threads_per_block = 256
    blocks_per_grid = B

    fused_kernel(
        (blocks_per_grid, ),
        (threads_per_block,),
        (x, out, N, eps)
    )
    return out
