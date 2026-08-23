#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>

__global__ void MatrixMulKernel(float *M, float *N, float *P, int m,int n, int o) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += M[row * n + i] * N[i * o + col];
        }
        P[row * o + col] = sum;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor matrixMul(torch::Tensor M, torch::Tensor N) {
    assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
    assert(M.dtype == torch::kFloat32 && N.dtype == torch::kFloat32);
    assert(M.size(1) == N.size(0));

    int m = M.size(0);
    int n = M.size(1);
    int o = N.size(1);

    auto P = torch::empty({m, o}, torch::TensorOptions().dtype(N.dtype()).device(N.device()));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(cdiv(o, threadsPerBlock.x), cdiv(m, threadsPerBlock.y));

    MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), m, n, o);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return P;
}