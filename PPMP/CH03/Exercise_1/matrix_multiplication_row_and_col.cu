#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>

__global__ void matrixMulRowKernel(float* M, float* N, float* P, int size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < size) {
      float sum = 0.0f;
      for (int k = 0; k < size; ++k) {
          sum += M[row * size + k] * N[k * size + col];
      }
      P[row * size + col] = sum;
  }
}

__global__ void matrixMulColKernel(float* M, float* N, float* P, int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < size) {
      float sum = 0.0f;
      for (int k = 0; k < size; ++k) {
          sum += M[row * size + k] * N[k * size + col];
      }
      P[row * size + col] = sum;
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor matrixRowMul(torch::Tensor M, torch::Tensor N) {
  assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA); // Ensure tensors are on CUDA device
  assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32); // Ensure tensors are of type float
  assert(M.size(0) == M.size(1) && N.size(0) == N.size(1) && M.size(0) == N.size(0)); // Ensure matrices are square
  const auto size = M.size(0);
  auto P = torch::empty_like(N);

  dim3 dimBlock(16);
  dim3 dimGrid(cdiv(size, dimBlock.x));

  matrixMulRowKernel<<<dimGrid, dimBlock,0,torch::cuda::getCurrrentStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return P;
}

torch::Tensor matrixColMul(torch::Tensor M, torch::Tensor N) {
  assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA); // Ensure tensors are on CUDA device
  assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32); // Ensure tensors are of type float
  assert(M.size(0) == M.size(1) && N.size(0) == N.size(1) && M.size(0) == N.size(0)); // Ensure matrices are square
  const auto size = M.size(0);
  auto P = torch::empty_like(N);

  dim3 dimBlock(16);
  dim3 dimGrid(cdiv(size, dimBlock.x));

  matrixMulColKernel<<<dimGrid, dimBlock,0,torch::cuda::getCurrrentStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return P;
}