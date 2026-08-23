#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>

__global__ void matrixMulKernel(float *B,float *C,float *result,int vector_size,int matrix_rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < matrix_rows) {
    float sum = 0.0f;
    for (int j = 0; j < vector_size; ++j) {
      sum += B[i * vector_size + j] * C[j];
    }
    result[i] = sum;
  }
}

torch::Tensor matrix_vector_multiplication(torch::Tensor B, torch::Tensor C) {
  // 确保输入张量在 CUDA 设备上，并且数据类型为 float32
  assert(B.device().type() == torch::kCUDA && C.device().type() == torch::kCUDA); // Ensure tensors are on CUDA device
  assert(B.dtype() == torch::kFloat32 && C.dtype() == torch::kFloat32); // Ensure tensors are of type float
  assert(B.size(1) == C.size(0)); // Ensure matrix and vector dimensions are compatible

  // 获取维度与分配输出内存
  int matrix_rows = B.size(0);
  int vector_size = B.size(1);

  // 创建输出张量
  auto result = torch::empty({matrix_rows}, torch::TensorOptions().dtype(torch::kFloat32).device(B.device()));

  // 定义 CUDA 核函数的线程块和网格大小
  int threads_per_block = 32;
  int blocks_per_grid = (matrix_rows + threads_per_block - 1) / threads_per_block;

  // 调用 CUDA 核函数进行矩阵向量乘法
  matrixMulKernel<<<blocks_per_grid, threads_per_block,0,torch::cuda::getCurrentCUDAStream()>>>(
    B.data_ptr<float>(), C.data_ptr<float>(), result.data_ptr<float>(), vector_size, matrix_rows
  );

  // 检查 CUDA 错误，捕获内核启动时的异步错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return result;
}