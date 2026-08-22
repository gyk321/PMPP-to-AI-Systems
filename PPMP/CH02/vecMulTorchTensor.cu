#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>

__global__ void vecMulKernel(const float *A, const float *B, float *C, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N){
    C[i] = A[i] * B[i];
  }
}

// 旨在通过自定义的 CUDA 核函数（Kernel）来实现两个张量（向量）的逐元素相乘
torch::Tensor vector_multiplication(torch::Tensor A, torch::Tensor B){
  assert(A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA);
  assert(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32);
  assert(A.size(0) == B.size(0));

  // 准备输出张量
  int N = A.size(0);
  auto C = torch::empty({N},torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  // 配置 CUDA 线程和线程块
  int threads_per_block = 256;
  int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  // 启动 CUDA 核函数进行计算
  // 使用了 torch::cuda::getCurrentCUDAStream() 确保该核函数在 PyTorch 当前正在使用的计算流中异步执行，避免同步冲突
  vecMulKernel<<<number_of_blocks, threads_per_block,0,torch::cuda::getCurrentCUDAStream()>>>(
    A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),N);

  // 检查 CUDA 错误
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;

}
