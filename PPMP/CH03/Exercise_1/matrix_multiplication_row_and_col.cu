#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>

// 每一个 CUDA 线程负责计算输出矩阵 P 的一整行(性能较差)
// 它们的内存访问步长很大（跨度为 size），
// 这导致了非合并的内存访问（Uncoalesced memory access），极大地浪费了 GPU 的显存带宽，导致执行效率极低。
__global__ void matrixMulRowKernel(float* M, float* N, float* P, int size) {
  // 计算当前线程的全局一维索引，并将其直接作为矩阵的行号
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  // 边界检查
  if (row < size) {
    for (int col = 0; col < size; ++col) {
      float sum = 0.0f;
      // 遍历该行的所有列
      for (int k = 0; k < size; ++k) {
          /* 线程是以线程束（Warp）为单位执行的
            在 matrixMulRowKernel 中：线程 0 读 M[0]，线程 1 读 M[size]，线程 2 读 M[2 * size]
          */
          sum += M[row * size + k] * N[k * size + col];
      }
      P[row * size + col] = sum;
    }
  }
}

// 每一个 CUDA 线程负责计算输出矩阵 P 的一整列(性能较好)
// 这种访问模式是完美的合并内存访问（Coalesced memory access）。
// 只需一次内存事务就能取回/写入整个 Warp 所需的数据，因此它的运行速度通常会比按行计算快很多倍。
__global__ void matrixMulColKernel(float* M, float* N, float* P, int size) {
  // 计算出的全局一维索引被用作矩阵的 列号
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 边界检查
  if (col < size) {
    // 遍历该列的所有行
    for(int row = 0; row < size; ++row) {
      float sum = 0.0f;
      // 计算点积。同一个 Warp 中的 32 个线程，读取的是 N 中完全连续的内存地址。
      // 线程 0： 代入 k = 0, col = 0，地址为 0 * size + 0 = 0，读取 N[0]。
      // 线程 1： 代入 k = 0, col = 1，地址为 0 * size + 1 = 1，读取 N[1]。
      // 线程 2： 读取 N[2]... 依此类推。
      for (int k = 0; k < size; ++k) {
          sum += M[row * size + k] * N[k * size + col];
      }
      P[row * size + col] = sum;
    }
  }
}

// 整数的向上取整除法
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor matrixRowMul(torch::Tensor M, torch::Tensor N) {
  // 确保输入张量在 CUDA 设备上，并且数据类型为 float32
  assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA); // Ensure tensors are on CUDA device
  assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32); // Ensure tensors are of type float
  assert(M.size(0) == M.size(1) && N.size(0) == N.size(1) && M.size(0) == N.size(0)); // Ensure matrices are square

  // 获取了方阵的维度大小,准备输出张量
  const auto size = M.size(0);
  auto P = torch::empty_like(N);

  // CUDA 线程调度配置.包含 16 个线程,定义网格（Grid）中包含多少个线程块。
  dim3 dimBlock(16);
  dim3 dimGrid(cdiv(size, dimBlock.x));

  // 启动 CUDA 核函数
  matrixMulRowKernel<<<dimGrid, dimBlock,0,torch::cuda::getCurrentCUDAStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), size);

  // 错误检查与返回
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return P;
}

torch::Tensor matrixColMul(torch::Tensor M, torch::Tensor N) {
  // 确保输入张量在 CUDA 设备上，并且数据类型为 float32
  assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA); // Ensure tensors are on CUDA device
  assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32); // Ensure tensors are of type float
  assert(M.size(0) == M.size(1) && N.size(0) == N.size(1) && M.size(0) == N.size(0)); // Ensure matrices are square

  // 获取了方阵的维度大小,准备输出张量
  const auto size = M.size(0);
  auto P = torch::empty_like(N);

  // CUDA 线程调度配置.包含 16 个线程,定义网格（Grid）中包含多少个线程块。
  dim3 dimBlock(16);
  dim3 dimGrid(cdiv(size, dimBlock.x));

  // 启动 CUDA 核函数
  matrixMulColKernel<<<dimGrid, dimBlock,0,torch::cuda::getCurrentCUDAStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), size);

  // 错误检查与返回
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return P;
}