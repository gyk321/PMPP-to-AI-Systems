#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>

__global__ void MatrixMulKernel(float *M, float *N, float *P, int m,int n, int o) {
  // 计算出当前线程负责的全局行索引 row 和列索引 col
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 矩阵M的维度是 m * n（m 行，n 列）矩阵 N 的维度是 n × o（n 行，o 列）
  // 因此，row 的有效范围应该是 0 到 m-1，col 的有效范围应该是 0 到 o-1
  if (row < m && col < o) {
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

// 矩阵乘法函数:接收两个输入矩阵 M 和 N，返回它们的乘积 P
torch::Tensor matrixMul(torch::Tensor M, torch::Tensor N) {
  // 断言输入矩阵都在 CUDA 设备上，并且数据类型为 float32
  assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
  assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32);
  assert(M.size(1) == N.size(0));   // 确保 M 的列数等于 N 的行数

  // 矩阵M的维度是 m * n（m 行，n 列）矩阵 N 的维度是 n × o（n 行，o 列）
  int m = M.size(0);
  int n = M.size(1);
  int o = N.size(1);

  auto P = torch::empty({m, o}, torch::TensorOptions().dtype(N.dtype()).device(N.device()));

  // 设定每个 Block 包含 16 * 16 个线程
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(cdiv(o, threadsPerBlock.x), cdiv(m, threadsPerBlock.y));

  // data_ptr() 是 PyTorch C++ 接口提供的一个核心方法。它的作用是剥离掉张量的所有高级外壳，直接提取出存储实际数据的底层连续内存块的首地址
  // <float>返回一个 float* 类型的指针，正好匹配核函数 MatrixMulKernel 所需的参数类型
  MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), m, n, o);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return P;
}