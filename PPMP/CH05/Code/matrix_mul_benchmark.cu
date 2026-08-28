/*
  用于实现和基准测试（Benchmark）两种不同的矩阵乘法（Matrix Multiplication）算法在 GPU 上的性能差异。
  它主要对比了朴素的矩阵乘法和使用了共享内存优化的分块（Tiled）矩阵乘法。
*/

#include<cuda_runtime.h>
#include<cmath>
#include<iomanip>
#include<iostream>

// 定义了分块矩阵乘法中，每个“数据块”（Tile）的长宽为 32。
#define TILE_WIDTH 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// 在多次运行相同的内核函数时，如果不清理 GPU 的 L2 缓存，后续的运行会因为数据已经缓存在 L2 中而异常得快。
void clear_l2(){
  static int l2_clear_size = 0;
  static unsigned char *gpu_scratch_l2_clear = NULL;
  if(!gpu_scratch_l2_clear){
    cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
    l2_clear_size *= 2; // double the size to ensure we clear the cache
    gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
  }
  gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

// 基础版（朴素）矩阵乘法
__global__ void MatrixMulKernel(float *M,float *N,float *P,int m,int n,int o){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(row < m && col < o){
    float Pvalue = 0;
    for(int k = 0; k < n; ++k){
      Pvalue += M[row * n + k] * N[k * o + col];
    }
    P[row * o + col] = Pvalue;
  }
}

// 基于共享内存的分块矩阵乘法
__global__ void TiledMatrixMulKernel(float *M,float *N,float *P,int m,int n,int o){
  // 共享内存声明,共享内存的读取速度远高于全局内存
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;

  for(int ph = 0; ph < ceil((float)n/TILE_WIDTH); ++ph){
    // 线程块内的每个线程顺手从全局内存中抓取一个元素，存入共享内存 Mds 和 Nds 中。
    if(Row < m && (ph * TILE_WIDTH + tx) < n)
      Mds[ty][tx] = M[Row * n + ph * TILE_WIDTH + tx];
    else
      Mds[ty][tx] = 0.0;

    if(Col < o && (ph * TILE_WIDTH + ty) < n)
      Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + Col];
    else
      Nds[ty][tx] = 0.0;

    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; ++k)
      // 从高速的共享内存 Mds 和 Nds 中读取数据，进行局部点积计算，并累加到寄存器变量 Pvalue 中
      Pvalue += Mds[ty][k] * Nds[k][tx];

    __syncthreads();
  }

  if(Row < m && Col < o)
    P[Row*o + Col] = Pvalue;
}

void matrixMul(float *M,float *N,float *P,int m,int n,int o){
  float *d_M, *d_N, *d_P;
  cudaMalloc((void**)&d_M, m * n * sizeof(float));
  cudaMalloc((void**)&d_N, n * o * sizeof(float));
  cudaMalloc((void**)&d_P, m * o * sizeof(float));

  cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

  cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
}

/* CUDA 主机端（Host）封装函数，用于在 GPU 上执行分块矩阵乘法
    M: 输入矩阵 M，存储在 CPU（Host）端，大小为 m x n。
    N: 输入矩阵 N，存储在 CPU（Host）端，大小为 n x o。
    P: 输出矩阵 P，存储在 CPU（Host）端，大小为 m x o。
    m, n, o: 分别代表矩阵的维度参数（行数和列数）。*/
void matrixMulTiling(float *M,float *N,float *P,int m,int n,int o){
  // GPU 无法直接读取 CPU 上的内存，因此需要在 GPU 的全局内存中为这三个矩阵开辟空间
  // d_M, d_N, d_P 是指向 GPU 显存地址的指针
  float *d_M, *d_N, *d_P;
  cudaMalloc((void**)&d_M, m * n * sizeof(float));
  cudaMalloc((void**)&d_N, n * o * sizeof(float));
  cudaMalloc((void**)&d_P, m * o * sizeof(float));

  cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

  // 定义了每个 Block 中包含的线程数量。
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  // 定义了需要启动多少个 Block
  dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

  TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

  // 检查在启动核函数时是否发生了同步错误
  gpuErrchk(cudaPeekAtLastError());
  // 强制 CPU 等待 GPU 完成该核函数的所有计算。
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
}

// 用于精准测试 CUDA 核函数执行时间的基准测试（Benchmark）函数
// void (*func)(...): 这是一个函数指针。它允许你将不同的矩阵乘法作为参数传进来进行测试，
float benchmark(void (*func)(float*, float*, float*, int, int, int), float *M, float *N, float *P, int m, int n, int o, int warmup = 25,int reps = 100){
  // 为什么需要预热？ 
  // 在 GPU 编程中，第一次启动 CUDA 环境、分配资源或加载指令缓存时，通常会有额外的延迟（冷启动开销）。
  // 预热循环通过先空跑几次，让 GPU 进入“工作状态”，确保后续测量的纯粹是算法本身的执行时间。
  for(int i = 0; i < warmup; ++i){
    func(M, N, P, m, n, o);
  }

  // 初始化 CUDA 事件计时器；创建了两个 CUDA 事件：iterStart 和 iterStop
  // 为什么不用传统的 CPU 计时器（如 std::chrono）？ 因为 CUDA 核函数的启动是异步的。
  // 如果用 CPU 计时，CPU 发出启动核函数的指令后会立刻执行下一行代码，此时 GPU 可能还没算完，导致测出的时间非常小且不准确。
  // 使用 CUDA 事件机制可以在 GPU 执行流中插入时间戳，获取最准确的 GPU 硬件级执行时间。
  cudaEvent_t iterStart,iterStop;
  cudaEventCreate(&iterStart);
  cudaEventCreate(&iterStop);

  float totalTime = 0.0f;
  for(int i = 0; i < reps; ++i){
    cudaEventRecord(iterStart);         // 记录开始时间戳
    func(M, N, P, m, n, o);             // 执行你要测试的矩阵乘法函数
    cudaEventRecord(iterStop);          // 记录结束时间戳
    cudaEventSynchronize(iterStop);     // 强制 CPU 阻塞并等待，直到 GPU 执行流运行到 iterStop 这个标记。这确保了核函数完全执行完毕。

    float iterTime = 0.0f;
    cudaEventElapsedTime(&iterTime, iterStart, iterStop);   // 计算两个事件之间经过的时间，并将结果（毫秒）存入 iterTime。
    totalTime += iterTime;
    clear_l2();                         // 清空 GPU 的 L2 缓存
  }
  // 销毁创建的 CUDA 事件对象，释放资源，防止内存泄漏
  cudaEventDestroy(iterStart);
  cudaEventDestroy(iterStop);
  return totalTime / reps;              // 将累加的总时间除以重复次数 reps，返回单次运行的平均耗时
}

bool allclose(float *M,float *N,int m,int n,float tol=1e-05){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      if(std::fabs(M[i * n + j] - N[i * n + j]) > tol){
        return false;
      }
    }
  }
  return true;
}

void printMatrix(float *M,int rows,int cols){
  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      std::cout << std::setw(6) << M[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(){
  int m = 4096, n = 4096, o = 4096;

  float *M = new float[m * n];
  float *N = new float[n * o];
  float *P1 = new float[m * o];
  float *P2 = new float[m * o];

  for(int i = 0; i < m * n; ++i){
    M[i] = static_cast<float>(1.0);
  }

  for(int i = 0; i < n * o; ++i){
    N[i] = static_cast<float>(1.5);
  }

  float avgTimeMatrixMulTiling = benchmark(matrixMulTiling, M, N, P1, m, n, o);
  float avgTimeMatrixMul = benchmark(matrixMul, M, N, P2, m, n, o);
  
  std::cout << "Average time for tiled matrix multiplication: " << avgTimeMatrixMulTiling << " ms" << std::endl;
  std::cout << "Average time for standard matrix multiplication: " << avgTimeMatrixMul << " ms" << std::endl;

  bool same = allclose(P1, P2, m, o);
  std::cout << "Are the results of both methods the same? " << (same ? "Yes" : "No") << std::endl;
  delete[] M;
  delete[] N;
  delete[] P1;
  delete[] P2;

  return 0;
}