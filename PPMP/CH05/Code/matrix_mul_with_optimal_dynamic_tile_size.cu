/*
  用于实现和比较基础版本（Naive）和分块版本（Tiled/Shared Memory）的矩阵乘法。
  程序不仅包含了内核计算逻辑，还包含了动态计算最优线程块大小、性能基准测试（Benchmarking）以及结果正确性验证等工程化实践。
*/

#include<cuda_runtime.h>

#include<algorithm>
#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<iomanip>
#include<iostream>

#define gpuErrchk(ans) do { \
  cudaError_t gpuErr = (ans); \
  if (gpuErr != cudaSuccess) { \
    std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(gpuErr), __FILE__, __LINE__); \
    std::exit(static_cast<int>(gpuErr)); \
  } \
} while (0)

int calculateOptimalTileWidth(int m,int n,int o){
  cudaDeviceProp deviceProp;
  gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));

  int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  int maxBlockDimX = deviceProp.maxThreadsDim[0];
  int maxBlockDimY = deviceProp.maxThreadsDim[1];
  int sharedMemPerBlock = deviceProp.sharedMemPerBlock;

  int minDim = std::min(m, std::min(n, o));
  if (minDim < 1) {
    return 1;
  }

  int tileWidth = static_cast<int>(std::sqrt(static_cast<double>(maxThreadsPerBlock)));
  tileWidth = std::min(tileWidth, std::min(maxBlockDimX, maxBlockDimY));

  int maxTileWidthBySharedMem = static_cast<int>(std::sqrt(static_cast<double>(sharedMemPerBlock) / (2 * sizeof(float))));
  tileWidth = std::min(tileWidth, maxTileWidthBySharedMem);
  if (tileWidth < 1) {
    tileWidth = 1;
  }
  tileWidth = 1 << static_cast<int>(std::floor(std::log2(tileWidth)));

  // Prefer a tile of at least 16 x 16, but never violate device or matrix limits.
  tileWidth = std::max(tileWidth, 16);
  tileWidth = std::min(tileWidth, static_cast<int>(std::sqrt(static_cast<double>(maxThreadsPerBlock))));
  tileWidth = std::min(tileWidth, std::min(maxBlockDimX, maxBlockDimY));
  tileWidth = std::min(tileWidth, maxTileWidthBySharedMem);
  tileWidth = std::min(tileWidth, minDim);

  return std::max(tileWidth, 1);
}

__device__ void printDeviceMatrix(float *matrix, int width,int height,const char *name){
    printf("Device Matrix %s (%d x %d):\n", name, height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void MatrixMulKernel(float *M,float *N,float *P,int m,int n,int o){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if(row < m && col < o){
    float Pvalue = 0;
    for(int k = 0; k < n; ++k){
      Pvalue += M[static_cast<size_t>(row) * n + k] * N[static_cast<size_t>(k) * o + col];
    }
    P[static_cast<size_t>(row) * o + col] = Pvalue;
  }
}

__global__ void TiledMatrixMulKernel(float *M,float *N,float *P,int m,int n,int o,int tileWidth){
  extern __shared__ float sharedMem[];              // 这表示共享内存的大小不是在编译时固定的，而是在启动内核时动态分配的
  float *Mds = sharedMem;                           // Mds 指向共享内存的开始部分，用于存储 M 的 tile
  float *Nds = &sharedMem[tileWidth * tileWidth];   // Nds 指向共享内存的第二部分，用于存储 N 的 tile

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * tileWidth + ty;
  int Col = bx * tileWidth + tx;

  float Pvalue = 0;

  // 由于 GPU 的共享内存容量有限，无法一次性装下整个矩阵，算法将内维度 $n$ 切分成了多个大小为 tileWidth 的块（Tile）。
  // 循环变量 ph（Phase）表示当前正在处理第几个块。总块数为 (n + tileWidth - 1) / tileWidth（向上取整）。
  for(int ph = 0; ph < (n + tileWidth - 1) / tileWidth; ++ph){
    // 协作加载: 线程块内的每个线程负责将全局内存中的一个元素搬运到共享内存 Mds 中
    if(Row < m && (ph * tileWidth + tx) < n){
      Mds[ty * tileWidth + tx] = M[static_cast<size_t>(Row) * n + ph * tileWidth + tx];
    }
    else{
      Mds[ty * tileWidth + tx] = 0.0;
    }

    // 协作加载: 线程块内的每个线程负责将全局内存中的一个元素搬运到共享内存 Nds 中
    if(Col < o && (ph * tileWidth + ty) < n){
      Nds[ty * tileWidth + tx] = N[(static_cast<size_t>(ph) * tileWidth + ty) * o + Col];
    }
    else{
      Nds[ty * tileWidth + tx] = 0.0;
    }

    __syncthreads();
    // 计算矩阵乘法
    for(int k = 0; k < tileWidth; ++k){
      Pvalue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
    }
    __syncthreads();
  }

  if(Row < m && Col < o){
    P[static_cast<size_t>(Row) * o + Col] = Pvalue;
  }
}

void matrixMul(float *M,float *N,float *P,int m,int n,int o){
  float *d_M,*d_N,*d_P;
  size_t size_M = static_cast<size_t>(m) * n * sizeof(float);
  size_t size_N = static_cast<size_t>(n) * o * sizeof(float);
  size_t size_P = static_cast<size_t>(m) * o * sizeof(float);

  gpuErrchk(cudaMalloc((void**)&d_M,size_M));
  gpuErrchk(cudaMalloc((void**)&d_N,size_N));
  gpuErrchk(cudaMalloc((void**)&d_P,size_P));

  gpuErrchk(cudaMemcpy(d_M,M,size_M,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_N,N,size_N,cudaMemcpyHostToDevice));

  dim3 dimBlock(16,16);
  dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x,(m + dimBlock.y - 1) / dimBlock.y);

  MatrixMulKernel<<<dimGrid,dimBlock>>>(d_M,d_N,d_P,m,n,o);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(P,d_P,size_P,cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_M));
  gpuErrchk(cudaFree(d_N));
  gpuErrchk(cudaFree(d_P));
}

void matrixMulTiling(float *M,float *N,float *P,int m,int n,int o){
  float *d_M,*d_N,*d_P;

  int tileWidth = calculateOptimalTileWidth(m,n,o);

  size_t size_M = static_cast<size_t>(m) * n * sizeof(float);
  size_t size_N = static_cast<size_t>(n) * o * sizeof(float);
  size_t size_P = static_cast<size_t>(m) * o * sizeof(float);

  gpuErrchk(cudaMalloc((void**)&d_M,size_M));
  gpuErrchk(cudaMalloc((void**)&d_N,size_N));
  gpuErrchk(cudaMalloc((void**)&d_P,size_P));

  gpuErrchk(cudaMemcpy(d_M,M,size_M,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_N,N,size_N,cudaMemcpyHostToDevice));

  dim3 dimBlock(tileWidth,tileWidth);
  dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x,(m + dimBlock.y - 1) / dimBlock.y);

  size_t sharedMemSize = 2 * static_cast<size_t>(tileWidth) * tileWidth * sizeof(float);
  TiledMatrixMulKernel<<<dimGrid,dimBlock,sharedMemSize>>>(d_M,d_N,d_P,m,n,o,tileWidth);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(P,d_P,size_P,cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_M));
  gpuErrchk(cudaFree(d_N));
  gpuErrchk(cudaFree(d_P));
}


float benchmark(
  void (*func)(float*, float*, float*, int, int, int),
  float *M, float *N, float *P, int m, int n, int o, int warmup = 25, int iterations = 100) {
    
  if (iterations <= 0) {
    return 0.0f;
  }

  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  // Warm-up
  for (int i = 0; i < warmup; ++i) {
      func(M, N, P, m, n, o);
  }

  gpuErrchk(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
      func(M, N, P, m, n, o);
  }
  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaEventSynchronize(stop));

  float milliseconds = 0;
  gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

  gpuErrchk(cudaEventDestroy(start));
  gpuErrchk(cudaEventDestroy(stop));

  return milliseconds / iterations;
}

bool allclose(float *M,float *N,int m,int n,float rtol=1e-3f,float atol=1e-5f){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      float diff = std::fabs(M[static_cast<size_t>(i) * n + j] - N[static_cast<size_t>(i) * n + j]);
      float scale = std::fabs(N[static_cast<size_t>(i) * n + j]);
      if(diff > atol + rtol * scale){
        return false;
      }
    }
  }
  return true;
}

void printMatrix(float *matrix,int rows,int cols){
  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      std::cout << std::setw(6) << matrix[static_cast<size_t>(i) * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(){
  int m = 2010, n = 3200, o = 9111;

  size_t numM = static_cast<size_t>(m) * n;
  size_t numN = static_cast<size_t>(n) * o;
  size_t numP = static_cast<size_t>(m) * o;

  float *M = new float[numM];
  float *N = new float[numN];
  float *P1 = new float[numP];
  float *P2 = new float[numP];

  for(size_t i = 0; i < numM; ++i){
    M[i] = static_cast<float>(1);
  }

  for(size_t i = 0; i < numN; ++i){
    N[i] = static_cast<float>(1.5);
  }

  float avgTimeMatrixMulTiling = benchmark(matrixMulTiling, M, N, P1, m, n, o);
  float avgTimeMatrixMul = benchmark(matrixMul, M, N, P2, m, n, o);
  bool same = allclose(P1, P2, m, o);
  std::cout << "Average time for matrix multiplication with tiling: " << avgTimeMatrixMulTiling << " ms" << std::endl;
  std::cout << "Average time for matrix multiplication without tiling: " << avgTimeMatrixMul << " ms" << std::endl;
  std::cout << "Results are close: " << (same ? "Yes" : "No") << std::endl;

  if(true && !same){
    std::cout << "\n Matrix P1(from matrixMulTiling):" << std::endl;
    printMatrix(P1, m, o);

    std::cout << "\n Matrix P2(from matrixMul):" << std::endl;
    printMatrix(P2, m, o);
  }

  delete[] M;
  delete[] N;
  delete[] P1;
  delete[] P2;
  
  return 0;
}
