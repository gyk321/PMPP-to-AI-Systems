#include<cuda_runtime.h>
#include<stdio.h>

// 在 CPU（主机）上执行两个浮点数向量的逐元素相乘
void vecMulHost(float *A_h, float *B_h, float *C_h, int N){
  for(int i = 0;i < N;i++){
    C_h[i] = A_h[i] * B_h[i];
  }

}

// 在 GPU 上执行的核函数，计算两个向量的逐元素相乘，但不同之处在于它是高度并行执行的
// __global__: 这是 CUDA 的执行空间限定符。它表示这个函数是一个“核函数”
//             1.在 GPU（Device）上执行。2. 由 CPU（Host）调用
__global__ void vecMulKernel(float *A, float *B, float *C, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N){
    C[i] = A[i] * B[i];
  }
}

// 在 GPU 上进行计算，必须经历一套标准的“分配 - 拷贝 - 计算 - 拷贝回 - 释放”流程。
void vecMulDevice(float *A_h, float *B_h, float *C_h, int N){

  // 计算内存大小并声明设备指针
  int size = N * sizeof(float);
  float *A_d, *B_d, *C_d;

  // 在 GPU 上分配内存 (Allocate)
  cudaMalloc((void**)&A_d, size);
  cudaMalloc((void**)&B_d, size);
  cudaMalloc((void**)&C_d, size);

  // 将数据从 CPU 传输到 GPU：从 Host（CPU）流向 Device（GPU）
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
  
  // 启动 GPU 核函数进行计算
  vecMulKernel<<<ceil(N / 256), 256>>>(A_d, B_d, C_d, N);
  
  // 将计算结果从 GPU 传回 CPU
  cudaMemcpy(C_d, C_h, size, cudaMemcpyDeviceToHost);

  // 释放 GPU 内存
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main(int argc, char const *argv[])
{
  // 准备数据与分配 CPU 内存
  int N = 1<<24;
  float *A, *B, *C;
  A = (float*)malloc(N * sizeof(float));
  B = (float*)malloc(N * sizeof(float));
  C = (float*)malloc(N * sizeof(float));

  // 初始化输入数据
  for (int i = 0; i < N; i++)
  {
    /* code */
    A[i] = i;
    B[i] = i;
  }

  // 执行 CPU 版本并计时
  clock_t start_host = clock();
  vecMulHost(A,B,C,N);
  clock_t end_host = clock();
  double time_host = (double)(end_host - start_host) / CLOCKS_PER_SEC;
  printf("Host time: %f seconds\n", time_host);

  // 执行 GPU 版本并计时
  clock_t start_device = clock();
  vecMulDevice(A,B,C,N);
  clock_t end_device = clock();
  double time_device = (double)(end_device - start_device) / CLOCKS_PER_SEC;
  printf("Device time: %f seconds\n", time_device);
  
  // 清理 CPU 内存
  free(A);
  free(B);
  free(C);

  return 0;
}
