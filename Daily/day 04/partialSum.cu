#include<stdio.h>

// 利用 共享内存（Shared Memory） 实现一个块内并行前缀和
__global__ void partialSumKernel(int *input,int *output,int n){
  extern __shared__ int sharedMemory[];   // 声明了一块动态共享内存

  int tid = threadIdx.x;                            // 当前线程在其所在的线程块（Block）内的局部索引
  int index = blockIdx.x * blockDim.x * 2 + tid;    // 整个网格（Grid）中的全局数据索引,每个线程块实际上负责处理 2 * blockDim.x 个全局内存元素

  if(index < n){
    sharedMemory[tid] = input[index] + input[index + blockDim.x];
    __syncthreads();

    // Fixed: start stride at 1 and shift left each iteration.
    // 它的逻辑是让每个元素去加上它前面的元素，步长不断翻倍（1, 2, 4, 8...）.stride <<= 1 也就是乘 2
    for (int stride = 1; stride < blockDim.x; stride <<= 1){
      int temp = 0;
      if(tid >= stride){
        temp = sharedMemory[tid - stride];
      }
      __syncthreads();
      sharedMemory[tid] += temp;
      __syncthreads();
    }
    
    output[index] = sharedMemory[tid];
  }
}

int main(){
  const int N = 16;
  const int blockSize = 8;

  int h_input[N] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  int h_output[N] = {0}; // zero-init so unwritten entries are visible

  int *d_input, *d_output;
  size_t size = N * sizeof(int);

  cudaMalloc(&d_input,size);
  cudaMalloc(&d_output,size);

  cudaMemcpy(d_input,h_input,size,cudaMemcpyHostToDevice);

  partialSumKernel<<<N / blockSize,blockSize,blockSize * sizeof(int)>>>(d_input,d_output,N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_output,d_output,size,cudaMemcpyDeviceToHost);

  printf("Input: ");

  for (int i = 0; i < N; i++){
    printf("%d ",h_input[i]);
  }

  printf("\nOutput: ");
  for (int i = 0; i < N; i++){
    printf("%d ",h_output[i]);
  }

  printf("\n");
  cudaFree(d_input);
  cudaFree(d_output);
  
  return 0;
}
