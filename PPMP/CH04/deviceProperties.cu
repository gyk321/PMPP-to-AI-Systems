# include <cuda_runtime.h>
#include <stdio.h>

int main(){
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  printf("Detected %d CUDA Capable device(s)\n", deviceCount);\

  for(int dev = 0;dev < deviceCount;dev++){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);

    printf("\nDevice %d: \"%s\"\n",dev,deviceProp.name);
    printf("  Major revision number:         %d\n",deviceProp.major);
    printf("  Minor revision number:         %d\n",deviceProp.minor);
    printf("  Total amount of global memory: %.2f GB\n",(float)deviceProp.totalGlobalMem/(pow(1024.0,3)));
    printf("  Total amount of constant memory: %zu bytes\n",deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block: %zu bytes\n",deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",deviceProp.regsPerBlock);
    printf("  Warp size:                      %d\n",deviceProp.warpSize);
    printf("  Maximum number of threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block: %d x %d x %d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid: %d x %d x %d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("  Clock rate:                     %.2f MHz\n",deviceProp.clockRate * 1e-3f);
    printf("  Memoery Clock rate:                %.2f MHz\n",deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:               %d-bit\n",deviceProp.memoryBusWidth);
    printf("  L2 Cache Size:                  %d bytes\n",deviceProp.l2CacheSize);
  }
  return 0;
}