#include <cuda_runtime.h>
#include <iostream>

#define WIDTH 1024
#define HEIGHT 1024

// 在 GPU 上对一个二维矩阵进行转置
__global__ void transposeMatrix(const float *input, float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程负责处理的全局列索引
  int y = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程负责处理的全局行索引

  if (x < width && y < height) {
    int inputIndex = y * width + x;           // 输入矩阵的索引
    int outputIndex = x * height + y;         // 输出矩阵的索引
    output[outputIndex] = input[inputIndex];  // 转置操作
  }
}

// CUDA API 调用和核函数执行通常是异步的，且不会直接抛出 C++ 异常。这个函数通过调用 cudaGetLastError() 获取最近一次 CUDA 调用的状态码。
void checkCudaError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(){
  int width = WIDTH;
  int height = HEIGHT;

  // 在主存（内存）中分配空间，并准备好要被转置的原始数据
  size_t size = width * height * sizeof(float);
  float* h_input = (float*)malloc(size);
  float* h_output = (float*)malloc(size);

  for (int i = 0; i < width * height; ++i) {
    h_input[i] = static_cast<float>(i);
  }

  // 利用 cudaMalloc 在显卡上分配对应的空间.然后用 cudaMemcpy 将初始化好的数据通过 PCIe 总线复制到显卡中
  float* d_input;
  float* d_output;
  cudaMalloc((void**)&d_input, size);
  cudaMalloc((void**)&d_output, size);
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  checkCudaError("Memory copy from host to device failed");

  dim3 blockSize(32, 32);   // 定义每个线程块 (Block) 包含 32 × 32 = 1024 个线程
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);        // 计算需要多少个 Block 才能覆盖整个矩阵

  transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
  cudaDeviceSynchronize();
  checkCudaError("Kernel launch failed");

  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
  checkCudaError("Memory copy from device to host failed");

  bool success = true;
  for (int i = 0; i < width; ++i) {
    for(int j = 0; j < height; ++j) {
      if (h_output[i * height + j] != h_input[j * width + i]) {
        success = false;
        std::cerr << "Mismatch at (" << i << ", " << j << "): "
                  << h_output[i * height + j] << " != " << h_input[j * width + i] << std::endl;
        break;
      }
    }
  }

  if (success) {
    std::cout << "Matrix transpose successful!" << std::endl;
  } else {
    std::cerr << "Matrix transpose failed!" << std::endl;
  }

  cudaFree(d_input);
  cudaFree(d_output);

  free(h_input);
  free(h_output);

  return 0;
}