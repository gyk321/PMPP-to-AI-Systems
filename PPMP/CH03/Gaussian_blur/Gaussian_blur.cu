#include<c10/cuda/CUDAStream.h>
#include<c10/cuda/CUDAException.h>
#include<cuda_runtime.h>

// 核函数:接收输入图像指针、输出图像指针、图像宽高以及模糊半径
__global__ void blur_kernel(unsigned char* Pin,unsigned char* Pout,int width,int height,int blur_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int channel = threadIdx.z; // Assuming 3 channels (RGB)

  // 计算出当前通道在内存中的起始位置。
  // 例如，如果是绿色通道 (channel=1)，它需要跳过整个红色通道的数据 (width * height 个像素)
  int baseOffset = channel * width * height; // Offset for the current channel

  // 边界保护:因为启动的线程网格尺寸通常是线程块的整数倍，可能会超出实际图像的尺寸
  if (col < width && row < height) {
    int grayOffset = row * width + col;   // 当前中心像素

    int pixelValues = 0;
    int pixels = 0;

    // 滑动窗口：通过两个嵌套的 for 循环，遍历以当前像素为中心，半径为 blur_size 的方形区域内的所有像素值，并累加它们的值。
    for (int blurRow = -blur_size; blurRow <= blur_size; ++blurRow) {
      for (int blurCol = -blur_size; blurCol <= blur_size; ++blurCol) {
        int curCol = col + blurCol;
        int curRow = row + blurRow;

        // 局部边界检查:当窗口延伸到图像外部时（例如 curRow < 0），忽略这些无效像素。
        if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
          // 精确地计算出了当前窗口正在扫描的邻近像素在物理内存中的绝对位置，取出该灰度值后，累加到 pixelValues 变量中
          pixelValues += Pin[baseOffset + curRow * width + curCol];
          ++pixels;
        }
      }
    }
    // 当前窗口内所有有效像素的算术平均值。这就是均值模糊（Box Blur）的本质所在——当前点的新颜色，等于周围邻居颜色的平均分
    Pout[baseOffset + grayOffset] = static_cast<unsigned char>(pixelValues / pixels);
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor gaussian_blur(torch::Tensor input, int blur_size) {
  // 数据校验 Ensure input tensor is on CUDA device and of type uint8
  assert(input.device().type() == torch::kCUDA);
  assert(input.dtype() == torch::kByte);
  assert(input.dim() == 3); // Ensure input is a 3D tensor (C, H, W)

  // 提取维度与分配显存
  int channels = input.size(0);
  int height = input.size(1);
  int width = input.size(2);

  // Create output tensor
  auto output = torch::empty_like(input);

  // 配置线程执行架构
  dim3 threadsPerBlock(16, 16, channels);
  dim3 numBlocks(cdiv(width, threadsPerBlock.x), cdiv(height, threadsPerBlock.y));

  // Launch the Gaussian blur kernel
  blur_kernel<<<numBlocks, threadsPerBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
    input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), width, height, blur_size
  );

  // Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return output;
}