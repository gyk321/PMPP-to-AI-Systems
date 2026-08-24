#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>
#include<cuda_runtime.h>

// 灰度图（单通道）：像素是一个接一个紧挨着的。
// - [Gray0, Gray1, Gray2, Gray3, ...]

// RGB图（三通道交错存储）：每个像素包含 3 个字节，按 R、G、B 的顺序交错排列。
// - [R0, G0, B0, R1, G1, B1, R2, G2, B2, ...]
__global__ void rgbToGrayscaleKernel(const unsigned char* rgbImage, unsigned char* grayImage, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  const int CHANNELS = 3; // RGB image has 3 channels

  // 确保当前线程处理的坐标没有超出图像的实际宽度和高度（防止内存越界）
  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;

    unsigned char r = rgbImage[rgbOffset];
    unsigned char g = rgbImage[rgbOffset + 1];
    unsigned char b = rgbImage[rgbOffset + 2];

    // 使用经典的心理声学灰度加权公式进行转换
    grayImage[grayOffset] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor rgb_to_gray(torch::Tensor rgbImage) {
  assert(rgbImage.device().type() == torch::kCUDA);
  assert(rgbImage.dtype() == torch::kUInt8);
  assert(rgbImage.size(2) == 3); // Ensure the input image has 3 channels (RGB)

  // 获取维度：假设输入图像的形状为 [height, width, channels]，提取出高和宽。
  int height = rgbImage.size(0);
  int width = rgbImage.size(1);

  auto grayImage = torch::empty({height, width}, torch::TensorOptions().dtype(torch::kUInt8).device(rgbImage.device()));

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(cdiv(width, threadsPerBlock.x), cdiv(height, threadsPerBlock.y));

  rgbToGrayscaleKernel<<<numBlocks, threadsPerBlock>>>(rgbImage.data_ptr<unsigned char>(), grayImage.data_ptr<unsigned char>(), width, height);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grayImage;
}