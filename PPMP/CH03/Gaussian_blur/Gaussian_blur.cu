#include<c10/cuda/CUDAStream.h>
#include<c10/cuda/CUDAException.h>
#include<cuda_runtime.h>

__global__ void blur_kernel(unsigned char* Pin,unsigned char* Pout,int width,int height,int blur_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int channel = threadIdx.z; // Assuming 3 channels (RGB)
  int baseOffset = channel * width * height; // Offset for the current channel

  if (col < width && row < height) {
    int grayOffset = row * width + col;

    int pixelValues = 0;
    int pixels = 0;

    for (int blurRow = -blur_size; blurRow <= blur_size; ++blurRow) {
      for (int blurCol = -blur_size; blurCol <= blur_size; ++blurCol) {
        int curCol = col + blurCol;
        int curRow = row + blurRow;

        if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
          pixelValues += Pin[baseOffset + curRow * width + curCol];
          ++pixels;
        }
      }
    }
    Pout[baseOffset + grayOffset] = static_cast<unsigned char>(pixelValues / pixels);
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor gaussian_blur(torch::Tensor input, int blur_size) {
  // Ensure input tensor is on CUDA device and of type uint8
  assert(input.device().type() == torch::kCUDA);
  assert(input.dtype() == torch::kByte);
  assert(input.dim() == 3); // Ensure input is a 3D tensor (C, H, W)

  int channels = input.size(0);
  int height = input.size(1);
  int width = input.size(2);

  // Create output tensor
  auto output = torch::empty_like(input);

  // Define CUDA kernel launch parameters
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