#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void LayerNormKernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;    // 当前线程负责处理的全局行索引
    int col = threadIdx.y;        // 当前线程在列方向上的起始列索引

    extern __shared__ float shared_data[];
    float* row_data = shared_data;

    // 将当前线程块负责的多行数据从全局内存读入共享内存，减少后续计算对全局内存的访问次数。
    if (row < rows) {
        for (int c = col; c < cols; c += blockDim.y) {
            row_data[threadIdx.x * cols + c] = input[row * cols + c];
        }
    }

    // 保证线程块内所有线程都到达此点后才能继续执行后续代码
    __syncthreads();

    // 边界检查：再次检查 row < rows，确保只对有效行进行计算
    if (row < rows) {
        //均值计算
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) {
            mean += row_data[threadIdx.x * cols + c];
        }
        mean /= cols;

        // 方差计算
        float variance = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = row_data[threadIdx.x * cols + c] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        // 添加小常数 1e-7f 防止方差为零时除以零，提高数值稳定性
        float stddev = sqrtf(variance + 1e-7f);

        // 将归一化后的结果写回输出矩阵
        for (int c = col; c < cols; c += blockDim.y) {
            output[row * cols + c] = (row_data[threadIdx.x * cols + c] - mean) / stddev;
        }
    }
}

int main() {
    const int rows = 1024;
    const int cols = 512;

    // 1. 查询设备属性.获取当前 GPU 的 每个线程块可用的最大共享内存字节数。
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

    // 2. 确定 blockSize.x，使共享内存不超过限制,如果超过设备限制，则不断将 block_x 减半，直到满足限制。
    int block_x = 32;
    size_t needed_shared = block_x * cols * sizeof(float);
    while (needed_shared > prop.sharedMemPerBlock && block_x > 1) {
        block_x /= 2;
        needed_shared = block_x * cols * sizeof(float);
    }
    if (needed_shared > prop.sharedMemPerBlock) {
        printf("Error: even block_x=1 requires %zu bytes > %zu bytes limit.\n", needed_shared, prop.sharedMemPerBlock);
        return -1;
    }

    int block_y = 32;   // 表示每一行有 32 个线程并行处理列数据
    dim3 blockSize(block_x, block_y);                       // block_x 行 × block_y 列的二维线程组
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x);  // 在 x 方向排列 ceil(rows / block_x) 个线程块
    size_t sharedMemSize = needed_shared;
    printf("Using blockDim.x=%d, shared memory=%zu bytes\n", block_x, sharedMemSize);

    // 3. 分配主机
    float *input = (float*)malloc(rows * cols * sizeof(float));
    float *output = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            input[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // 设备内存分配
    float *d_a, *d_b;   // 声明两个指向设备内存的指针
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, rows * cols * sizeof(float));
    if (err != cudaSuccess) { printf("cudaMalloc d_a failed: %s\n", cudaGetErrorString(err)); return -1; }

    err = cudaMalloc((void**)&d_b, rows * cols * sizeof(float));
    if (err != cudaSuccess) { printf("cudaMalloc d_b failed: %s\n", cudaGetErrorString(err)); return -1; }

    err = cudaMemcpy(d_a, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); return -1; }

    // 4. 启动内核
    // 如果内核中使用了 extern __shared__ 声明动态共享内存，则需要在这里指定其大小。
    LayerNormKernel<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_b, rows, cols);

    // 检查启动时的错误
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("Kernel launch failed: %s\n", cudaGetErrorString(err)); return -1; }

    // 阻塞主机，直到设备上所有先前提交的工作（包括内核执行）全部完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); return -1; }

    // 5. 拷贝结果并验证
    err = cudaMemcpy(output, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err)); return -1; }

    // 只打印输出矩阵的前 3 行、每行前 5 个元素，用于快速验证结果是否合理
    printf("First few output values:\n");
    for (int i = 0; i < 3 && i < rows; ++i) {
        for (int j = 0; j < 5 && j < cols; ++j) {
            printf("%.2f ", output[i * cols + j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(input);
    free(output);
    return 0;
}