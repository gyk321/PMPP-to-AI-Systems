from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

def compile_extension():
  cuda_source = (
    Path(__file__).parent / "matrix_vector_multiplication.cu"
  ).read_text(encoding="utf-8")

  # 声明了 C++ 侧的函数接口
  cpp_source = (
    "torch::Tensor matrix_vector_multiplication(torch::Tensor B,torch::Tensor C);"
  )

  return load_inline(
    name = "matrixMul_extension",
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ["matrix_vector_multiplication"],
    with_cuda = True,
  )

def main():
  ext = compile_extension()

  DEVICE,DTYPE = "cuda",torch.float32

  B = torch.randn(1000,256).to(DEVICE,DTYPE)
  C = torch.randn(256).to(DEVICE,DTYPE)

  # 在 GPU 上调用这个 CUDA kernel 完成 result = B @ C
  res = ext.matrix_vector_multiplication(B,C)

  # 和 PyTorch 自带的 torch.matmul(B, C) 对比结果是否正确
  torch_res = torch.matmul(B,C)

  print(res.shape,torch_res.shape)

  # 检查自定义 CUDA 结果和 PyTorch 结果是否近似相等
  print(torch.allclose(res,torch_res,rtol = 1e-3,atol = 1e-3))
  print()
  print(res[:4])
  print(torch_res[:4])

if __name__ == "__main__":
  main()