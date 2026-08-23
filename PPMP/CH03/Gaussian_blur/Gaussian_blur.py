from pathlib import Path

from torch.utils.cpp_extension import load_inline
from torchvision.io import read_image,write_png

def compile_extension():
  cuda_source = (
    Path(__file__).parent / "Gaussian_blur.cu").read_text(encoding="utf-8")
  cpp_source = "torch::Tensor gaussian_blur(torch::Tensor input, int blur_size);"

  return load_inline(
    name = "Gaussian_blur_extension",
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ["gaussian_blur"],
    with_cuda = True,
  )

def main():
  current_dir = Path(__file__).parent

  # 使用 torchvision.io.read_image 读取经典的测试图 Grace_Hopper.jpg。
  # 读进来的 img 默认是一个形状为 (C, H, W)、数据类型为 uint8 的 PyTorch 张量。
  img = read_image(current_dir.parent / "Grace_Hopper.jpg").contiguous().cuda()

  print("Original:")
  print("mean:",img.float().mean())
  print("Input image:",img.shape,img.dtype)
  print()

  ext = compile_extension()
  blur_size = 3

  blur_img = ext.gaussian_blur(img,blur_size)
  print("Converted:")
  print("mean:",blur_img.float().mean())
  print("Input image:",blur_img.shape,blur_img.dtype)

  save_path = current_dir / "output.png"
  print(f"Blurred the image. It is saved at {save_path}")
  write_png(blur_img.cpu(),save_path)

if __name__ == "__main__":
  main()
