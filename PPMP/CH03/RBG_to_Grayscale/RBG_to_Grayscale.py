from pathlib import Path

from torch.utils.cpp_extension import load_inline
from torchvision.io import read_image,write_png

def compile_extension():
  cuda_source = (
    Path(__file__).parent / "RBG_to_Grayscale.cu").read_text(encoding="utf-8")
  cpp_source = ("torch::Tensor rgb_to_gray(torch::Tensor rgbImage);")

  return load_inline(
    name="rgb_to_gray",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["rgb_to_gray"],
    verbose=True,
  )

def main():
  current_dir = Path(__file__).parent

  img = read_image(current_dir.parent / "Grace_Hopper.jpg").permute(1, 2, 0).cuda()  # HWC
  print("Original:")
  print("mean:", img.float().mean())
  print("Input image:", img.shape, img.dtype)
  print()

  ext = compile_extension()
  gray_img = ext.rgb_to_gray(img)

  print("Converted:")
  print("mean:", gray_img.float().mean())
  print("Output image:", gray_img.shape, gray_img.dtype)

  save_path = current_dir / "output.png"
  print(f"Converted to grayscale, saving to {save_path}")
  write_png(gray_img.unsqueeze(0).cpu(), save_path)

if __name__ == "__main__":
  main()