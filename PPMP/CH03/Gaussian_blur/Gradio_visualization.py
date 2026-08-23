from pathlib import Path

import gradio as gr
import numpy as np
import torch
from Gaussian_blur import compile_extension

ext = compile_extension()

def blur(image:np.ndarray,blur_size:int):
  # 前处理：NumPy (HWC) -> PyTorch (CHW) 并传入 GPU
  img = torch.tensor(image).to("cuda").permute(2,0,1).contiguous()
  blur_img = ext.gaussian_blur(img,blur_size)

  # 将结果 .cpu() 搬回内存，
  # 再通过 .permute(1,2,0) 将 CHW 转换回 Gradio 认识的 HWC 格式，
  # 最后 .numpy() 转回 NumPy 数组返回给前端
  return blur_img.cpu().permute(1,2,0).numpy()

image_path = Path(__file__).parent.parent / "Grace_Hopper.jpg"

demo = gr.Interface(
  fn = blur,      # 绑定上面定义好的处理函数
  inputs=["image",gr.Slider(minimum=0,maximum=30,step=1,value=3)],
  outputs=["image"],
  examples=[[str(image_path),3]]
)

if __name__ == "__main__":
  demo.launch()