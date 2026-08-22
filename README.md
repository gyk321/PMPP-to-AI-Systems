# PMPP to AI Systems

> 以《Programming Massively Parallel Processors》(PMPP) 为主线，从 CUDA 基础算子手写逐步进阶到 AI 系统高性能算子开发的 12 周学习仓库。

本仓库记录从 GPU 并行编程基础（PMPP 理论）到现代 AI 系统（FlashAttention、GEMM 优化等）的完整进阶路径：**每日一练** 的 CUDA 手写算子、**课本配套代码** 的实践复现，以及配套的 **12 周三阶段学习计划**。

---

## 仓库结构

```
PMPP-to-AI-Systems/
├── Daily/                    # 每日一练：手写基础算子（参考 GPU 100 days 挑战）
│   ├── day 01/vectAdd.cu           # 向量加法：grid/block/thread 层次、设备内存管理
│   ├── day 02/MatrixAdd.cu         # 矩阵加法：2D 线程索引映射
│   ├── day 03/Matrix_vec_mult.cu   # 矩阵 × 向量：shared memory 优化
│   └── day 04/partialSum.cu        # 部分和规约：动态共享内存 + stride 循环
├── PPMP/CH02/                # 《PMPP》第 2 章配套代码：向量逐元素乘法
│   ├── vecMul.cu                    # CUDA 核函数实现
│   ├── vecMulTorchTensor.cu        # 自定义 CUDA 核函数扩展 PyTorch Tensor
│   ├── vecMul.py                   # 纯 Python 循环 vs CUDA 扩展的性能对照
│   ├── Makefile                    # Linux 一键编译运行（nvcc）
├── README.md                 # 本文件：仓库说明 + 学习计划
└── README_offcial.md         # 参考资料：GPU 100 days 挑战学习日志（来源见下）
```

---

## 学习计划：理论与实战结合（12 周 · 三阶段）

以《PMPP》为理论主线，配合五个开源项目构建完整的 CUDA 学习生态，递进式将理论融入代码实践。

### 阶段一：夯实底层基础（第 1–4 周）

以课本为主，通过简单内核验证理论。

- **理论学习**：精读《PMPP》第 2–6 章，掌握线程层次结构、内存模型与基础性能优化。
- **代码实操**：参考 [pmpp](https://github.com/tugot17/pmpp) 仓库中的课后习题完整解答，比对你的代码思路。
- **每日一练**：模仿 [GPU](https://github.com/a-hamdi/GPU) 仓库的 "100 days" 挑战前 10 天，手写向量加法、矩阵乘法等基础算子。

### 阶段二：进阶并行算法（第 5–8 周）

学习复杂数据访问模式与并行设计。

- **理论学习**：精读《PMPP》第 7–13 章，攻克卷积、规约（Reduction）、前缀和（Scan）与排序算法。
- **工业级参考**：查阅官方 [cuda-samples](https://github.com/NVIDIA/cuda-samples) 中 `2_Concepts_and_Techniques` 目录下的标准实现，学习 NVIDIA 的最佳实践。
- **题库刷题**：在 [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 仓库中完成 200+ CUDA Kernels 的 Easy 和 Medium 级别题目，巩固 Warp 级规约等技巧。

### 阶段三：AI 系统与前沿优化（第 9–12 周）

脱离课本，直击现代大模型底座技术。

- **高性能算子**：深入研究 [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) 仓库中的 `cuda-kernels` 和 `cutlass` 目录，学习高性能 GEMM 的手写与调优。
- **前沿技术**：挑战 [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 中的 Hard+ 级别题目（如 FlashAttention-2 的纯 MMA 实现），并学习使用 Triton 编写 AI 算子。
- **系统实战**：研读 [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) 的 `large-language-model` 目录，了解大模型推理优化的真实工业场景。

---

## 运行方式

### CUDA 程序（Linux）

```bash
# 每日一练
nvcc -o vectAdd "Daily/day 01/vectAdd.cu" && ./vectAdd

# CH02 配套代码（或直接使用 Makefile）
cd PPMP/CH02 && make
```

### Python 性能对照（vecMul.py）

`vecMul.py` 对比纯 Python 循环与 CUDA 扩展的性能，需 CUDA 版 PyTorch：

```bash
cd PPMP/CH02
uv run python vecMul.py
```

---

## 参考资料

| 仓库 | 用途 |
|------|------|
| [tugot17/pmpp](https://github.com/tugot17/pmpp) | 《PMPP》课后习题完整解答 |
| [a-hamdi/GPU](https://github.com/a-hamdi/GPU) | GPU 100 days 挑战（学习日志见 `README_offcial.md`） |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | 官方 CUDA 示例与最佳实践 |
| [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) | 200+ CUDA Kernel 练习题 |
| [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | 高性能算子 / GEMM / 大模型推理优化实战 |
| [hkproj](https://github.com/hkproj/) | 学习路线指导 |
