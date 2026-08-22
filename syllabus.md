## 核心学习计划：理论与实战结合

你当前打开的这五个开源项目构成了一个非常完整的 CUDA 学习生态系统。我为你设计了一个递进式的学习计划，将《PMPP》的理论与这些代码库完美融合。

### 阶段一：夯实底层基础（第1-4周）

以课本为主，通过简单内核验证理论。



- **理论学习**：精读《PMPP》第 2-6 章，掌握线程层次结构、内存模型与基础性能优化。
- **代码实操**：参考 [pmpp](https://github.com/tugot17/pmpp) 仓库中的课后习题完整解答，比对你的代码思路。
- **每日一练**：模仿 [GPU](https://github.com/a-hamdi/GPU) 仓库的 "100 days" 挑战前 10 天，手写向量加法、矩阵乘法等基础算子。

### 阶段二：进阶并行算法（第5-8周）

学习复杂数据访问模式与并行设计。



- **理论学习**：精读《PMPP》第 7-13 章，攻克卷积、规约（Reduction）、前缀和（Scan）与排序算法。
- **工业级参考**：查阅官方 [cuda-samples](https://github.com/NVIDIA/cuda-samples) 中 `2_Concepts_and_Techniques` 目录下的标准实现，学习 NVIDIA 的最佳实践。
- **题库刷题**：在 [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 仓库中完成 200 多个 CUDA Kernels 的 Easy 和 Medium 级别题目，巩固 Warp 级规约等技巧。

### 阶段三：AI 系统与前沿优化（第9-12周）

脱离课本，直击现代大模型底座技术。



- **高性能算子**：深入研究 [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) 仓库中的 `cuda-kernels` 和 `cutlass` 目录，学习高性能 GEMM 的手写与调优。
- **前沿技术**：挑战 [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 中的 Hard+ 级别（如 FlashAttention-2 的纯 MMA 实现），并学习使用 Triton 编写 AI 算子。
- **系统实战**：研读 [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) 的 `large-language-model` 目录，了解大模型推理优化的真实工业场景。