// mini_fa.cu — 这段代码实现了一个教学版的 FlashAttention，其采用了纯 fp32 数据格式且未使用 Tensor Core。
//
// 设计（对应文中"仓库"比喻）：
//   - HBM 全局内存 = 中央仓库：Q/K/V 都在这里，又大又慢
//   - 共享内存     = 车间工作台：sK/sV 是分批搬进来的"货箱"
//   - 寄存器       = 工人口袋：q、累加和 O、账本 m/l 都放在口袋里
//   - 一个 block 一次处理 Br = WP*ROWS = 64 个 query 行，逐批吃掉全部 K/V 块，
//     中间矩阵 S/P 从不写回中央仓库 —— 这就是 FlashAttention 的核心思想。
//
// 编译：nvcc -O3 -arch=native mini_fa.cu -o mini_fa
// 运行：./mini_fa   （N 必须能被 64 整除；D 固定为 64；支持 causal 掩码）

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                              \
    do {                                                                           \
        cudaError_t _e = (x);                                                      \
        if (_e != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                                  \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

// ---------- 几何参数 ----------
constexpr int BN   = 32;   // K/V 块宽：一个 warp 一次吃 32 个 key
constexpr int WP   = 32;   // 每 block 的 warp 数（即 1024 线程/block）
constexpr int ROWS = 2;    // 每个 warp 负责 2 个 query 行
constexpr int BR   = WP * ROWS;   // 每 block 覆盖 64 个 query 行
constexpr int NT   = WP * 32;     // 每 block 线程数 = 1024
constexpr int D    = 64;          // 头维度（教学固定值）

// 完整 warp 参与 shuffle 的掩码
#define FULL 0xffffffffu
// 负无穷（0xff800000 本身就是 -inf 的位模式，不要再加负号）
#define NEG_INF (__int_as_float(0xff800000))

// 教学版 FlashAttention 前向。
// 输入 Q/K/V 布局均为 [B*H, N, D]（row-major），输出 O 同布局。
// 注意：真实实现会在点积后乘 1/sqrt(D)；这里为保持 CPU 对照简单，省略了缩放。
__global__ void __launch_bounds__(NT) mini_flash_attn_kernel(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        int N, bool causal)
{
    const int warp = threadIdx.x >> 5;              // 本线程属于 block 内第几个 warp
    const int lane = threadIdx.x & 31;              // 本线程在 warp 内的编号 0..31
    const int row_base = blockIdx.x * BR + warp * ROWS;  // 本 warp 负责的第 1 个 query 行
    const int bh = blockIdx.y;                      // batch*head 编号

    // 工作台上只放两件货：K 货箱和 V 货箱。
    // 每行多 1 格 padding（D+1 列）：
    // 读取 sK[j][d] 时 32 个 lane 的地址差 D+1=65，
    // 65 mod 32 = 1，恰好错开 32 个"抽屉"（bank），避免 bank conflict。
    extern __shared__ float smem[];
    float* sK = smem;                  // [BN][D+1]
    float* sV = sK + BN * (D + 1);     // [BN][D+1]

    // 把本 warp 的 2 行 q 装进口袋：
    // lane 负责 d = lane 和 d = lane+32 两个位置，读全局内存时天然合并访问。
    float q[ROWS][2];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            q[r][h] = Q[((size_t)bh * N + row_base + r) * D + lane + h * 32];

    // 每行的三件随身物品：运行最大值 m、运行和 l、输出累加 O（2 个 d 位置）
    float m[ROWS], l[ROWS], o[ROWS][2];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        m[r] = NEG_INF;      // 账本第一页：最大值
        l[r] = 0.0f;           // 账本第二页：exp 之和
        o[r][0] = 0.0f;        // 输出累加，保存在寄存器里直到最后一刻
        o[r][1] = 0.0f;
    }

    const float* k_base = K + (size_t)bh * N * D;
    const float* v_base = V + (size_t)bh * N * D;

    // 外层循环：按列遍历所有 K/V 块（货箱按批搬进车间）
    for (int j0 = 0; j0 < N; j0 += BN) {
        // causal：若整块都在"未来"，全 block 统一跳过，保证 __syncthreads 步调一致。
        // 块内部分遮蔽的情况交给后面的逐元素掩码处理。
        if (causal && j0 > blockIdx.x * BR + BR - 1) continue;

        // 全体 1024 个线程合作，把 K/V 货箱从中央仓库搬进工作台（合并访问）
        for (int i = threadIdx.x; i < BN * D; i += NT) {
            int j = i / D, d = i - j * D;          // j: key 编号，d: 头内维度
            int key = j0 + j;
            float kv = (key < N) ? k_base[(size_t)key * D + d] : 0.0f;
            sK[j * (D + 1) + d] = kv;
            kv       = (key < N) ? v_base[(size_t)key * D + d] : 0.0f;
            sV[j * (D + 1) + d] = kv;
        }
        __syncthreads();   // 货箱全部就位后才能开工

        // ---- 第一步：算分数 S = q·K^T，每个 lane 负责第 lane 个 key ----
        float s[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) s[r] = 0.0f;
        for (int d = 0; d < D; ++d) {
            int owner = d & 31;    // 持有 q[d] 的 lane
            int half  = d >> 5;    // q[d] 在口袋的哪一格
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                // 从持有者口袋里广播 q[d]；sK 用 padding 后的行地址，抽屉不打架
                float qd = __shfl_sync(FULL, q[r][half], owner);
                s[r] += qd * sK[lane * (D + 1) + d];
            }
        }

        // ---- 第二步：causal 掩码，把"未来"位置的分数量为 -inf ----
        const int key = j0 + lane;
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
            if (causal && key > row_base + r) s[r] = NEG_INF;

        // ---- 第三步：在线 softmax + 就地累加 O（账本更新）----
        float pv[ROWS];   // 本块每个 key 的未归一化概率 p = exp(s - m_new)
        float mblk[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            // warp 内蝶形归约求本块最大值
            mblk[r] = s[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mblk[r] = fmaxf(mblk[r], __shfl_xor_sync(FULL, mblk[r], off));

            // 该行在本块全部被掩码遮住 → 账本不动（避免 -inf 和 -inf 相减产生 NaN）
            if (mblk[r] > NEG_INF) {
                float m_new = fmaxf(m[r], mblk[r]);
                float p = expf(s[r] - m_new);      // 被掩码的 lane 自然得到 0
                pv[r] = p;
                float lsum = p;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    lsum += __shfl_xor_sync(FULL, lsum, off);

                // 关键两行：旧账本按比例"缩水"，再并入新块 —— 这就是在线 softmax
                float scale = expf(m[r] - m_new);
                l[r] = l[r] * scale + lsum;
                o[r][0] *= scale;
                o[r][1] *= scale;
                m[r] = m_new;
            } else {
                pv[r] = 0.0f;
            }
        }

        // ---- 第四步：O += p · V。把每个 lane 的 p 广播给全 warp ----
        for (int j = 0; j < BN; ++j) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                float pj = __shfl_sync(FULL, pv[r], j);   // 同一 key 的 p 对全组可见
                o[r][0] += pj * sV[j * (D + 1) + lane];       // 自己口袋里的两个 d
                o[r][1] += pj * sV[j * (D + 1) + lane + 32];
            }
        }
        __syncthreads();   // 所有人用完这批货，才能搬下一批
    }

    // ---- 收尾：除以 l 归一化，O 只写回中央仓库这一次 ----
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float inv = (l[r] > 0.0f) ? 1.0f / l[r] : 0.0f;
        float* og = O + ((size_t)bh * N + row_base + r) * D;
        og[lane]      = o[r][0] * inv;
        og[lane + 32] = o[r][1] * inv;
    }
}

// CPU 对照实现：同样不缩放，逐行两遍求 max/sum + 一遍输出（纯验证用）
void cpu_attention(const float* q, const float* k, const float* v, float* o,
                   int BH, int N, int D_, bool causal)
{
    for (int bh = 0; bh < BH; ++bh) {
        const float* qb = q + (size_t)bh * N * D_;
        const float* kb = k + (size_t)bh * N * D_;
        const float* vb = v + (size_t)bh * N * D_;
        float* ob = o + (size_t)bh * N * D_;
        for (int i = 0; i < N; ++i) {
            const float* qi = qb + (size_t)i * D_;
            float m = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < N; ++j) {
                if (causal && j > i) continue;
                float s = 0.0f;
                for (int d = 0; d < D_; ++d) s += qi[d] * kb[(size_t)j * D_ + d];
                m = fmaxf(m, s);
            }
            float l = 0.0f;
            for (int j = 0; j < N; ++j) {
                if (causal && j > i) continue;
                float s = 0.0f;
                for (int d = 0; d < D_; ++d) s += qi[d] * kb[(size_t)j * D_ + d];
                l += expf(s - m);
            }
            float inv = 1.0f / l;
            for (int d = 0; d < D_; ++d) ob[i * D_ + d] = 0.0f;
            for (int j = 0; j < N; ++j) {
                if (causal && j > i) continue;
                float s = 0.0f;
                for (int d = 0; d < D_; ++d) s += qi[d] * kb[(size_t)j * D_ + d];
                float p = expf(s - m) * inv;
                for (int d = 0; d < D_; ++d)
                    ob[i * D_ + d] += p * vb[(size_t)j * D_ + d];
            }
        }
    }
}

int main()
{
    const int B = 2, H = 4, N = 512;        // B*H=8 个头，seq=512
    const int BH = B * H;                   // 总共需要处理的注意力头数
    
    // 检查序列长度 N 是否能被 query 块大小 BR（64）和 key/value 块大小 BN（32）整除。
    if (N % BR != 0 || N % BN != 0) { std::fprintf(stderr, "N must be multiple of %d\n", BR); return 1; }

    // 计算每个输入/输出张量（Q、K、V 或 O）所需的浮点数总个数：总头数 × 序列长度 × 头部维度。
    const size_t n = (size_t)BH * N * D;
    // 在 CPU端分配并初始化连续内存，分别用于存储 Q、K、V 的输入数据，GPU 计算的输出结果 ho，以及用于校验的 CPU 参考结果 href
    std::vector<float> hq(n), hk(n), hv(n), ho(n), href(n);
    // 初始化一个固定随机种子（42）的随机数生成器，并设定一个在 -1.0 到 1.0 之间均匀分布的浮点数分布器，以确保每次运行的结果可复现
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 循环遍历所有元素，将生成的随机数填充进 Q、K、V 的 CPU 数组中
    for (size_t i = 0; i < n; ++i) { hq[i] = dist(rng); hk[i] = dist(rng); hv[i] = dist(rng); }

    // 声明四个指针，准备用于指向 GPU（设备）上申请的显存地址
    // 在 GPU 上为 Q、K、V 矩阵和输出结果矩阵分配对应大小的显存
    float *dq, *dk, *dv, *dout;
    CUDA_CHECK(cudaMalloc(&dq,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dk,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));

    // 将 CPU 上生成好的输入数据（hq, hk, hv）拷贝到刚刚在 GPU 上分配的显存
    CUDA_CHECK(cudaMemcpy(dq, hq.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, hk.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, hv.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // 定义 CUDA 线程网格的维度：x 轴的维度是将总序列长度 N 拆分出的 query 块数，y 轴的维度是所有需要独立计算的批次×头数
    dim3 grid(N / BR, BH);                 // x: query 块数，y: batch*head
    // 计算内核所需的动态共享内存大小，用于存放当前分块的 K 和 V 矩阵
    size_t smem = 2 * BN * (D + 1) * sizeof(float);   // sK + sV（含 padding）
    // 设置一个布尔变量开启“因果掩码”（causal mask），即确保序列前面的 token 无法“看到”它后面的 token，常用于 GPT 等自回归模型
    const bool causal = true;

    mini_flash_attn_kernel<<<grid, NT, smem>>>(dq, dk, dv, dout, N, causal);   // 预热。抵消 CUDA 上下文初始化时的延迟，保证后续测速的准确性
    CUDA_CHECK(cudaDeviceSynchronize());                                       // 阻塞直到执行完成

    // 创建两个 CUDA 事件指针（t0 和 t1）并记录下起始事件 t0
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    // 连续执行 20 次内核计算，以此获取多次运行的平均时间以减小偶然波动误差
    for (int it = 0; it < 20; ++it){
      mini_flash_attn_kernel<<<grid, NT, smem>>>(dq, dk, dv, dout, N, causal);
    }
    // 记录结束事件 t1，并强制 CPU 等待，直到这 20 次内核运算全部在 GPU 上完成
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    // 计算事件 t0 和 t1 之间流逝的总毫秒数并存入 ms 中，随后检查运行过程中是否发生任何隐藏的异步 CUDA 错误。
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaGetLastError());

    // 将 GPU 计算出的结果（dout）拷贝回 CPU 主机内存（ho）中，以便进行正确性校验。
    CUDA_CHECK(cudaMemcpy(ho.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_attention(hq.data(), hk.data(), hv.data(), href.data(), BH, N, D, causal);

    // 遍历所有计算结果，比较 GPU 版本和 CPU 版本的差值，找出整型矩阵中的最大绝对误差值
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) max_err = fmaxf(max_err, fabsf(ho[i] - href[i]));

    /* 全局显存HBM 流量对比:定义并计算理论上传统做法与分块优化（FlashAttention）做法的全局显存（HBM）访问次数。

    6.0 * N * N 的来源（平方项，性能灾难的根源）：
      1 次写： 计算 S = Q * K^T 后，将体积为 N x N 的 $S$ 矩阵写入显存。  
      3 次读： 标准 Softmax 需要多趟遍历 S（一趟求行最大值、一趟求指数和、一趟计算真实概率），需要读 S 矩阵 3 次。  
      1 次写： 将计算出的概率矩阵 P 写入显存。  
      1 次读： 计算输出 O = P * V 时，需要再把 P 从显存读出来 1 次。  
      合计 6 次对 N x N 矩阵的访问，即 6 * N^2。  

    4.0 * N * D 的来源（线性项）：
      读取输入矩阵 Q, K, V 各 1 次，
      体积分别为 N x D，共 3ND。  
      将最终结果矩阵 O 写回显存 1 次，体积为 N x D。
      合计为 4ND。  
    */
    double naive_elem = 6.0 * N * N + 4.0 * N * D;

    /*  分块（FlashAttention）做法的显存流量

    2.0 * N * D 的来源：
      读取查询矩阵 Q 1 次（体积 ND）。  
      写回最终输出矩阵 O 1 次（体积 ND）。  
      不需要一上来就把 K 和 V 全读完。
      合计为 2ND。  

    2.0 * N * N / BR 的来源（反复搬运 K 和 V 的开销）：
      因为矩阵 Q 被分成了多个块，总块数为 N / BR（在本例代码中，BR = 64）。  
      对于 Q 的每一个块，内核都需要去显存中把完整的 K 和 V 矩阵（体积各为 ND，合计 2ND）分批搬运到共享内存中。  
      因此，K 和 V 被重复读取的次数等于 Q 的块数。总读取量为 (N / BR) x 2ND = 2 * N^2 * D / BR。
    */
    double tiled_elem = 2.0 * N * D + 2.0 * N * N * D / BR;

    std::printf("mini-FlashAttention (N=%d, D=%d, causal=%d)\n", N, D, (int)causal);
    std::printf("  max abs error vs CPU ref : %e\n", max_err);
    std::printf("  avg kernel time          : %.3f ms\n", ms / 20.0f);
    std::printf("  HBM element traffic      : %.0f (naive) vs %.0f (tiled), %.1fx less\n",
                naive_elem, tiled_elem, naive_elem / tiled_elem);
    std::printf("%s\n", max_err < 1e-4f ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(dout));
    return 0;
}
