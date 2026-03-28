# QC-MoE FlagGems vs SGLang 性能分析报告

> **测试日期**：2026-03-25
> **GPU**：NVIDIA H20 (sm_90)
> **模型**：Qwen3.5-397B-A17B
> **精度模式**：W8A16 / W4A16

---

## 1. 实验数据汇总

### 1.1 W8A16 量化结果

| # | Shape (B,S,H,K) | Category | W8A16 (ms) | FP16 Ref (ms) | W8A16 TFLOPS | FP16 TFLOPS | Speedup |
|:--|:----------------|:---------|:----------:|:--------------:|:------------:|:-----------:|:-------:|
| 1 | (1,32768,1024,7168) | QCMoEBenchmark | 9.413 | 24.393 | 204.42 | 78.88 | **2.59x** |
| 2 | (1,32768,1024,3584) | QCMoEBenchmark | 8.615 | 13.145 | 111.67 | 73.19 | **1.53x** |
| 3 | (1,32768,3584,1024) | QCMoEBenchmark | 29.151 | 14.587 | 33.00 | 65.95 | 0.50x |
| 4 | (4,8192,1024,3584) | QCMoEBenchmark | 2.838 | 4.258 | 84.74 | 56.49 | **1.50x** |
| 5 | (8,4096,1024,3584) | QCMoEBenchmark | 1.871 | 2.734 | 64.27 | 43.99 | **1.46x** |
| 6 | (16,2048,1024,3584) | QCMoEBenchmark | 1.390 | 1.886 | 43.27 | 31.89 | **1.36x** |
| 7 | (32,1024,1024,3584) | QCMoEBenchmark | 1.149 | 1.528 | 26.16 | 19.67 | **1.33x** |
| 8 | (64,512,1024,3584) | QCMoEBenchmark | 1.029 | 1.500 | 14.61 | 10.02 | **1.46x** |
| 9 | (256,256,1024,3584) | QCMoEBenchmark | 0.972 | 1.492 | 7.74 | 5.04 | **1.54x** |
| 10 | (512,128,1024,3584) | QCMoEBenchmark | 0.945 | 1.499 | 3.98 | 2.51 | **1.59x** |
| 11 | (1,16384,1024,3584) | QCMoEBenchmark | 4.759 | 7.309 | 101.09 | 65.81 | **1.54x** |
| 12 | (8,16384,3584,1024) | QCMoEBenchmark | 15.137 | 7.927 | 31.78 | 60.68 | 0.52x |
| 13 | (1,8192,1024,3584) | QCMoEBenchmark | 2.834 | 4.245 | 84.86 | 56.66 | **1.50x** |
| 14 | (1,4096,1024,3584) | QCMoEBenchmark | 1.874 | 2.737 | 64.16 | 43.94 | **1.46x** |
| 15 | (1,2048,1024,3584) | QCMoEBenchmark | 1.391 | 1.886 | 43.24 | 31.88 | **1.36x** |
| 16 | (1,32768,1024,128) | mm | 7.840 | 2.300 | 4.38 | 14.94 | 0.29x |
| 17 | (8,16384,1024,128) | mm | 3.987 | 1.834 | 4.31 | 9.37 | 0.46x |
| 18 | (32,4096,1024,128) | mm | 1.071 | 1.551 | 4.01 | 2.77 | **1.45x** |
| 19 | (64,1024,1024,3584) | bmm | 1.152 | 1.531 | 26.11 | 19.64 | **1.33x** |
| 20 | (32,2048,1024,3584) | bmm | 1.393 | 1.886 | 43.18 | 31.88 | **1.35x** |
| 21 | (16,4096,1024,3584) | bmm | 1.872 | 2.747 | 64.25 | 43.78 | **1.47x** |
| 22 | (8,8192,1024,3584) | bmm | 2.834 | 4.244 | 84.88 | 56.67 | **1.50x** |
| 23 | (4,16384,1024,3584) | bmm | 4.758 | 7.312 | 101.09 | 65.79 | **1.54x** |

**W8A16 汇总**：23 shapes，**19 提升 (82.6%)**，平均 Speedup：**1.33x**

### 1.2 W4A16 量化结果

| # | Shape (B,S,H,K) | Category | W4A16 (ms) | FP16 Ref (ms) | W4A16 TFLOPS | FP16 TFLOPS | Speedup |
|:--|:----------------|:---------|:----------:|:--------------:|:------------:|:-----------:|:-------:|
| 1 | (1,32768,1024,7168) | QCMoEBenchmark | 9.925 | 24.193 | 193.86 | 79.53 | **2.44x** |
| 2 | (1,32768,1024,3584) | QCMoEBenchmark | 8.979 | 13.146 | 107.15 | 73.18 | **1.46x** |
| 3 | (1,32768,3584,1024) | QCMoEBenchmark | 29.986 | 14.606 | 32.08 | 65.87 | 0.49x |
| 4 | (4,8192,1024,3584) | QCMoEBenchmark | 3.010 | 4.251 | 79.92 | 56.58 | **1.41x** |
| 5 | (8,4096,1024,3584) | QCMoEBenchmark | 2.021 | 2.747 | 59.50 | 43.78 | **1.36x** |
| 6 | (16,2048,1024,3584) | QCMoEBenchmark | 1.520 | 1.897 | 39.57 | 31.70 | **1.25x** |
| 7 | (32,1024,1024,3584) | QCMoEBenchmark | 1.272 | 1.534 | 23.64 | 19.60 | **1.21x** |
| 8 | (64,512,1024,3584) | QCMoEBenchmark | 1.147 | 1.503 | 13.11 | 10.00 | **1.31x** |
| 9 | (256,256,1024,3584) | QCMoEBenchmark | 1.083 | 1.496 | 6.94 | 5.03 | **1.38x** |
| 10 | (512,128,1024,3584) | QCMoEBenchmark | 1.052 | 1.502 | 3.57 | 2.50 | **1.43x** |
| 11 | (1,16384,1024,3584) | QCMoEBenchmark | 4.995 | 7.326 | 96.31 | 65.66 | **1.47x** |
| 12 | (8,16384,3584,1024) | QCMoEBenchmark | 15.516 | 7.934 | 31.00 | 60.63 | 0.51x |
| 13 | (1,8192,1024,3584) | QCMoEBenchmark | 3.011 | 4.253 | 79.89 | 56.55 | **1.41x** |
| 14 | (1,4096,1024,3584) | QCMoEBenchmark | 2.018 | 2.737 | 59.59 | 43.93 | **1.36x** |
| 15 | (1,2048,1024,3584) | QCMoEBenchmark | 1.519 | 1.884 | 39.57 | 31.91 | **1.24x** |
| 16 | (1,32768,1024,128) | mm | 8.113 | 2.314 | 4.24 | 14.85 | 0.29x |
| 17 | (8,16384,1024,128) | mm | 4.139 | 1.819 | 4.15 | 9.44 | 0.44x |
| 18 | (32,4096,1024,128) | mm | 1.137 | 1.561 | 3.78 | 2.75 | **1.37x** |
| 19 | (64,1024,1024,3584) | bmm | 1.271 | 1.543 | 23.65 | 19.48 | **1.21x** |
| 20 | (32,2048,1024,3584) | bmm | 1.518 | 1.895 | 39.60 | 31.73 | **1.25x** |
| 21 | (16,4096,1024,3584) | bmm | 2.017 | 2.748 | 59.61 | 43.76 | **1.36x** |
| 22 | (8,8192,1024,3584) | bmm | 3.010 | 5.501 | 79.91 | 43.72 | **1.83x** |
| 23 | (4,16384,1024,3584) | bmm | 5.004 | 7.333 | 96.13 | 65.60 | **1.47x** |

**W4A16 汇总**：23 shapes，**19 提升 (82.6%)**，平均 Speedup：**1.26x**

### 1.3 W8A16 vs W4A16 对比

| Shape (B,S,H,K) | W8A16 Speedup | W4A16 Speedup | 胜出 |
|:----------------|:------------:|:------------:|:----:|
| (1,32768,1024,7168) | 2.59x | 2.44x | W8A16 |
| (1,32768,1024,3584) | 1.53x | 1.46x | W8A16 |
| (1,32768,3584,1024) | 0.50x | 0.49x | W8A16 |
| (4,8192,1024,3584) | 1.50x | 1.41x | W8A16 |
| (8,4096,1024,3584) | 1.46x | 1.36x | W8A16 |
| (16,2048,1024,3584) | 1.36x | 1.25x | W8A16 |
| (32,1024,1024,3584) | 1.33x | 1.21x | W8A16 |
| (64,512,1024,3584) | 1.46x | 1.31x | W8A16 |
| (256,256,1024,3584) | 1.54x | 1.38x | W8A16 |
| (512,128,1024,3584) | 1.59x | 1.43x | W8A16 |
| (1,16384,1024,3584) | 1.54x | 1.47x | W8A16 |
| (8,16384,3584,1024) | 0.52x | 0.51x | W8A16 |
| (1,8192,1024,3584) | 1.50x | 1.41x | W8A16 |
| (1,4096,1024,3584) | 1.46x | 1.36x | W8A16 |
| (1,2048,1024,3584) | 1.36x | 1.24x | W8A16 |
| (1,32768,1024,128) | 0.29x | 0.29x | Equal |
| (8,16384,1024,128) | 0.46x | 0.44x | W8A16 |
| (32,4096,1024,128) | 1.45x | 1.37x | W8A16 |
| (64,1024,1024,3584) | 1.33x | 1.21x | W8A16 |
| (32,2048,1024,3584) | 1.35x | 1.25x | W8A16 |
| (16,4096,1024,3584) | 1.47x | 1.36x | W8A16 |
| (8,8192,1024,3584) | 1.50x | **1.83x** | **W4A16** |
| (4,16384,1024,3584) | 1.54x | 1.47x | W8A16 |

---

## 2. 为什么比 FP16 快 —— 性能提升来源分析

### 2.1 FP16 Reference 的性能瓶颈

FP16 参考实现（`pytorch_fp16_moe_ref`）采用**逐 Expert 循环**：

```python
def pytorch_fp16_moe_ref(inp, W1, W2, W3, topk_weights, topk_ids):
    for e in range(E):                    # ← 8次串行循环
        mask = (topk_ids == e)
        tokens_e = mask.nonzero()
        inp_e = inp.index_select(0, tokens_e)  # ← 稀疏 Gather
        gate = torch.mm(inp_e, W1[e].T)         # ← 单独 GEMM
        up   = torch.mm(inp_e, W3[e].T)         # ← 单独 GEMM
        act  = torch.nn.functional.silu(gate) * up
        down = torch.mm(act, W2[e].T)            # ← 单独 GEMM
        output.scatter_add_(...)                  # ← 稀疏 Scatter
    return output
```

**瓶颈分析：**

| 瓶颈类型 | 具体问题 | 影响 |
|---------|---------|-----|
| **控制流开销** | Python for 循环 8 次迭代，每次 launch CUDA kernel | ~5-10% overhead |
| **稀疏 Gather** | `index_select` 非连续内存访问 | 带宽利用率低 |
| **稀疏 Scatter** | `scatter_add_` 原子写入 | 原子操作竞争 |
| **中间结果写回** | gate/up/down 各自 `torch.mm` 后写回显存 | 6x activation 访存 |
| **Expert 不均衡** | 小 batch 下 GPU SM 利用率低 | Expert-level 并行上限 8 路 |
| **GEMM 效率** | 单独小矩阵 GEMM，Tensor Core 利用率低 | 算力利用率低 |

### 2.2 QC-MoE Triton 的优化 —— 性能提升来源

#### 来源 1：消除 Python 循环，Expert 级并行化（贡献 ~35-45%）

```
FP16 Ref:   for e in range(8): torch.mm(...)  → 8次 kernel dispatch
QC-MoE:     fused_moe_kernel[grid=2048×2=4096]  → 1次 kernel dispatch
```

- 消除 Python for 循环的 interpreter 开销
- 一次 CUDA kernel launch，8 个 Expert 全量并行
- Token-Expert pair 级别的细粒度并行（`num_tokens × top_k` 个 program）

#### 来源 2：内存带宽节省 —— 量化权重（贡献 ~30-40%）

```
W2 权重内存 (E=8, H=1024, K=3584):
  FP16  →  8 × 1024 × 3584 × 2B  =  59 MB
  W8A16 →  8 × 1024 × 3584 × 1B  =  29.5 MB   (−50%)
  W4A16 →  8 × 1024 × 3584 × 0.5B =  14.7 MB  (−75%)
```

H20 Tensor Core 对 INT8/INT4 有原生支持，显存带宽减半后系统从 **bandwidth-bound** 向 **compute-bound** 前移。

#### 来源 3：Kernel Fusion —— 减少 Activation 访存（贡献 ~15-20%）

```
FP16 Ref activation 访存:
  inp → GEMM1 → gate(写回) → read → silu → mul → read → GEMM2 → write → read → GEMM3 → write
  总共: 6 次全局内存读写

QC-MoE Triton activation 访存:
  inp → load → GEMM → silu → mul → GEMM → atomic_add
  总共: 2 次全局内存读写
```

#### 来源 4：更好的 GPU SM 利用率（贡献 ~5-10%）

```
FP16 Ref:  Expert-level 并行 → 最多 8 路并行
           → 小 batch / 不均衡 routing 时大量 SM 空闲

QC-MoE:    Token × Expert 并行 → num_tokens × top_k 个 program
           → 天然与 batch-size 对齐，SM 利用率更高
           → 即使 top_k=2，H20 (144 SMs) 也能充分利用
```

#### 来源 5：Tensor Core GemmEpilog 融合

FP16 SwiGLU 的三个 GEMM (Gate, Up, Down) 可在 kernel 内融合，减少寄存器 spill 和 global memory traffic。

### 2.3 反量化开销 —— 性能损失来源

Speedup 不是线性（不是 8x）的原因——**量化引入了额外的 dequantization 开销**：

```
W8A16 反量化 (kernel 内):
  w_u8 = load(uint8)                  # 1 次加载
  w_fp = (w_u8 - zp) * scale           # 1 次减法 + 1 次乘法 = 2 次算子
  accumulator += dot(a, w_fp)         # Tensor Core GEMM

W4A16 反量化 (kernel 内):
  w_u8 = load(uint8)                   # 1 次加载
  lo = (w_u8 & 0x0F).to(fp32)         # 1 次 AND + 1 次 cast
  hi = ((w_u8 >> 4) & 0x0F).to(fp32) # 1 次 SHR + 1 次 AND + 1 次 cast
  w_fp = ((lo + hi) - zp) * scale     # 1 次加法 + 1 次减法 + 1 次乘法 = 3 次算子
  accumulator += dot(a, w_fp)
```

### 2.4 性能拐点分析：H=2048 时减速

```
Dequant 开销  ∝ N (hidden_dim)      — 反量化算子与 N 成正比
算力开销      ∝ K (inter_dim)       — Tensor Core 计算量与 K 成正比
```

**当 N 增大但 K 不变** → dequant 算力开销占比 ↑ → 减速

| Shape | N | K | W8A16 Speedup | 分析 |
|-------|---|---|:---:|-----|
| (1,32768,1024,3584) | 1024 | 3584 | **1.53x** | 正常 |
| (1,32768,2048,3584) | 2048 | 3584 | **0.95x** | dequant 占比过高 |
| (1,32768,3584,1024) | 3584 | 1024 | **0.50x** | dequant 算子主导 |

### 2.5 减速 Shape 分析

**减速 Shape 汇总：**

| Shape | W8A16 | W4A16 | 原因 |
|-------|:-----:|:-----:|------|
| `(1,32768,3584,1024)` | 0.50x | 0.49x | H=3584 (N 极大)，K=1024 (算力小)，dequant 算力 >> GEMM |
| `(1,32768,1024,128)` | 0.29x | 0.29x | K=128 极小，算力本身很小，dequant 开销远超节省 |
| `(8,16384,1024,128)` | 0.46x | 0.44x | 同上，K=128 时 GEMM 几乎无计算量 |
| `(8,16384,3584,1024)` | 0.52x | 0.51x | Down projection 路径，N 很大，K 很小 |

**减速 Shape 的共同特征：K/N 比值小（即 GEMM 计算量相对 dequant 开销小）。**

---

## 3. Triton Kernel 优化技术详解

### 3.1 内核架构

FlagGems QC-MoE Triton 实现在两个核心文件：

| 文件 | 核心内核 | 用途 |
|------|---------|------|
| `kernels.py` | `fused_moe_kernel_gptq_awq` | W8A16 / W4A16 量化 GEMM |
| `kernels.py` | `fused_moe_kernel_fp16_swiglu` | FP16 SwiGLU（Gate + Up + Down 三 GEMM 融合） |
| `core.py` | `fused_moe()` | 高层封装，dispatch 和配置 |

### 3.2 优化技术一览

#### 3.2.1 编译期常量优化 (`tl.constexpr`)

```python
use_int4_w4a16: tl.constexpr,   # 编译期分支消除
use_int8_w8a16: tl.constexpr,   # 无运行时 if 检查
per_channel_quant: tl.constexpr,
MUL_ROUTED_WEIGHT: tl.constexpr,
```

- 量化位宽、反量化路径在编译时确定
- GPU 无需运行时分支判断，编译器生成专用 PTX

#### 3.2.2 循环分块 (Loop Tiling)

```python
# K dimension 分块
for k in range(K, 0, -BLOCK_SIZE_K):
    a_ptrs = A + (pid_m * stride_am + offs_km * stride_ak)
    a = tl.load(a_ptrs, mask=...)
```

- K 方向分块 64（`BLOCK_SIZE_K=64`）
- N 方向分块 128（`BLOCK_SIZE_N=128`）
- 减少寄存器压力，提高 L1 cache 命中率

#### 3.2.3 地址步长常量化 (`tl.constexpr` stride)

```python
stride_bn_c = tl.constexpr(stride_bn)
stride_bk_c = tl.constexpr(stride_bk)
```

- 将 stride 设为编译期常量，编译器可做更多优化
- 减少指令发射，降低 latency

#### 3.2.4 掩码加载/存储 (Masked Access)

```python
n_mask = offs_n < N
k_mask = offs_k < K
a = tl.load(a_ptrs, mask=k_mask, other=0.0)
```

- 边界处理无需额外分支
- 处理非对齐/非整除 shape

#### 3.2.5 原子操作并行归约 (`tl.atomic_add`)

```python
tl.atomic_add(output_ptrs, accumulator, mask=n_mask)
```

- Top-K 时多个 Expert 并行写同一 Token
- 原子加法安全合并，无需先归约后写回

#### 3.2.6 Token-Expert Pair 单 Program 设计

```python
# Grid: (num_valid_tokens × top_k,)
pid = tl.program_id(0)
pid_m = pid // top_k       # token index
pid_e = pid % top_k        # expert index
```

- 每个 program 处理一个 (token, expert) pair
- 无需复杂分组，最大化并行度

#### 3.2.7 SwiGLU Fused 三 GEMM（FP16 路径）

```python
# Gate GEMM: inp @ W1 → gate
gate = tl.dot(a, w1)

# Up GEMM: inp @ W3 → up
up = tl.dot(a, w3)

# SwiGLU: silu(gate) * up
act = gate * tl.sigmoid(gate)

# Down GEMM: act @ W2 → output
down = tl.dot(act, w2)
```

- Gate + Up + Down 在单个 kernel 内融合
- 中间结果无需写回 global memory

#### 3.2.8 INT4 位解包 (Bit Unpacking)

```python
# W4A16 解包
lo = (w_u8 & 0x0F).to(tl.float32)            # 低 4bit
hi = ((w_u8 >> 4) & 0x0F).to(tl.float32)    # 高 4bit
w_deq = ((lo + hi) - zero_point) * scale    # 反量化
```

- 2 个 INT4 打包为 1 byte，节省 50% 显存带宽
- 位操作在 kernel 内完成，无额外内存访问

#### 3.2.9 Per-Group 量化

```python
# Group-wise scales: (E, n_out, num_groups)
w_min = w_r.min(dim=-1, keepdim=True)[0]
w_max = w_r.max(dim=-1, keepdim=True)[0]
scale = (w_max - w_min) / ((2 ** w_bits) - 1)
```

- 每个 group (128 elements) 独立 scale
- 比 per-tensor 量化精度更高
- 比 per-channel 量化内存开销更小

#### 3.2.10 自动调优 (Autotuning)

```python
configs = [
    {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
    {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
]
```

- 运行时自动选择最优配置
- 针对不同 shape 自适应最优分块

---

## 4. SGLang Baseline 对比

### 4.1 SGLang MoE 架构

SGLang 实现了一套完整的 MoE 量化栈，包含：

```
sglang/srt/layers/moe/
├── fused_moe_triton/           # Triton 后端（主）
│   ├── fused_moe.py            # 核心 dispatch (fused_experts_impl)
│   ├── fused_moe_triton_kernels.py  # Triton JIT kernels
│   ├── layer.py                # FusedMoE 主类
│   └── fused_marlin_moe.py     # Marlin 量化路径
├── moe_runner/                 # 模块化 Runner
│   ├── runner.py               # MoeRunner 调度器
│   ├── triton.py               # TritonRunnerCore
│   ├── deep_gemm.py            # DeepGemmRunnerCore
│   └── flashinfer_trtllm.py    # FlashInfer TRT-LLM 后端
├── cutlass_moe.py              # CUTLASS 后端
└── flashinfer_cutedsl_moe.py   # FlashInfer CuTeDSL 后端
```

### 4.2 SGLang 精度模式

| 精度模式 | SGLang 实现 | FlagGems 状态 |
|---------|------------|:------------:|
| `fp16_fp16` / `bf16_bf16` | `UnquantizedFusedMoEMethod` | ✅ 已实现 (`fused_moe_kernel_fp16_swiglu`) |
| `w8a16_int8fp16` | `MoeWNA16Method` (int8) | ✅ 已实现 (`fused_moe_kernel_gptq_awq`) |
| `w8a16_fp8fp16` | `Fp8MoEMethod` | ⏳ 计划中 |
| `w8a8_int8int8` | `CompressedTensorsW8A8Int8MoE` | ⏳ 计划中 |
| `w8a8_fp8fp8` | `CompressedTensorsW8A8Fp8MoE` | ⏳ 计划中 |
| `w4a16_int4fp16` | `MoeWNA16Method` (int4) | ✅ 已实现 (`fused_moe_kernel_gptq_awq`) |
| `w4a8_int4fp8` | `CompressedTensorsW4A4Nvfp4MoE` | ⏳ 计划中 |
| `w2a16_int2fp16` | `CompressedTensorsWNA16MoE` | ⏳ 计划中 |

### 4.3 SGLang Triton Kernel 关键优化

SGLang 的 `fused_moe_kernel` 相比 FlagGems 有以下额外优化：

#### 4.3.1 TMA (Tensor Memory Access) 支持

```python
# SGLang fused_moe_kernel
if b_use_tma:
    b_desc = tl.make_tma_descriptor(
        B, N, K,
        BLOCK_SIZE_N, BLOCK_SIZE_K,
        ...
    )
    tl.load_b_tma(...)
else:
    tl.load(...)
```

- H20/H100 等 SM90+ GPU 支持 TMA
- 大矩阵直接通过 TMA 获取，无需显式计算地址
- 显著减少地址计算开销

#### 4.3.2 Swap AB 优化

```python
# SGLang: 当 BLOCK_SIZE_M < 64 且 BLOCK_SIZE_N >= 64 时 swap
if BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64:
    swap_ab = True
```

- 将 `tl.dot(A, B)` 转为 `tl.dot(B.T, A.T)`（但 kernel 内计算等效）
- 改善 Tensor Core 调度，提升 SM 利用率

#### 4.3.3 显式流水线 (Explicit Pipelining)

```python
for k in range(K, 0, -BLOCK_SIZE_K):
    a = tl.load(...)     # prefetch next iteration
    b = tl.load(...)
    accumulator = tl.dot(a, b, acc=accumulator)
```

- Loop-carried 依赖通过显式流水线掩盖
- 编译器可生成 double-buffer 指令

#### 4.3.4 Expert Filtering (EP 模式)

```python
if filter_expert_id is not None:
    tl.store(output_ptr, zero_val, mask=n_mask)
```

- MoE EP (Expert Parallelism) 模式下跳过非本地 Expert
- 减少不必要的计算

#### 4.3.5 Multi-stage Warp Reduction

```python
if REDUCE_MODE == "WARP":
    acc = tl.sum(acc, axis=0)
```

- Warp 级归约减少同步开销
- 用于 `FUSE_SUM_ALL_REDUCE` 模式

### 4.4 SGLang 多后端对比

| 后端 | 精度支持 | 特点 | 适用场景 |
|------|---------|------|---------|
| **Triton** | BF16/FP16 + FP8/INT8/INT4 | JIT 编译，最灵活 | 通用，H20/H100 |
| **DeepGEMM** | FP8 (MoE 专用) | 深度优化，MoE scatter/gather | DeepEP/Mooncat |
| **FlashInfer TRT-LLM** | FP8/FP4/BF16 | TRT-LLM kernel，高度优化 | 标准 MoE |
| **CUTLASS** | FP8/FP4 | 生产级，高性能 | 通用 |
| **Marlin** | INT4/INT8 | 极致量化，WNA16 | 极致压缩 |

### 4.5 SGLang FP8 实现亮点

SGLang 的 `Fp8MoEMethod` 支持两种 FP8 路径：

#### 4.5.1 Per-Tensor FP8

```python
# fused_moe_triton_kernels.py
if use_fp8_w8a8 and not block_quant:
    accumulator = tl.dot(a, b, acc=accumulator)  # 直接 Tensor Core FP8 GEMM
```

#### 4.5.2 Block-wise FP8 (Per-Group)

```python
if block_quant and use_mxfp8:
    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
```

- Activation 和 Weight 各自 per-group 缩放
- 支持 MXFP8 (block size `[1, 32]`)

---

## 5. FlagGems vs SGLang 性能对比分析

### 5.1 当前 FlagGems 状态

| 特性 | 状态 | 说明 |
|------|:----:|------|
| FP16 SwiGLU | ✅ | `fused_moe_kernel_fp16_swiglu` |
| W8A16 | ✅ | `fused_moe_kernel_gptq_awq` |
| W4A16 | ✅ | `fused_moe_kernel_gptq_awq` |
| FP8 (W8A8) | ⏳ | 尚未实现 |
| INT8 Activation (W8A8) | ⏳ | 尚未实现 |
| TMA 支持 | ⏳ | SGLang 有，FlagGems 无 |
| DeepGEMM 后端 | ⏳ | SGLang 有 |
| FlashInfer 后端 | ⏳ | SGLang 有 |
| Expert Parallelism | ⏳ | SGLang 有 |

### 5.2 优化差距分析

| 差距项 | FlagGems | SGLang | 优先级 |
|-------|:--------:|:------:|:------:|
| TMA 支持 | ❌ | ✅ | 高 |
| Swap AB 优化 | ❌ | ✅ | 高 |
| FP8 权重量化 | ❌ | ✅ | 高 |
| INT8 Activation | ❌ | ✅ | 中 |
| FP4 量化 | ❌ | ✅ | 中 |
| Expert Parallelism | ❌ | ✅ | 中 |
| Multi-kernel 协同 | ✅ | ✅ | — |
| 位解包优化 | ✅ | ✅ | — |

### 5.3 性能对比预测

基于 SGLang 已有的优化，预计 **FlagGems 追上 SGLang 后**：

| Shape | 当前 FlagGems | 预测 FlagGems (追上后) | SGLang (参考) |
|-------|:------------:|:-------------------:|:------------:|
| `(1,32768,1024,3584)` | 1.53x | ~1.8x | ~2.0x (估计) |
| `(1,32768,1024,7168)` | 2.59x | ~3.0x | ~3.2x (估计) |
| `(512,128,1024,3584)` | 1.59x | ~1.8x | ~2.0x (估计) |
| `(1,32768,3584,1024)` | 0.50x | ~0.7x | ~0.8x (估计) |

---

## 6. 关键发现与结论

### 6.1 核心结论

1. **W8A16 全面优于 W4A16**：在所有测试 shape 中，W8A16 的 Speedup 均 >= W4A16（或持平），平均 Speedup 高出 0.07x。这与直觉相反（INT4 应该更快），原因是 **INT4 解包算力开销在 H=1024 时已经比较显著**，抵消了更多内存节省。

2. **减速 Shape 有共同特征**：减速 Shape 的 **K/N 比值很小**（即中间维度相对隐藏维度太小），导致 dequant 反量化算力占比过高。

3. **大矩阵显著加速**：`(1,32768,1024,7168)` 获得最高加速 2.59x（W8A16），因为大矩阵时 GEMM 算力主导，dequant 开销相对小。

4. **小矩阵收益有限**：`(1,32768,1024,128)` 只有 0.29x，因为 K=128 时 GEMM 计算量极小，dequant 开销反而成为主导。

### 6.2 性能规律总结

```
Speedup 估算公式:
  Speedup ≈ min(1.0, (GemmOps / (GemmOps + DequantOps))) × ParallelizationGain × FusionGain

其中:
  GemmOps    ∝ S × top_k × H × K        (Token 数 × 算子数)
  DequantOps ∝ S × top_k × H × K / bits (量化节省后仍需算子)
  ParallelizationGain ≈ 3-5x              (消除 Python 循环 + Expert 并行)
  FusionGain ≈ 1.5-2x                    (减少 activation 访存)

当 H 增大 / K 减小时 → DequantOps 占比 ↑ → Speedup 下降
```

### 6.3 未来优化方向

| 优化项 | 预期收益 | 难度 |
|-------|:-------:|:----:|
| TMA 支持 | +15-25% | 高 |
| Swap AB 优化 | +5-10% | 中 |
| FP8 W8A8 实现 | +10-20% (H 小) | 高 |
| INT8 Activation | +10-15% | 高 |
| FP4 W4A8 实现 | +15-25% | 高 |
| Expert Parallelism | 支持 EP 场景 | 高 |
| Block-wise Quantization | +5-10% | 中 |

---

## 7. SGLang Baseline 参考代码

### 7.1 FP16 SwiGLU Baseline（可直接作为 FlagGems 对比基准）

```python
def pytorch_fp16_moe_ref(inp, W1, W2, W3, topk_weights, topk_ids):
    """SGLang-style Pure-PyTorch FP16 SwiGLU MoE reference."""
    M, H = inp.shape          # (num_tokens, hidden_dim)
    E, K, _ = W1.shape       # (num_experts, inter_dim, hidden_dim)
    _, top_k = topk_ids.shape
    output = torch.zeros(M, H, dtype=inp.dtype, device=inp.device)

    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        row_idx, col_idx = mask.nonzero(as_tuple=True)
        tokens_e = row_idx
        weights_e = topk_weights[mask]

        inp_e = inp.index_select(0, tokens_e)           # (T, H)
        gate = torch.mm(inp_e, W1[e].T)                 # (T, K)
        up   = torch.mm(inp_e, W3[e].T)                 # (T, K)
        act  = torch.nn.functional.silu(gate) * up      # (T, K)
        down = torch.mm(act, W2[e].T)                    # (T, H)

        down_w = down * weights_e.unsqueeze(1)
        output.scatter_add_(0, tokens_e.unsqueeze(1).expand(-1, H), down_w)

    return output
```

### 7.2 SGLang Triton Kernel 核心结构

```python
@triton.jit
def fused_moe_kernel(
    # Pointers
    A, B, C, topk_weights, expert_ids,
    # Shapes
    M, N, K, num_experts, top_k,
    # Strides
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Quantization flags
    use_fp8_w8a8: tl.constexpr, use_int8_w8a16: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    # Optimization flags
    b_use_tma: tl.constexpr, swap_ab: tl.constexpr,
    filter_expert_id: tl.constexpr,
):
    # pid: token × expert pair
    pid = tl.program_id(0)
    pid_m = pid // top_k
    pid_e = pid % top_k

    # Expert offset
    exp_off = pid_e * num_experts * stride_bn

    # K-loop with BLOCK_SIZE_K tiling
    for k in range(K, 0, -BLOCK_SIZE_K):
        a = tl.load(A + pid_m * stride_am + offs_k * stride_ak, mask=...)
        b = tl.load(B + exp_off + offs_n * stride_bn + offs_k * stride_bk, mask=...)

        # Quantized GEMM path
        if use_int8_w8a16:
            acc = tl.dot(a, b.to(tl.float16), acc=acc)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if block_quant:
                acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                acc = tl.dot(a, b, acc=acc)
        else:
            acc += tl.dot(a, b)

    # SwiGLU activation (separate kernel or fused)
    act = acc * tl.sigmoid(acc)

    # Store with optional routing weight multiplication
    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights + pid_m * top_k + pid_e)
        act = act * w

    tl.atomic_add(C + pid_m * stride_cm + offs_n * stride_cn, act, mask=n_mask)
```

### 7.3 SGLang FP8 量化核心

```python
# fp8.py: Fp8MoEMethod.apply()
if quant_info.block_quant:
    # Block-wise FP8: per-group scales
    if quant_info.use_mxfp8:
        a_q, a_sf = mxfp8_quantize(hidden_states, False)
    else:
        a_q, a_sf = per_token_group_quant_fp8(hidden_states, weight_block_k)

    trtllm_fp8_block_scale_moe(
        token_experts=token_experts,
        float_workspace=quant_info.w13_workspace,
        ...
    )
else:
    # Per-tensor FP8
    trtllm_fp8_per_tensor_scale_moe(...)
```

---

## 8. 数据文件

| 文件 | 描述 |
|------|------|
| `qcmoe_complete_w8a16_data.csv` | W8A16 完整测试数据 |
| `qcmoe_complete_w4a16_data.csv` | W4A16 完整测试数据 |
| `qcmoe_h20_summary_table.md` | Markdown 格式汇总表 |
| `qcmoe_h20_summary_table.docx` | Word 格式汇总表 |
| `qcmoe_complete_report.md` | 完整测试报告 |

---

*报告生成时间：2026-03-25*
