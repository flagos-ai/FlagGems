# QC-GEM W8A16 / W4A16 Benchmark: 结果分析文档

> FlagGems Triton Kernel vs PyTorch FP16 GEMM
> 日期: 2026-03-27 | GPU: NVIDIA H20 | 模型: Qwen3.5-397B-A17B

---

## 1. 测试概述

### 1.1 测试目标

评估 **FlagGems QC-GEM W8A16 / W4A16 Triton Kernel** 相对于 **PyTorch FP16 GEMM** 的性能表现，涵盖 MoE FFN 投影层的典型 shape。

### 1.2 量化方案

| 模式 | 权重量化 | 激活值 | Group Size | 量化方式 |
|------|---------|--------|------------|---------|
| **W8A16** | INT8 | FP16 | 128 | per-group min-max 均匀量化，含零偏 |
| **W4A16** | INT4（每字节 2 值） | FP16 | 128 | per-group min-max 均匀量化，含零偏 |

- W4A16: 每 2 个 int4 nibble pack 进 1 byte；Triton kernel 内 on-the-fly unpack + 反量化
- W8A16: 每 byte 1 个 uint8；Triton kernel 内直接反量化
- FP16 参考: PyTorch 标准 GEMM（`torch.mm`）

### 1.3 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA H20 (sm_90a) |
| 精度 | FP16 |
| 框架 | PyTorch + FlagGems Triton Kernel |
| Triton 版本 | 见 GemLite 环境 |
| Benchmark 循环 | 10 次计时，3 次 warmup |
| 随机种子 | 42 |

### 1.4 测试 Shape 分类

Shapes 来自 `models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml` QCGEMBenchmark 段，`(M, N, K)` 格式（M = batch×seq tokens）。

| 分类 | Matrix (N×K) | 说明 |
|------|-------------|------|
| Down projection | 1024×3584 | MoE FFN Down 投影（最大矩阵） |
| Up projection | 3584×1024 | MoE FFN Up 投影 |
| Gate projection | 1024×7168 | MoE Gate 投影 |
| Router | 128×1024 | Router / scoring 层（最小 N） |

---

## 2. 核心结果

### 2.1 W8A16 性能汇总表

| Matrix (N×K) | M (tokens) | W8A16 (ms) | FP16 ref (ms) | W8A16 Speedup | QC TFLOPS | FP16 TFLOPS |
|-------------|-----------|-----------|---------------|--------------|-----------|-------------|
| 1024×3584 | 32768 | 2.5630 | 2.6303 | **1.03x** | 93.84 | 93.84 |
| 1024×3584 | 16384 | 1.3943 | 1.4074 | **1.01x** | 86.25 | 86.25 |
| 1024×3584 | 8192 | 0.8118 | 0.8184 | **1.01x** | 74.07 | 74.07 |
| 1024×3584 | 4096 | 0.5052 | 0.5114 | **1.01x** | 59.51 | 59.51 |
| 1024×3584 | 2048 | 0.3708 | 0.3568 | **0.96x** | 40.54 | 40.54 |
| 1024×3584 | 1024 | 0.3625 | 0.2785 | **0.77x** | 20.73 | 20.73 |
| 1024×3584 | 512 | 0.3184 | 0.2377 | **0.75x** | 11.80 | 11.80 |
| 1024×3584 | 256 | 0.2878 | 0.2213 | **0.77x** | 6.53 | 6.53 |
| 3584×1024 | 32768 | 2.6841 | 2.7118 | **1.01x** | 89.61 | 89.61 |
| 3584×1024 | 16384 | 1.3905 | 1.3957 | **1.00x** | 86.49 | 86.49 |
| 3584×1024 | 8192 | 0.7868 | 0.7924 | **1.01x** | 76.42 | 76.42 |
| 3584×1024 | 4096 | 0.4853 | 0.4903 | **1.01x** | 61.95 | 61.95 |
| 3584×1024 | 2048 | 0.3590 | 0.3458 | **0.96x** | 41.88 | 41.88 |
| 1024×7168 | 32768 | 5.4988 | 5.3918 | **0.98x** | 87.48 | 87.48 |
| 1024×7168 | 16384 | 2.7408 | 2.7420 | **1.00x** | 87.76 | 87.76 |
| 1024×7168 | 8192 | 1.5384 | 1.5429 | **1.00x** | 78.17 | 78.17 |
| 1024×7168 | 4096 | 0.9337 | 0.9387 | **1.00x** | 64.40 | 64.40 |
| 1024×7168 | 2048 | 0.6314 | 0.6368 | **1.01x** | 47.62 | 47.62 |
| 128×1024 | 32768 | 0.3418 | 0.1999 | **0.58x** | 25.13 | 25.13 |
| 128×1024 | 16384 | 0.2991 | 0.2000 | **0.67x** | 14.36 | 14.36 |
| 128×1024 | 4096 | 0.2899 | 0.1941 | **0.67x** | 3.70 | 3.70 |

### 2.2 W4A16 性能汇总表

| Matrix (N×K) | M (tokens) | W4A16 (ms) | FP16 ref (ms) | W4A16 Speedup | QC TFLOPS | FP16 TFLOPS |
|-------------|-----------|-----------|---------------|--------------|-----------|-------------|
| 1024×3584 | 32768 | 2.5630 | 2.6453 | **1.03x** | 93.84 | 93.84 |
| 1024×3584 | 16384 | 1.3414 | 1.3967 | **1.04x** | 89.65 | 89.65 |
| 1024×3584 | 8192 | 0.7506 | 0.7575 | **1.01x** | 80.11 | 80.11 |
| 1024×3584 | 4096 | 0.4417 | 0.4491 | **1.02x** | 68.07 | 68.07 |
| 1024×3584 | 2048 | 0.3380 | 0.2825 | **0.84x** | 44.48 | 44.48 |
| 1024×3584 | 1024 | 0.3252 | 0.2083 | **0.64x** | 23.12 | 23.12 |
| 1024×3584 | 512 | 0.2885 | 0.1856 | **0.64x** | 13.03 | 13.03 |
| 1024×3584 | 256 | 0.2810 | 0.1856 | **0.66x** | 6.69 | 6.69 |
| 3584×1024 | 32768 | 2.6792 | 2.7097 | **1.01x** | 89.77 | 89.77 |
| 3584×1024 | 16384 | 1.3464 | 1.3520 | **1.00x** | 89.32 | 89.32 |
| 3584×1024 | 8192 | 0.7308 | 0.7384 | **1.01x** | 82.28 | 82.28 |
| 3584×1024 | 4096 | 0.4218 | 0.4280 | **1.01x** | 71.28 | 71.28 |
| 3584×1024 | 2048 | 0.3355 | 0.2738 | **0.82x** | 44.80 | 44.80 |
| 1024×7168 | 32768 | 5.2889 | 5.3654 | **1.01x** | 90.95 | 90.95 |
| 1024×7168 | 16384 | 2.6531 | 2.6621 | **1.00x** | 90.65 | 90.65 |
| 1024×7168 | 8192 | 1.4260 | 1.4318 | **1.00x** | 84.33 | 84.33 |
| 1024×7168 | 4096 | 0.8089 | 0.8156 | **1.01x** | 74.33 | 74.33 |
| 1024×7168 | 2048 | 0.4998 | 0.5049 | **1.01x** | 60.16 | 60.16 |
| 128×1024 | 32768 | 0.3093 | 0.1865 | **0.60x** | 27.77 | 27.77 |
| 128×1024 | 16384 | 0.2742 | 0.1867 | **0.68x** | 15.66 | 15.66 |
| 128×1024 | 4096 | 0.2505 | 0.1824 | **0.73x** | 4.29 | 4.29 |

### 2.3 W8A16 vs W4A16 对比分析

#### Speedup 对比

| 分类 | W8A16 Speedup 范围 | W4A16 Speedup 范围 | 结论 |
|------|-------------------|-------------------|------|
| Down projection (N=1024) | 0.75x ~ 1.03x | 0.64x ~ 1.04x | M≥4096 时两者均接近 1.0x；M≤2048 时均减速 |
| Up projection (N=3584) | 0.96x ~ 1.01x | 0.82x ~ 1.01x | W4A16 在 N=3584, M=2048 减速更明显 |
| Gate projection (N=1024) | 0.98x ~ 1.01x | 1.00x ~ 1.01x | Gate 投影两者均稳定在 1.0x |
| Router (N=128) | 0.58x ~ 0.67x | 0.60x ~ 0.73x | N 极小时两者均大幅减速；W4A16 略优 |

#### 关键发现

**大矩阵（M≥4096）表现良好：**
- Down projection M=32k: W8A16 1.03x，W4A16 1.03x，均略超 FP16
- Gate projection M=32k: W8A16 0.98x，W4A16 1.01x，W4A16 略优
- TFLOPS 峰值达 93.84（Down projection），接近 H20 Tensor Core 理论性能

**小矩阵（M≤2048）表现较差：**
- Down projection M=1024: W8A16 0.77x，W4A16 0.64x
- 原因：量化/反量化开销在计算量小时占比过高
- N=128 Router 层：减速最严重（0.58x~0.73x），因 GEMM 计算量极小

**W4A16 vs W8A16 结论：**
- 大矩阵：两者性能接近，W4A16 权重内存节省 50%（vs FP16），W8A16 节省 50%
- 小矩阵：W8A16 反量化开销更小（无 nibble unpack），略优于 W4A16
- 总体：差异不显著，在可接受范围内

---

## 3. 技术实现分析

### 3.1 Kernel 调用链

```
QCGEMBenchmark → core.py: get_matmul_type()
  ├─ GEMV (M < threshold): gemv / gemv_splitK / gemv_revsplitK
  └─ GEMM (M ≥ threshold): gemm / gemm_splitK / gemm_splitK_persistent
```

### 3.2 计算流程

```
x @ W^T  — quantized GEMM
  x ∈ (M, K)      激活 (FP16)
  W ∈ (N, K)      权重 (W8A16 / W4A16)
  ↓ on-the-fly dequant
  y = x @ W_deq^T ∈ (M, N)
```

### 3.3 W4A16 nibble pack/unpack

```python
# 量化：2 nibble → 1 byte
W_q = (W_lo & 0xF) | (W_hi << 4)   # (N, K//2)

# 反量化 kernel 内：
lo = (W_u8 & 0x0F).float()         # 低 4bit
hi = ((W_u8 >> 4) & 0x0F).float()  # 高 4bit
W_deq = ((lo + hi) - Z) * S          # per-group 反量化
```

### 3.4 参考实现（PyTorch FP16）

```python
y_ref = torch.mm(x.half(), w.half())   # PyTorch FP16 GEMM
```

---

## 4. FLOPs 计算公式

```
total_flops = M × N × K × 2    # 一次 GEMM
TFLOPS = total_flops / latency_ms / 1e6
```

---

## 5. 复现指南

### 5.1 环境依赖

```bash
# 基础环境 (FlagGems conda env)
conda activate FlagGems
pip install torch triton flag_gems pytest numpy matplotlib pandas python-docx
```

### 5.2 运行测试

```bash
cd /data/nfs3/shared_workspace_mixq/zhiyuan/FlagEnv/FlagGems

# 运行 W8A16 benchmark
srun --partition=long --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  --time=12:00:00 --pty bash -c \
  "conda activate FlagGems && python benchmark/run_qcgem_benchmark.py --mode w8a16 --dtype float16"

# 运行 W4A16 benchmark
srun --partition=long --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  --time=12:00:00 --pty bash -c \
  "conda activate FlagGems && python benchmark/run_qcgem_benchmark.py --mode w4a16 --dtype float16"

# 生成 W8A16 图表
python ./benchmark/qcgem_analysis/plot_results_w8a16.py

# 生成 W4A16 图表
python ./benchmark/qcgem_analysis/plot_results_w4a16.py

# 生成 W8A16 + W4A16 并排对比图表
python ./benchmark/qcgem_analysis/plot_results.py

# 导出汇总表（Markdown + Word）
python ./benchmark/qcgem_analysis/export_qcgem_summary_table.py
```

### 5.3 独立脚本说明

| 文件 | 说明 |
|------|------|
| `run_qcgem_benchmark.py` | benchmark 入口，支持 `--mode w8a16/w4a16 --dtype float16` |
| `plot_results_w8a16.py` | W8A16 专用绘图脚本（深色主题，4-panel） |
| `plot_results_w4a16.py` | W4A16 专用绘图脚本（深色主题，4-panel） |
| `plot_results.py` | W8A16 + W4A16 并排对比绘图脚本 |
| `export_qcgem_summary_table.py` | 汇总表导出脚本（Markdown + Word） |
| `benchmark_w8a16_fp16.csv` | W8A16 原始数据 |
| `benchmark_w4a16_fp16.csv` | W4A16 原始数据 |

---

## 6. 已知限制

1. **小矩阵减速**：M≤2048 时量化路径均慢于 FP16 参考，主要因反量化开销相对较大。建议在 M≥4096 时启用量化路径。

2. **Router 层（N=128）减速明显**：N=128 时 GEMM 计算量极小（128×K），反量化开销成为主导因素。建议 Router 层使用 FP16。

3. **仅单卡测试**：当前 benchmark 在单卡 H20 上运行，未测试多卡场景。

4. **无精度验证**：benchmark 使用随机权重，未验证量化后数值精度。建议使用固定种子权重做端到端验证。

---

## 7. 完整性能分析报告

详细分析见：

| 文件 | 描述 |
|------|------|
| [`benchmark_analysis.md`](./benchmark_analysis.md) | **完整性能分析报告** |
| [`benchmark_analysis.docx`](./benchmark_analysis.docx) | Word 版本，可编辑 |
| [`qcgem_summary_table.md`](./qcgem_summary_table.md) | 汇总表（对齐 QC-MoE 样式） |
| [`qcgem_summary_table.docx`](./qcgem_summary_table.docx) | Word 版本汇总表 |

### 7.1 核心结论

1. **大矩阵量化优势明显**：M≥4096 时，Down/Gate 投影均可达到 1.0x~1.04x，与 FP16 持平，同时节省 50%（W8A16）或 75%（W4A16）权重内存。

2. **小矩阵量化劣势**：M≤2048 时，反量化开销主导，导致 0.64x~0.96x 减速。建议小矩阵场景回退到 FP16。

3. **Router 层避免量化**：N=128 时无论 W8A16 或 W4A16 均大幅减速（0.58x~0.73x），建议 Router 层保持 FP16。

4. **W8A16 vs W4A16 差异小**：两者在大多数 shape 上表现接近，W4A16 内存节省更多（-75% vs -50%），推荐大矩阵使用 W4A16。

---

## 8. 文件清单

```
qcgem_analysis/
├── README.md                               ← 本文档
├── benchmark_analysis.md                   ← **完整性能分析报告**
├── benchmark_analysis.docx                 ← Word 版本
├── qcgem_summary_table.md                  ← 汇总表（对齐 QC-MoE 样式）
├── qcgem_summary_table.docx                ← Word 版本
├── qc_gem_vs_pytorch_fp16_benchmark_summary.md  ← W8A16/W4A16 汇总表
├── qc_gem_vs_pytorch_fp16_benchmark_summary.docx ← Word 版本
├── qcgem_complete_report.md                ← **完整测试报告**
├── qcgem_complete_report.docx              ← Word 版本
├── plot_results_w8a16.py                  ← W8A16 绘图脚本（深色主题）
├── plot_results_w4a16.py                  ← W4A16 绘图脚本（深色主题）
├── plot_results.py                         ← W8A16 + W4A16 并排绘图脚本
├── export_qcgem_summary_table.py           ← 汇总表导出脚本
├── export_qcgem_terminal_tables.py          ← 终端表格导出脚本
├── generate_docx.py                         ← Word 报告生成脚本
├── qcgem_w8a16_data.csv                   ← W8A16 分类数据
├── qcgem_w4a16_data.csv                   ← W4A16 分类数据
├── benchmark_w8a16_fp16.csv               ← W8A16 原始数据
├── benchmark_w4a16_fp16.csv               ← W4A16 原始数据
├── qcgem_w8a16_benchmark.png              ← W8A16 图表 (PNG)
├── qcgem_w8a16_benchmark.pdf              ← W8A16 图表 (PDF)
├── qcgem_w4a16_benchmark.png              ← W4A16 图表 (PNG)
├── qcgem_w4a16_benchmark.pdf              ← W4A16 图表 (PDF)
├── qcgem_benchmark.png                     ← W8A16 + W4A16 对比图表 (PNG)
└── qcgem_benchmark.pdf                    ← W8A16 + W4A16 对比图表 (PDF)
```
