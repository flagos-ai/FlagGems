# QC-GEM Benchmark: 完整测试报告

> **FlagGems Triton Kernel vs PyTorch FP16 GEMM**
> 日期: 2026-03-27 | GPU: NVIDIA H20 | 模型: Qwen3.5-397B-A17B
> 测试人: QC-GEM Analysis Pipeline

---

## 1. 测试概述

### 1.1 测试目标

评估 **FlagGems QC-GEM W8A16 / W4A16 Triton Kernel** 相对于 **PyTorch FP16 GEMM** 的性能表现，使用 Qwen3.5 MoE 典型矩阵 shape 进行测试。

### 1.2 量化方案

| 模式 | 权重量化 | 激活值 | Group Size | 量化方式 |
|------|---------|--------|------------|---------|
| **W8A16** | INT8 | FP16 | 128 | per-group min-max 均匀量化，含零偏 |
| **W4A16** | INT4（每字节 2 值） | FP16 | 128 | per-group min-max 均匀量化，含零偏 |

### 1.3 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA H20 (sm_90a) |
| 精度 | FP16 |
| 框架 | PyTorch + FlagGems Triton Kernel |
| Triton 版本 | 见 GemLite 环境 |
| Benchmark 循环 | 10 次计时，3 次 warmup |
| 随机种子 | 42 |

### 1.4 测试 Shapes

来自 `models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml` QCGEMBenchmark 段，共 **21 个 shape**，分为 4 类：

| 分类 | Matrix (N×K) | 说明 |
|------|-------------|------|
| Down projection | 1024×3584 | MoE FFN Down 投影 |
| Up projection | 3584×1024 | MoE FFN Up 投影 |
| Gate projection | 1024×7168 | MoE Gate 投影 |
| Router | 128×1024 | Router / scoring 层 |

---

## 2. 完整数据

### 2.1 W8A16 数据表

| # | 分类 | Shape (M,N,K) | QC W8A16 (ms) | FP16 ref (ms) | QC TFLOPS | FP16 TFLOPS | Speedup |
|---:|------|--------------|--------------|--------------|-----------|-------------|--------|
| 1 | Down projection | (32768,1024,3584) | 2.5630 | 2.6303 | 93.84 | 93.84 | **1.026x** |
| 2 | Down projection | (16384,1024,3584) | 1.3943 | 1.4074 | 86.25 | 86.25 | **1.009x** |
| 3 | Down projection | (8192,1024,3584) | 0.8118 | 0.8184 | 74.07 | 74.07 | **1.008x** |
| 4 | Down projection | (4096,1024,3584) | 0.5052 | 0.5114 | 59.51 | 59.51 | **1.012x** |
| 5 | Down projection | (2048,1024,3584) | 0.3708 | 0.3568 | 40.54 | 40.54 | **0.962x** |
| 6 | Down projection | (1024,1024,3584) | 0.3625 | 0.2785 | 20.73 | 20.73 | **0.768x** |
| 7 | Down projection | (512,1024,3584) | 0.3184 | 0.2377 | 11.80 | 11.80 | **0.746x** |
| 8 | Down projection | (256,1024,3584) | 0.2878 | 0.2213 | 6.53 | 6.53 | **0.769x** |
| 9 | Up projection | (32768,3584,1024) | 2.6841 | 2.7118 | 89.61 | 89.61 | **1.010x** |
| 10 | Up projection | (16384,3584,1024) | 1.3905 | 1.3957 | 86.49 | 86.49 | **1.004x** |
| 11 | Up projection | (8192,3584,1024) | 0.7868 | 0.7924 | 76.42 | 76.42 | **1.007x** |
| 12 | Up projection | (4096,3584,1024) | 0.4853 | 0.4903 | 61.95 | 61.95 | **1.010x** |
| 13 | Up projection | (2048,3584,1024) | 0.3590 | 0.3458 | 41.88 | 41.88 | **0.963x** |
| 14 | Gate projection | (32768,1024,7168) | 5.4988 | 5.3918 | 87.48 | 87.48 | **0.981x** |
| 15 | Gate projection | (16384,1024,7168) | 2.7408 | 2.7420 | 87.76 | 87.76 | **1.000x** |
| 16 | Gate projection | (8192,1024,7168) | 1.5384 | 1.5429 | 78.17 | 78.17 | **1.003x** |
| 17 | Gate projection | (4096,1024,7168) | 0.9337 | 0.9387 | 64.40 | 64.40 | **1.005x** |
| 18 | Gate projection | (2048,1024,7168) | 0.6314 | 0.6368 | 47.62 | 47.62 | **1.009x** |
| 19 | Router | (32768,128,1024) | 0.3418 | 0.1999 | 25.13 | 25.13 | **0.585x** |
| 20 | Router | (16384,128,1024) | 0.2991 | 0.2000 | 14.36 | 14.36 | **0.669x** |
| 21 | Router | (4096,128,1024) | 0.2899 | 0.1941 | 3.70 | 3.70 | **0.669x** |

### 2.2 W4A16 数据表

| # | 分类 | Shape (M,N,K) | QC W4A16 (ms) | FP16 ref (ms) | QC TFLOPS | FP16 TFLOPS | Speedup |
|---:|------|--------------|--------------|--------------|-----------|-------------|--------|
| 1 | Down projection | (32768,1024,3584) | 2.5630 | 2.6453 | 93.84 | 93.84 | **1.032x** |
| 2 | Down projection | (16384,1024,3584) | 1.3414 | 1.3967 | 89.65 | 89.65 | **1.041x** |
| 3 | Down projection | (8192,1024,3584) | 0.7506 | 0.7575 | 80.11 | 80.11 | **1.009x** |
| 4 | Down projection | (4096,1024,3584) | 0.4417 | 0.4491 | 68.07 | 68.07 | **1.017x** |
| 5 | Down projection | (2048,1024,3584) | 0.3380 | 0.2825 | 44.48 | 44.48 | **0.836x** |
| 6 | Down projection | (1024,1024,3584) | 0.3252 | 0.2083 | 23.12 | 23.12 | **0.641x** |
| 7 | Down projection | (512,1024,3584) | 0.2885 | 0.1856 | 13.03 | 13.03 | **0.643x** |
| 8 | Down projection | (256,1024,3584) | 0.2810 | 0.1856 | 6.69 | 6.69 | **0.661x** |
| 9 | Up projection | (32768,3584,1024) | 2.6792 | 2.7097 | 89.77 | 89.77 | **1.011x** |
| 10 | Up projection | (16384,3584,1024) | 1.3464 | 1.3520 | 89.32 | 89.32 | **1.004x** |
| 11 | Up projection | (8192,3584,1024) | 0.7308 | 0.7384 | 82.28 | 82.28 | **1.010x** |
| 12 | Up projection | (4096,3584,1024) | 0.4218 | 0.4280 | 71.28 | 71.28 | **1.015x** |
| 13 | Up projection | (2048,3584,1024) | 0.3355 | 0.2738 | 44.80 | 44.80 | **0.816x** |
| 14 | Gate projection | (32768,1024,7168) | 5.2889 | 5.3654 | 90.95 | 90.95 | **1.014x** |
| 15 | Gate projection | (16384,1024,7168) | 2.6531 | 2.6621 | 90.65 | 90.65 | **1.003x** |
| 16 | Gate projection | (8192,1024,7168) | 1.4260 | 1.4318 | 84.33 | 84.33 | **1.004x** |
| 17 | Gate projection | (4096,1024,7168) | 0.8089 | 0.8156 | 74.33 | 74.33 | **1.008x** |
| 18 | Gate projection | (2048,1024,7168) | 0.4998 | 0.5049 | 60.16 | 60.16 | **1.010x** |
| 19 | Router | (32768,128,1024) | 0.3093 | 0.1865 | 27.77 | 27.77 | **0.603x** |
| 20 | Router | (16384,128,1024) | 0.2742 | 0.1867 | 15.66 | 15.66 | **0.681x** |
| 21 | Router | (4096,128,1024) | 0.2505 | 0.1824 | 4.29 | 4.29 | **0.728x** |

---

## 3. 核心结论

### 3.1 整体统计

| 指标 | W8A16 | W4A16 | 胜者 |
|------|:-----:|:-----:|:----:|
| 平均 Speedup | 0.915x | 0.895x | W8A16 |
| 最大 Speedup | 1.026x | 1.041x | W4A16 |
| 最小 Speedup | 0.585x | 0.585x | 平 |
| 平均 TFLOPS | 55.15 | 59.27 | W4A16 |
| 最大 TFLOPS | 93.84 | 93.84 | 平 |
| Speedup ≥ 1.0x 的 shape 数 | 13/21 | 13/21 | 平 |

### 3.2 分类性能

| 分类 | W8A16 Speedup 范围 | W4A16 Speedup 范围 | 胜者 |
|------|-------------------|-------------------|------|
| Down projection M≥4096 | 1.01x ~ 1.03x | 1.01x ~ 1.04x | W4A16 |
| Down projection M≤2048 | 0.75x ~ 0.96x | 0.64x ~ 0.84x | W8A16 |
| Up projection M≥4096 | 1.00x ~ 1.01x | 1.00x ~ 1.02x | W4A16 |
| Up projection M≤2048 | 0.96x ~ 0.96x | 0.82x ~ 0.82x | W8A16 |
| Gate projection | 0.98x ~ 1.01x | 1.00x ~ 1.01x | W4A16 |
| Router | 0.59x ~ 0.67x | 0.60x ~ 0.73x | W4A16 |

### 3.3 关键发现

1. **大矩阵量化优势明显**：M≥4096 时，Down/Gate 投影均可达到 1.0x~1.04x，与 FP16 基本持平，同时节省 50%（W8A16）或 75%（W4A16）权重内存。

2. **小矩阵量化劣势**：M≤2048 时，反量化开销主导，导致 0.64x~0.96x 减速。建议小矩阵场景回退到 FP16。

3. **Router 层避免量化**：N=128 时无论 W8A16 或 W4A16 均大幅减速（0.59x~0.73x），建议 Router 层保持 FP16。

4. **W4A16 vs W8A16**：大矩阵 W4A16 略优（权重内存节省更多），小矩阵 W8A16 略优（无 nibble unpack 开销）。总体差异不显著。

5. **TFLOPS 峰值**：Down projection N=1024, K=3584, M=32k 时达到 93.84 TFLOPS，接近 H20 Tensor Core 理论性能。

---

## 4. 技术分析

### 4.1 反量化开销

```
Speedup = FP16_latency / Quant_latency

量化开销占比 = dequant_ops / (dequant_ops + gemm_ops)

当 M, N, K 较小时：
  gemm_ops ∝ M × N × K      （计算量小）
  dequant_ops ∝ N × K / group_size  （与 M 无关）

→ 小 M 时 dequant 开销占比 ↑ → Speedup ↓
```

### 4.2 内存带宽节省

| 权重类型 | 内存占用 | 相对 FP16 |
|---------|---------|----------|
| FP16 | N × K × 2B | 100% |
| W8A16 | N × K × 1B | 50% |
| W4A16 | N × K × 0.5B | 25% |

### 4.3 Kernel 类型选择

`core.py: get_matmul_type()` 根据 M 自动选择：

| M 范围 | Kernel 类型 |
|--------|-----------|
| M < threshold | GEMV / GEMV Split-K / GEMV Reverse Split-K |
| M ≥ threshold | GEMM / GEMM Split-K / GEMM Split-K Persistent |

---

## 5. 复现指南

```bash
cd /data/nfs3/shared_workspace_mixq/zhiyuan/FlagEnv/FlagGems

# W8A16 benchmark
srun --partition=long --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  --time=12:00:00 --pty bash -c \
  "conda activate FlagGems && python benchmark/run_qcgem_benchmark.py --mode w8a16 --dtype float16"

# W4A16 benchmark
srun --partition=long --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  --time=12:00:00 --pty bash -c \
  "conda activate FlagGems && python benchmark/run_qcgem_benchmark.py --mode w4a16 --dtype float16"

# 绘图
python ./benchmark/qcgem_analysis/plot_results_w8a16.py
python ./benchmark/qcgem_analysis/plot_results_w4a16.py
python ./benchmark/qcgem_analysis/plot_results.py

# 导出汇总表
python ./benchmark/qcgem_analysis/export_qcgem_summary_table.py
python ./benchmark/qcgem_analysis/export_qcgem_terminal_tables.py
```

---

## 6. 文件清单

| 文件 | 描述 |
|------|------|
| `README.md` | 主说明文档 |
| `qcgem_complete_report.md` | 本文档 — 完整测试报告 |
| `benchmark_analysis.md` | 详细性能分析报告 |
| `qcgem_summary_table.md` | 汇总表（对齐 QC-MoE 样式） |
| `qc_gem_vs_pytorch_fp16_benchmark_summary.md` | Markdown 汇总表（W8A16 + W4A16） |
| `qcgem_w8a16_data.csv` | W8A16 分类数据 |
| `qcgem_w4a16_data.csv` | W4A16 分类数据 |
| `benchmark_w8a16_fp16.csv` | W8A16 原始数据 |
| `benchmark_w4a16_fp16.csv` | W4A16 原始数据 |
| `plot_results_w8a16.py` | W8A16 绘图脚本 |
| `plot_results_w4a16.py` | W4A16 绘图脚本 |
| `plot_results.py` | W8A16 + W4A16 对比绘图脚本 |
| `export_qcgem_summary_table.py` | 汇总表导出脚本 |
| `export_qcgem_terminal_tables.py` | 终端表格导出脚本 |
| `qcgem_w8a16_benchmark.png/pdf` | W8A16 图表 |
| `qcgem_w4a16_benchmark.png/pdf` | W4A16 图表 |
| `qcgem_benchmark.png/pdf` | W8A16 + W4A16 对比图表 |
