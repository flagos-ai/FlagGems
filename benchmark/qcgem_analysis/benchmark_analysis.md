# QC-GEM Benchmark Analysis Report

**Author:** QC-GEM Analysis Script  
**Date:** 2026-03-27  
**Test Environment:** Qwen3.5-397B-A17B MoE Shapes

---

## 1. Executive Summary

This report presents the performance analysis of QC-GEM kernel implementation compared to PyTorch baseline for quantization-aware GEMM operations. Two quantization configurations were evaluated: **w4A16** (4-bit weight, 16-bit activation) and **w8A16** (8-bit weight, 16-bit activation).

### Key Findings

| Metric | w4A16 | w8A16 | Difference |
|--------|-------|-------|------------|
| **Average Speedup** | 0.895x | 0.915x | -0.020 |
| **Max Speedup** | 1.041x | 1.026x | +0.015 |
| **Min Speedup** | 0.603x | 0.585x | +0.018 |
| **Average TFLOPS** | 59.27 | 55.15 | +4.11 |
| **Max TFLOPS** | 93.84 | 93.84 | 0.00 |

**Conclusion:** w8A16 achieves slightly higher average speedup (+2%) compared to w4A16, but w4A16 achieves higher peak performance on large shapes. Both quantization modes show similar performance characteristics with significant variation based on problem size.

---

## 2. Performance Analysis

### 2.1 Large Shapes (M ≥ 4096)

| Metric | w4A16 | w8A16 |
|--------|-------|-------|
| Average Speedup | 0.945x | 0.933x |
| Average TFLOPS | 70.15 | 65.94 |

**Analysis:** For large batch/sequence sizes, both modes approach baseline performance. w4A16 shows slightly better average speedup (+1.3%) and higher TFLOPS throughput.

### 2.2 Small Shapes (M < 4096)

| Metric | w4A16 | w8A16 |
|--------|-------|-------|
| Average Speedup | 0.768x | 0.870x |
| Average TFLOPS | 32.04 | 28.18 |

**Analysis:** Small shapes favor w8A16 with +13% better speedup, though w4A16 maintains higher TFLOPS. Both modes show significant overhead due to quantization/dequantization dominating compute time.

---

## 3. Shape-Specific Performance

### 3.1 FFN Layer Shapes (N=1024, K=3584)

| Shape (M) | w4A16 Speedup | w8A16 Speedup | w4A16 TFLOPS | w8A16 TFLOPS |
|-----------|---------------|---------------|--------------|--------------|
| 32k | 1.032x | 1.026x | 93.84 | 93.84 |
| 16k | 1.041x | 1.009x | 89.65 | 86.25 |
| 8k | 1.009x | 1.008x | 80.11 | 74.07 |
| 4k | 1.017x | 1.012x | 68.07 | 59.51 |
| 2k | 0.836x | 0.962x | 44.48 | 40.54 |
| 1k | 0.641x | 0.768x | 23.12 | 20.73 |
| 512 | 0.643x | 0.746x | 13.03 | 11.80 |
| 256 | 0.661x | 0.769x | 6.69 | 6.53 |

**Analysis:** Large FFN shapes (M≥4096) show consistent 1.0-1.04x speedup for both modes. w4A16 outperforms w8A16 on large shapes with up to +5% higher TFLOPS.

### 3.2 FFN Layer Shapes (N=3584, K=1024)

| Shape (M) | w4A16 Speedup | w8A16 Speedup | w4A16 TFLOPS | w8A16 TFLOPS |
|-----------|---------------|---------------|--------------|--------------|
| 32k | 1.011x | 1.010x | 89.77 | 89.61 |
| 16k | 1.004x | 1.004x | 89.32 | 86.49 |
| 8k | 1.010x | 1.007x | 82.28 | 76.42 |
| 4k | 1.015x | 1.010x | 71.28 | 61.95 |
| 2k | 0.816x | 0.963x | 44.80 | 41.88 |

### 3.3 FFN Layer Shapes (N=7168, K=1024)

| Shape (M) | w4A16 Speedup | w8A16 Speedup | w4A16 TFLOPS | w8A16 TFLOPS |
|-----------|---------------|---------------|--------------|--------------|
| 32k | 1.014x | 0.981x | 90.95 | 87.48 |
| 16k | 1.003x | 1.000x | 90.65 | 87.76 |
| 8k | 1.004x | 1.003x | 84.33 | 78.17 |
| 4k | 1.008x | 1.005x | 74.33 | 64.40 |
| 2k | 1.010x | 1.009x | 60.16 | 47.62 |

**Analysis:** w4A16 consistently shows higher TFLOPS across all N=7168 shapes, with up to +26% higher throughput at M=2k.

### 3.4 Router Layer Shapes (N=128)

| Shape (M) | w4A16 Speedup | w8A16 Speedup | w4A16 TFLOPS | w8A16 TFLOPS |
|-----------|---------------|---------------|--------------|--------------|
| 32k | 0.603x | 0.585x | 27.77 | 25.13 |
| 16k | 0.681x | 0.669x | 15.66 | 14.36 |
| 4k | 0.728x | 0.669x | 4.29 | 3.70 |

**Analysis:** Router layer shapes show significant slowdown for both modes. w4A16 slightly outperforms w8A16 on N=128 shapes with ~10% higher TFLOPS.

---

## 4. Performance Characteristics

### 4.1 Speedup by Problem Size

```
M (Batch*Seq) →  [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

N=1024, K=3584: 0.66x → 0.64x → 0.64x → 0.84x → 1.02x → 1.01x → 1.04x → 1.03x  (w4A16)
                0.77x → 0.75x → 0.77x → 0.96x → 1.01x → 1.01x → 1.01x → 1.03x  (w8A16)

N=7168, K=1024:  N/A  →  N/A  →  N/A  → 1.01x → 1.01x → 1.00x → 1.00x → 1.01x  (w4A16)
                N/A  →  N/A  →  N/A  → 1.01x → 1.01x → 1.00x → 1.00x → 0.98x  (w8A16)

N=128,  K=1024:  N/A  →  N/A  →  N/A  → 0.73x → 0.73x →  N/A  → 0.68x → 0.60x  (w4A16)
                N/A  →  N/A  →  N/A  → 0.67x → 0.67x →  N/A  → 0.67x → 0.59x  (w8A16)
```

**Key Observations:**
- Performance crossover point occurs around M=2048-4096 for both modes
- w8A16 shows better speedup on small shapes (M≤1024)
- w4A16 shows better speedup on large shapes (M≥4096)
- Small N (128) consistently underperforms regardless of mode

### 4.2 TFLOPS Analysis

| Shape Category | w4A16 Max | w8A16 Max | w4A16 Min | w8A16 Min |
|----------------|-----------|-----------|-----------|-----------|
| N=1024 | 93.84 | 93.84 | 6.69 | 6.53 |
| N=3584 | 89.77 | 89.61 | 44.80 | 41.88 |
| N=7168 | 90.95 | 87.48 | 60.16 | 47.62 |
| N=128 | 27.77 | 25.13 | 4.29 | 3.70 |

**Key Observations:**
- Peak throughput (93.84 TFLOPS) achieved with both modes on N=1024, K=3584 shapes
- w4A16 maintains higher TFLOPS across all shape categories
- Small N severely limits computational intensity for both modes

---

## 5. Mode Selection Recommendations

### 5.1 When to Use w4A16

- Large batch sizes (M ≥ 4096) with FFN layers
- Maximum throughput requirements
- Memory bandwidth limited scenarios (higher compression ratio)
- FFN layer computations with N ≥ 1024

### 5.2 When to Use w8A16

- Small batch sizes (M < 4096)
- Router layer computations
- Cases requiring better small-shape performance
- When weight quantization precision matters for accuracy

### 5.3 When to Fall Back to PyTorch

- Router layer computations (N = 128)
- Very small shapes (M < 512) where overhead dominates
- Low-latency requirements with small shapes
- Cases where quantization overhead exceeds compute savings

---

## 6. Optimization Opportunities

1. **Shape-Aware Kernel Selection:** Implement dynamic dispatch based on problem size heuristics
2. **N=128 Specialization:** Develop dedicated kernel for router layer with different block sizes
3. **Quantization/Compute Overlap:** Overlap dequantization with GEMM for small shapes
4. **Mixed-Mode Kernel:** Fuse w4A16 and w8A16 paths with minimal overhead switching

---

## 7. Appendix

### A. Test Configuration

| Parameter | Value |
|-----------|-------|
| Mode | kernel |
| Dtype | float16 |
| Framework | FlagGems |
| Test Shapes | Qwen3.5-397B-A17B |
| Quantization | w4A16, w8A16 |

### B. File List

| File | Description |
|------|-------------|
| `benchmark_w4a16_fp16.csv` | w4A16 raw benchmark data |
| `benchmark_w8a16_fp16.csv` | w8A16 raw benchmark data |
| `benchmark_w4a16_fp16.png/pdf` | w4A16 visualization |
| `benchmark_w8a16_fp16.png/pdf` | w8A16 visualization |
| `benchmark_comparison.png/pdf` | w4A16 vs w8A16 comparison |
| `plot_all.py` | Unified plotting script |
| `benchmark_analysis.md` | This analysis report |
| `benchmark_analysis.docx` | Word format report |

---

*Report generated automatically by QC-GEM Analysis Pipeline*
