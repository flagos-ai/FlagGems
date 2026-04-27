---
title: Experimental Operators
weight: 20
---

# FlagGems Experimental Operators

This document lists all experimental operators in FlagGems that have achieved an average speedup of 0.8x or higher compared to PyTorch implementations.

## Performance Overview

- **Total Operators**: 142
- **Average Speedup Range**: 0.81x - 7.23x
- **Test Environment**: Hopper GPU
- **Filtering Criteria**: Average speedup ≥ 0.8x

## Operators by Performance

| Rank | Operator                      | Avg Speedup    | Category   |
|------|-------------------------------|----------------|------------|
|   1 | `_safe_softmax`               | **7.23x** 🏆  | Internal   |
|   2 | `digamma_`                    | **2.41x** 🏆  | Math       |
|   3 | `zero`                        | **1.85x** ✅  | Other      |
|   4 | `relu`                        | **1.79x** ✅  | Activation |
|   5 | `mse_loss`                    | **1.64x** ✅  | Loss       |
|   6 | `masked_select`               | **1.47x** ✅  | Other      |
|   7 | `masked_scatter`              | **1.44x** ✅  | Other      |
|   8 | `eye`                         | **1.43x** ✅  | Other      |
|   9 | `t_copy`                      | **1.41x** ✅  | Shape      |
|  10 | `trace`                       | **1.40x** ✅  | Math       |
|  11 | `i0_`                         | **1.37x** ✅  | Math       |
|  12 | `zeros_like`                  | **1.32x** ✅  | Other      |
|  13 | `diag`                        | **1.27x** ✅  | Other      |
|  14 | `lift_fresh_copy`             | **1.24x** ✅  | Other      |
|  15 | `alias_copy`                  | **1.23x** ✅  | Other      |
|  16 | `pixel_unshuffle`             | **1.20x** 📈  | Vision     |
|  17 | `triu`                        | **1.18x** 📈  | Shape      |
|  18 | `rrelu_with_noise_backward`   | **1.17x** 📈  | Activation |
|  19 | `glu`                         | **1.17x** 📈  | Activation |
|  20 | `tril`                        | **1.16x** 📈  | Shape      |
|  21 | `silu_`                       | **1.16x** 📈  | Activation |
|  22 | `asinh_`                      | **1.14x** 📈  | Math       |
|  23 | `mv`                          | **1.14x** 📈  | Linear Algebra |
|  24 | `arcsinh_`                    | **1.13x** 📈  | Math       |
|  25 | `pixel_shuffle`               | **1.12x** 📈  | Vision     |
|  26 | `replication_pad3d`           | **1.11x** 📈  | Padding    |
|  27 | `_upsample_nearest_exact1d`   | **1.11x** 📈  | Vision     |
|  28 | `i0`                          | **1.11x** 📈  | Math       |
|  29 | `softplus`                    | **1.10x** 📈  | Activation |
|  30 | `selu_`                       | **1.10x** 📈  | Activation |
|  31 | `upsample_nearest1d`          | **1.10x** 📈  | Vision     |
|  32 | `special_i1`                  | **1.09x** 📈  | Math       |
|  33 | `selu`                        | **1.09x** 📈  | Activation |
|  34 | `amin`                        | **1.09x** 📈  | Math       |
|  35 | `sinh_`                       | **1.09x** 📈  | Math       |
|  36 | `logit_`                      | **1.08x** 📈  | Math       |
|  37 | `upsample_nearest3d`          | **1.07x** 📈  | Vision     |
|  38 | `im2col`                      | **1.06x** 📈  | Vision     |
|  39 | `reflection_pad1d`            | **1.06x** 📈  | Padding    |
|  40 | `elu`                         | **1.06x** 📈  | Activation |
|  41 | `arctanh_`                    | **1.05x** 📈  | Math       |
|  42 | `sigmoid`                     | **1.05x** 📈  | Activation |
|  43 | `replication_pad1d`           | **1.04x** 📈  | Padding    |
|  44 | `silu`                        | **1.04x** 📈  | Activation |
|  45 | `sigmoid_`                    | **1.04x** 📈  | Activation |
|  46 | `addcdiv`                     | **1.04x** 📈  | Arithmetic |
|  47 | `sinc_`                       | **1.03x** 📈  | Math       |
|  48 | `relu6`                       | **1.03x** 📈  | Activation |
|  49 | `hardtanh`                    | **1.03x** 📈  | Activation |
|  50 | `hardtanh_`                   | **1.03x** 📈  | Activation |
|  51 | `hardswish_`                  | **1.03x** 📈  | Activation |
|  52 | `reciprocal_`                 | **1.03x** 📈  | Math       |
|  53 | `sinc`                        | **1.03x** 📈  | Math       |
|  54 | `hardsigmoid`                 | **1.03x** 📈  | Activation |
|  55 | `logaddexp2`                  | **1.02x** 📈  | Math       |
|  56 | `logit`                       | **1.02x** 📈  | Math       |
|  57 | `arctanh`                     | **1.02x** 📈  | Math       |
|  58 | `logaddexp`                   | **1.02x** 📈  | Math       |
|  59 | `cosh_`                       | **1.02x** 📈  | Math       |
|  60 | `special_xlog1py`             | **1.02x** 📈  | Math       |
|  61 | `celu`                        | **1.02x** 📈  | Activation |
|  62 | `hardsigmoid_`                | **1.02x** 📈  | Activation |
|  63 | `arcsinh`                     | **1.02x** 📈  | Math       |
|  64 | `sign`                        | **1.02x** 📈  | Math       |
|  65 | `absolute_`                   | **1.01x** 📈  | Math       |
|  66 | `_adaptive_avg_pool3d`        | **1.01x** 📈  | Vision     |
|  67 | `special_i0e`                 | **1.01x** 📈  | Math       |
|  68 | `cos_`                        | **1.01x** 📈  | Math       |
|  69 | `deg2rad_`                    | **1.01x** 📈  | Math       |
|  70 | `floor_`                      | **1.01x** 📈  | Math       |
|  71 | `negative`                    | **1.01x** 📈  | Math       |
|  72 | `xlogy`                       | **1.01x** 📈  | Math       |
|  73 | `exp2`                        | **1.01x** 📈  | Math       |
|  74 | `exp_`                        | **1.00x** 📈  | Math       |
|  75 | `fix`                         | **1.00x** 📈  | Math       |
|  76 | `xlogy_`                      | **1.00x** 📈  | Math       |
|  77 | `absolute`                    | **1.00x** 📈  | Math       |
|  78 | `prelu`                       | **1.00x** 📈  | Activation |
|  79 | `hypot`                       | **1.00x** 📈  | Math       |
|  80 | `rad2deg_`                    | **1.00x** 📈  | Math       |
|  81 | `smooth_l1_loss`              | **1.00x** 📈  | Loss       |
|  82 | `deg2rad`                     | **1.00x** 📈  | Math       |
|  83 | `log_`                        | **1.00x** 📈  | Math       |
|  84 | `sgn_`                        | **1.00x** 📈  | Math       |
|  85 | `sin_`                        | **1.00x** 📈  | Math       |
|  86 | `heaviside`                   | **1.00x** 📈  | Math       |
|  87 | `logical_xor_`                | **1.00x** 📈  | Other      |
|  88 | `trunc`                       | **1.00x** 📈  | Math       |
|  89 | `heaviside_`                  | **1.00x** 📈  | Math       |
|  90 | `hardshrink`                  | **1.00x** 📈  | Activation |
|  91 | `huber_loss`                  | **1.00x** 📈  | Loss       |
|  92 | `threshold_`                  | **1.00x** 📈  | Activation |
|  93 | `addcmul_`                    | **1.00x** 📈  | Arithmetic |
|  94 | `neg_`                        | **1.00x** 📈  | Math       |
|  95 | `hypot_`                      | **1.00x** 📈  | Math       |
|  96 | `leaky_relu`                  | **1.00x** 📈  | Activation |
|  97 | `fmin`                        | **1.00x** 📈  | Math       |
|  98 | `erfinv`                      | **1.00x** 📈  | Math       |
|  99 | `log1p_`                      | **1.00x** 📈  | Math       |
| 100 | `frac`                        | **1.00x** ⚡  | Math       |
| 101 | `_functional_sym_constrain_range_for_size` | **1.00x** ⚡  | Internal   |
| 102 | `expand`                      | **1.00x** ⚡  | Shape      |
| 103 | `lift`                        | **1.00x** ⚡  | Other      |
| 104 | `unsqueeze`                   | **1.00x** ⚡  | Shape      |
| 105 | `_unsafe_view`                | **1.00x** ⚡  | Internal   |
| 106 | `softshrink`                  | **1.00x** ⚡  | Activation |
| 107 | `log2_`                       | **1.00x** ⚡  | Math       |
| 108 | `permute`                     | **1.00x** ⚡  | Shape      |
| 109 | `leaky_relu_`                 | **1.00x** ⚡  | Activation |
| 110 | `atanh_`                      | **1.00x** ⚡  | Math       |
| 111 | `permute_copy`                | **1.00x** ⚡  | Shape      |
| 112 | `fft_ifftshift`               | **1.00x** ⚡  | Other      |
| 113 | `copy_`                       | **1.00x** ⚡  | Other      |
| 114 | `fix_`                        | **1.00x** ⚡  | Math       |
| 115 | `slice_scatter`               | **0.99x** ⚡  | Other      |
| 116 | `exp2_`                       | **0.99x** ⚡  | Math       |
| 117 | `rsqrt_`                      | **0.99x** ⚡  | Math       |
| 118 | `threshold`                   | **0.98x** ⚡  | Activation |
| 119 | `reciprocal`                  | **0.97x** ⚡  | Math       |
| 120 | `maximum`                     | **0.97x** ⚡  | Arithmetic |
| 121 | `abs`                         | **0.96x** ⚡  | Math       |
| 122 | `arccosh`                     | **0.96x** ⚡  | Math       |
| 123 | `multiply`                    | **0.95x** ⚡  | Arithmetic |
| 124 | `margin_ranking_loss`         | **0.95x** ⚡  | Loss       |
| 125 | `celu_`                       | **0.92x** ⚡  | Activation |
| 126 | `hardswish`                   | **0.91x** ⚡  | Activation |
| 127 | `soft_margin_loss`            | **0.90x** ⚡  | Loss       |
| 128 | `replication_pad2d`           | **0.90x** ⚡  | Padding    |
| 129 | `unsqueeze_copy`              | **0.89x** ⚡  | Shape      |
| 130 | `native_dropout_backward`     | **0.89x** ⚡  | Other      |
| 131 | `slice_backward`              | **0.88x** ⚡  | Other      |
| 132 | `relu_`                       | **0.86x** ⚡  | Activation |
| 133 | `negative_`                   | **0.86x** ⚡  | Math       |
| 134 | `abs_`                        | **0.86x** ⚡  | Math       |
| 135 | `take`                        | **0.86x** ⚡  | Other      |
| 136 | `sgn`                         | **0.86x** ⚡  | Math       |
| 137 | `erf_`                        | **0.82x** ⚡  | Math       |
| 138 | `gelu_`                       | **0.82x** ⚡  | Activation |
| 139 | `erfinv_`                     | **0.82x** ⚡  | Math       |
| 140 | `_log_softmax_backward_data`  | **0.82x** ⚡  | Internal   |
| 141 | `log10_`                      | **0.81x** ⚡  | Math       |
| 142 | `rmsnorm`                     | **special** ⚡ | Normalization |

**Legend**:

- 🏆 **Outstanding**: Speedup ≥ 2.0x
- ✅ **Excellent**: Speedup ≥ 1.5x
- 📈 **Good**: Speedup ≥ 1.0x
- ⚡ **Decent**: Speedup ≥ 0.8x

## Categories

- **Activation**: Activation functions (ReLU, GELU, Sigmoid, etc.)
- **Arithmetic**: Basic arithmetic operations (add, mul, div, etc.)
- **Comparison**: Comparison operations (eq, ne, gt, lt, etc.)
- **Internal**: Internal/utility operations
- **Linear Algebra**: Matrix operations (matmul, mv, etc.)
- **Loss**: Loss functions (MSE, Cross-Entropy, etc.)
- **Math**: Mathematical functions (sin, cos, exp, log, etc.)
- **NLP**: Natural language processing operations
- **Other**: Miscellaneous operations
- **Padding**: Padding operations (reflection_pad, replication_pad, etc.)
- **Shape**: Shape manipulation operations
- **Vision**: Computer vision operations

## Notes

- All operators have passed accuracy tests
- Performance measured on Hopper GPU with various input shapes
- Speedup calculated as: `PyTorch_time / FlagGems_time`
- Higher values indicate better performance
