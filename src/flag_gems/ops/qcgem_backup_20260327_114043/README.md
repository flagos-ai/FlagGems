#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QC-GEM: Quantized Computing GEM library for FlagGems
# Based on GemLite by Mobius Labs GmbH
# Author: FlagGems Team
# Version: 1.0.0

## Overview

QC-GEM (Quantized Computing GEM) is a high-performance quantized matrix multiplication library integrated into FlagGems. It provides optimized Triton kernels for quantized GEMM operations, supporting various quantization formats including:

- INT4/INT8 weight-only quantization
- FP8/FP4 mixed-precision computation
- MX-FP8/MX-FP4 (Microscaling) formats
- NVFP4 format

## Directory Structure

```
qcgem/
├── __init__.py          # Main package exports
├── dtypes.py            # Data type definitions
├── utils.py             # Utility functions
├── config.py            # Autotune configuration
├── bitpack.py           # Weight packing kernels
├── gemm_kernels.py      # Main GEMM kernels
└── core.py              # Core classes and functions
```

## Usage

### Basic GEMM

```python
from flag_gems.ops.qcgem import QCGeMLinear, DType

layer = QCGeMLinear(
    W_nbits=4,
    group_size=128,
    in_features=4096,
    out_features=4096,
    input_dtype=DType.FP16,
)
layer.pack(W_q, scales, zeros)
output = layer(input)
```

### Using qcgem_mm function

```python
from flag_gems.ops.qcgem import qcgem_mm

output = qcgem_mm(x, W_q, scales, zeros, W_nbits=4, group_size=128)
```

## API Reference

### QCGeMLinear

Main linear layer class for quantized matrix multiplication.

Parameters:
- `W_nbits` (int): Weight quantization bits (1, 2, 4, 8, 16)
- `group_size` (int): Quantization group size
- `in_features` (int): Input feature dimension
- `out_features` (int): Output feature dimension
- `input_dtype` (DType): Input data type
- `output_dtype` (DType): Output data type
- `acc_dtype` (DType): Accumulator data type

### qcgem_mm

Function interface for quantized matrix multiplication.

Parameters:
- `x` (torch.Tensor): Input tensor
- `W_q` (torch.Tensor): Quantized weights
- `scales` (torch.Tensor): Quantization scales
- `zeros` (torch.Tensor): Quantization zeros
- `W_nbits` (int): Weight quantization bits
- `group_size` (int): Quantization group size
- `input_dtype` (DType): Input data type

## Performance

QC-GEM leverages Triton's automatic kernel tuning to optimize performance across different GPU architectures. The library automatically:

- Selects optimal block sizes based on matrix dimensions
- Optimizes memory access patterns
- Uses hardware-accelerated dot products

## Supported Data Types

| Type | Description |
|------|-------------|
| FP16 | Half precision float |
| BF16 | Bfloat16 |
| FP32 | Full precision float |
| INT8 | 8-bit integer |
| FP8 | 8-bit floating point (E4M3/E5M2) |
| MXFP8 | Microscaling FP8 |
| MXFP4 | Microscaling FP4 |
| NVFP4 | NVIDIA FP4 format |
