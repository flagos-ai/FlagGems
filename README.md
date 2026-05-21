# Triton Log10 Operator

This project implements a Triton-accelerated log10 operator for FlagGems.

## Features

- GPU accelerated Triton kernel
- PyTorch-compatible API
- Numerical stability protection
- float32 + float16 support
- Comprehensive tests
- Performance benchmark included

## Tests

Covers:
- Small tensors
- Large tensors
- Different shapes
- Zero handling
- float16 validation

## Benchmark

Benchmarked against PyTorch CUDA implementation.

## Usage

from flag_gems.ops.log10 import log10

x = torch.rand(1024, device='cuda')

y = log10(x)