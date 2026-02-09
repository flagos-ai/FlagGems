# Copyright 2024 FlagGems Authors. All Rights Reserved.
import torch
import torch.nn.functional as F
import pytest
from src.flag_gems.ops.conv_transpose2d import conv_transpose2d 

SHAPES = [(1, 4, 8, 8), (4, 64, 32, 32), (2, 128, 64, 64)]
DTYPES = [torch.float32, torch.float16]
PARAMS = [
    {"stride": 1, "padding": 0, "dilation": 1},
    {"stride": 2, "padding": 1, "dilation": 1},
    {"stride": 2, "padding": 0, "dilation": 1},
]

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("params", PARAMS)
def test_conv_transpose2d_correctness(shape, dtype, params):
    N, C_in, H_in, W_in = shape
    stride, padding, dilation = params["stride"], params["padding"], params["dilation"]
    K, R, S = C_in, 3, 3
    input = torch.randn((N, C_in, H_in, W_in), device="cuda", dtype=dtype).to(memory_format=torch.channels_last)
    weight = torch.randn((C_in, C_in, R, S), device="cuda", dtype=dtype)
    bias = torch.randn((C_in,), device="cuda", dtype=dtype)

    out_pt = F.conv_transpose2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    out_triton = conv_transpose2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, allow_tf32=False)

    # Relaxed tolerance for accumulated GEMM errors in FP16/FP32
    tol = 1e-1 
    assert torch.allclose(out_triton, out_pt, atol=tol, rtol=1e-3)

@pytest.mark.parametrize("shape", [(1, 64, 32, 32), (2, 128, 64, 64), (8, 256, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_benchmark(shape, dtype):
    N, C, H, W = shape
    input = torch.randn((N, C, H, W), device="cuda", dtype=dtype).to(memory_format=torch.channels_last)
    weight = torch.randn((C, C, 3, 3), device="cuda", dtype=dtype)
    for _ in range(5): conv_transpose2d(input, weight, stride=2, padding=1)
    
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100): F.conv_transpose2d(input, weight, stride=2, padding=1)
    end.record()
    torch.cuda.synchronize()
    ms_pt = start.elapsed_time(end) / 100

    start.record()
    for _ in range(100): conv_transpose2d(input, weight, stride=2, padding=1, allow_tf32=True)
    end.record()
    torch.cuda.synchronize()
    ms_triton = start.elapsed_time(end) / 100
    print(f"\n[Bench] Shape {shape} {dtype} | PT: {ms_pt:.4f}ms | FlagGems: {ms_triton:.4f}ms | Speedup: {ms_pt/ms_triton:.2f}x")