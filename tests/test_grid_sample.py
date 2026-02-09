# Copyright 2024 FlagGems Authors. All Rights Reserved.
import torch
import torch.nn.functional as F
import pytest
from src.flag_gems.ops.grid_sample import grid_sample 

SHAPES = [(1, 1, 2, 2, 2, 2), (4, 64, 64, 64, 64, 64), (1, 32, 512, 512, 512, 512)]
DTYPES = [torch.float32, torch.float16]
ALIGN_CORNERS = [True, False]

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", ALIGN_CORNERS)
def test_grid_sample_correctness(shape, dtype, align_corners):
    N, C, H_in, W_in, H_out, W_out = shape
    input = torch.randn((N, C, H_in, W_in), device="cuda", dtype=dtype)
    grid = torch.rand((N, H_out, W_out, 2), device="cuda", dtype=dtype) * 2 - 1
    
    out_pt = F.grid_sample(input, grid, align_corners=align_corners, mode='bilinear', padding_mode='zeros')
    out_triton = grid_sample(input, grid, align_corners=align_corners)
    
    if dtype == torch.float16:
        assert torch.allclose(out_triton, out_pt, atol=1e-3, rtol=1e-3)
    else:
        # Relaxed slightly for large coordinate math
        assert torch.allclose(out_triton, out_pt, atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("shape", [(1, 32, 256, 256, 256, 256), (8, 64, 128, 128, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_grid_sample_benchmark(shape, dtype):
    N, C, H_in, W_in, H_out, W_out = shape
    input = torch.randn((N, C, H_in, W_in), device="cuda", dtype=dtype)
    grid = torch.rand((N, H_out, W_out, 2), device="cuda", dtype=dtype) * 2 - 1
    for _ in range(5): grid_sample(input, grid)
    
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100): F.grid_sample(input, grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    end.record()
    torch.cuda.synchronize()
    ms_pt = start.elapsed_time(end) / 100
    
    start.record()
    for _ in range(100): grid_sample(input, grid, align_corners=False)
    end.record()
    torch.cuda.synchronize()
    ms_triton = start.elapsed_time(end) / 100
    print(f"\n[Bench] Shape {shape} {dtype} | PT: {ms_pt:.3f}ms | FlagGems: {ms_triton:.3f}ms | Speedup: {ms_pt/ms_triton:.2f}x")