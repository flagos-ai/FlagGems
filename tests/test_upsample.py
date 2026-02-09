# Copyright 2024 FlagGems Authors. All Rights Reserved.
import torch
import torch.nn.functional as F
import pytest
from src.flag_gems.ops.upsample_nearest2d import upsample_nearest2d

SHAPES = [(1, 1, 4, 4), (4, 64, 32, 32), (2, 128, 64, 64), (1, 3, 512, 512)]
SCALES = [(2.0, 2.0), (3.0, 3.0), (1.5, 1.5)]
DTYPES = [torch.float32, torch.float16]
LAYOUTS = [torch.contiguous_format, torch.channels_last]

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scales", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_upsample_correctness(shape, scales, dtype, layout):
    N, C, H_in, W_in = shape
    scale_h, scale_w = scales
    H_out, W_out = int(H_in * scale_h), int(W_in * scale_w)
    input = torch.randn(shape, device="cuda", dtype=dtype).to(memory_format=layout)
    input.requires_grad = True

    out_pt = F.interpolate(input, size=(H_out, W_out), mode='nearest')
    out_pt.sum().backward()
    grad_pt = input.grad.clone()
    input.grad.zero_()

    out_triton = upsample_nearest2d(input, (H_out, W_out))
    out_triton.sum().backward()
    grad_triton = input.grad.clone()

    tol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(out_triton, out_pt, atol=tol, rtol=tol)
    assert torch.allclose(grad_triton, grad_pt, atol=tol, rtol=tol)

@pytest.mark.parametrize("shape", [(16, 128, 128, 128)])
@pytest.mark.parametrize("layout", [torch.channels_last, torch.contiguous_format])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_upsample_benchmark(shape, layout, dtype):
    H_in, W_in = shape[2], shape[3]
    output_size = (H_in * 2, W_in * 2)
    input = torch.randn(shape, device="cuda", dtype=dtype).to(memory_format=layout)
    
    for _ in range(5): upsample_nearest2d(input, output_size)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100): F.interpolate(input, size=output_size, mode='nearest')
    end.record()
    torch.cuda.synchronize()
    ms_pt = start.elapsed_time(end) / 100

    start.record()
    for _ in range(100): upsample_nearest2d(input, output_size)
    end.record()
    torch.cuda.synchronize()
    ms_triton = start.elapsed_time(end) / 100
    
    layout_name = "NHWC" if layout == torch.channels_last else "NCHW"
    print(f"\n[Bench] Shape {shape} {dtype} Layout={layout_name} | PT: {ms_pt:.4f}ms | FlagGems: {ms_triton:.4f}ms | Speedup: {ms_pt/ms_triton:.2f}x")