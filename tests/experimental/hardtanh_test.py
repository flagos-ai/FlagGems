# HARDTANH operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    
    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)

import pytest
import triton

import flag_gems
from flag_gems.experimental_ops.hardtanh import hardtanh as gems_hardtanh, hardtanh_out as gems_hardtanh_out

import torch

@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardtanh_tensor_default(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_out = torch.ops.aten.hardtanh(ref_x)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out = gems_hardtanh(act_x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_max", [(-1.0, 1.0), (-0.5, 0.5), (0.0, 6.0), (-2.0, 0.5)])
def test_hardtanh_tensor_explicit(shape, dtype, min_max):
    min_val, max_val = min_max
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_out = torch.ops.aten.hardtanh(ref_x, min_val, max_val)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out = gems_hardtanh(act_x, min_val, max_val)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardtanh_out_default(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    ref_res = torch.ops.aten.hardtanh.out(ref_x, out=ref_out_buf)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out_buf = torch.empty_like(act_x)
        act_res = gems_hardtanh_out(act_x, out=act_out_buf)

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)
    gems_assert_close(act_res, ref_res, dtype=dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_max", [(-1.0, 1.0), (-0.5, 0.5), (0.0, 6.0), (-2.0, 0.5)])
def test_hardtanh_out_explicit(shape, dtype, min_max):
    min_val, max_val = min_max
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    ref_res = torch.ops.aten.hardtanh.out(ref_x, min_val, max_val, out=ref_out_buf)

    with flag_gems.use_gems():
        act_x = x.clone()
        act_out_buf = torch.empty_like(act_x)
        act_res = gems_hardtanh_out(act_x, min_val, max_val, out=act_out_buf)

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)
    gems_assert_close(act_res, ref_res, dtype=dtype)

@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardtanh_tensor_default_performance(shape, dtype):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardtanh(ref_x),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        act_x = x.clone()
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardtanh(act_x),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardtanh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_max", [(-1.0, 1.0), (-0.5, 0.5), (0.0, 6.0), (-2.0, 0.5)])
def test_hardtanh_tensor_explicit_performance(shape, dtype, min_max):

    quantiles = [0.5, 0.2, 0.8]

    min_val, max_val = min_max
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardtanh(ref_x, min_val, max_val),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        act_x = x.clone()
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardtanh(act_x, min_val, max_val),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardtanh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardtanh_out_default_performance(shape, dtype):

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardtanh.out(ref_x, out=ref_out_buf),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        act_x = x.clone()
        act_out_buf = torch.empty_like(act_x)
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardtanh_out(act_x, out=act_out_buf),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardtanh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardtanh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_max", [(-1.0, 1.0), (-0.5, 0.5), (0.0, 6.0), (-2.0, 0.5)])
def test_hardtanh_out_explicit_performance(shape, dtype, min_max):

    quantiles = [0.5, 0.2, 0.8]

    min_val, max_val = min_max
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardtanh.out(ref_x, min_val, max_val, out=ref_out_buf),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        act_x = x.clone()
        act_out_buf = torch.empty_like(act_x)
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardtanh_out(act_x, min_val, max_val, out=act_out_buf),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardtanh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardtanh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")