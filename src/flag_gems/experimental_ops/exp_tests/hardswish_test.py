# HARDSWISH operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.hardswish import hardswish as gems_hardswish
from flag_gems.experimental_ops.hardswish import hardswish_out as gems_hardswish_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.hardswish
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardswish_tensor(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()
    ref_out = torch.ops.aten.hardswish(ref_x)

    with flag_gems.use_gems():
        act_out = gems_hardswish(x.clone())

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardswish
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardswish_out(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()
    act_x = x.clone()

    ref_out = torch.empty_like(ref_x)
    act_out = torch.empty_like(act_x)

    ref_ret = torch.ops.aten.hardswish.out(ref_x, out=ref_out)
    with flag_gems.use_gems():
        act_ret = gems_hardswish_out(act_x, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(act_ret, ref_ret, dtype=dtype)


@pytest.mark.hardswish
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardswish_benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardswish(ref_x), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardswish(x.clone()), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardswish {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.hardswish
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardswish_benchmark_out(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()
    act_x = x.clone()

    ref_out = torch.empty_like(ref_x)
    act_out = torch.empty_like(act_x)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardswish.out(ref_x, out=ref_out),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardswish_out(act_x, act_out), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardswish {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardswish {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
