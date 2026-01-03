# TRIU operator test

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
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.triu import triu as gems_triu
from flag_gems.experimental_ops.triu import triu_out as gems_triu_out


@pytest.mark.triu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 1, 3])
def test_triu_tensor(shape, dtype, diagonal):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.triu(ref_x, diagonal)

    with flag_gems.use_gems():
        act_out = gems_triu(x, diagonal)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.triu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 1, 3])
def test_triu_out(shape, dtype, diagonal):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)

    ref_out = torch.ops.aten.triu.out(ref_x, diagonal, out=ref_out_buf)

    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_triu_out(x, diagonal, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.triu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 1, 3])
def test_triu_benchmark_tensor(shape, dtype, diagonal):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.triu(ref_x, diagonal), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_triu(x, diagonal), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"triu {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.triu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-1, 0, 1, 3])
def test_triu_benchmark_out(shape, dtype, diagonal):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.triu.out(ref_x, diagonal, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty_like(x)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_triu_out(x, diagonal, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"triu {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
