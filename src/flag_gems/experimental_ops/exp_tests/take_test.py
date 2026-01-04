# TAKE operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.take import take as gems_take
from flag_gems.experimental_ops.take import take_out as gems_take_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_tensor(in_shape, idx_shape, dtype):
    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    ref_x = x.clone()
    ref_idx = idx.clone()
    ref_out = torch.ops.aten.take(ref_x, ref_idx)

    with flag_gems.use_gems():
        act_out = gems_take(x, idx)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_out(in_shape, idx_shape, dtype):
    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    out_ref = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)
    out_act = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_idx = idx.clone()
    ref_out = torch.ops.aten.take.out(ref_x, ref_idx, out=out_ref)

    with flag_gems.use_gems():
        act_out = gems_take_out(x, idx, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_benchmark_tensor(in_shape, idx_shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    ref_x = x.clone()
    ref_idx = idx.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.take(ref_x, ref_idx), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_take(x, idx), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"take {idx_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_benchmark_out(in_shape, idx_shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    out_ref = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)
    out_act = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_idx = idx.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.take.out(ref_x, ref_idx, out=out_ref),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_take_out(x, idx, out_act), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"take {idx_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
