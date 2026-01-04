# TRIL_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.tril_ import tril_ as gems_tril_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.tril_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (32, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
def test_tril__tensor(shape, dtype, diagonal):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    act_x = x.clone()

    ref_out = torch.ops.aten.tril_(ref_x, diagonal)

    with flag_gems.use_gems():
        act_out = gems_tril_(act_x, diagonal)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.tril_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (32, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tril__tensor_default_diagonal(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    act_x = x.clone()

    ref_out = torch.ops.aten.tril_(ref_x)

    with flag_gems.use_gems():
        act_out = gems_tril_(act_x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.tril_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (32, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
def test_tril__benchmark_tensor(shape, dtype, diagonal):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    act_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.tril_(ref_x, diagonal), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_tril_(act_x, diagonal), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"tril_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.tril_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (32, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tril__tensor_default_diagonal_performance(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    act_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.tril_(ref_x), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_tril_(act_x), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"tril_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
