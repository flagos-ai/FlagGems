# FLOOR_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.floor_ import floor_ as gems_floor_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.floor_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_floor__tensor(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 5.3

    ref_input = base.clone()
    act_input = base.clone()

    ref_out = torch.ops.aten.floor_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_floor_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.floor_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_floor__benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 5.3

    ref_input = base.clone()
    act_input = base.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.floor_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_floor_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"floor_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
