# TRIU_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
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
from flag_gems.experimental_ops.triu_ import triu_ as gems_triu_


@pytest.mark.triu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 64, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [0, 1, -1, 3])
def test_triu__tensor(shape, dtype, diagonal):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = base.clone()
    act_input = base.clone()

    ref_out = torch.ops.aten.triu_(ref_input, diagonal)

    with flag_gems.use_gems():
        act_out = gems_triu_(act_input, diagonal)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.triu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 64, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [0, 1, -1, 3])
def test_triu__benchmark_tensor(shape, dtype, diagonal):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = base.clone()
    act_input = base.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.triu_(ref_input, diagonal), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_triu_(act_input, diagonal), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"triu_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
