# T operator test

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
from flag_gems.experimental_ops.t import t as gems_t


@pytest.mark.t
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("contig", [True, False])
def test_t_tensor(shape, dtype, contig):
    if contig:
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        base = torch.randn((shape[1], shape[0]), dtype=dtype, device=flag_gems.device)
        input_tensor = base.t()
    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.t(ref_input)
    with flag_gems.use_gems():
        act_out = gems_t(input_tensor)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.t
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("contig", [True, False])
def test_t_benchmark_tensor(shape, dtype, contig):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    if contig:
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        base = torch.randn((shape[1], shape[0]), dtype=dtype, device=flag_gems.device)
        input_tensor = base.t()
    ref_input = input_tensor.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.t(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_t(input_tensor), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"t {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
