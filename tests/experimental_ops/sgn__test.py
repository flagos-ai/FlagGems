# SGN_ operator test

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


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.sgn_ import sgn_ as gems_sgn_  # noqa: E402


@pytest.mark.sgn_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sgn__tensor(shape, dtype):
    input_tensor = torch.randn(shape, device=flag_gems.device, dtype=torch.float32).to(
        dtype
    )
    flat = input_tensor.view(-1)
    if flat.numel() >= 1:
        flat[0] = flat.new_zeros(1)
    if flat.numel() >= 5:
        flat[4] = flat.new_zeros(1)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.sgn_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_sgn_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.sgn_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sgn__benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, device=flag_gems.device, dtype=torch.float32).to(
        dtype
    )
    flat = input_tensor.view(-1)
    if flat.numel() >= 1:
        flat[0] = flat.new_zeros(1)
    if flat.numel() >= 5:
        flat[4] = flat.new_zeros(1)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.sgn_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_sgn_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"sgn_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
