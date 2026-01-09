# DIGAMMA_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.digamma_ import digamma_ as gems_digamma_  # noqa: E402


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.digamma_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_digamma__tensor(shape, dtype):
    base = torch.rand(shape, device=flag_gems.device, dtype=torch.float32) * 9.5 + 0.5
    input_tensor = base.to(dtype)

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.digamma_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_digamma_(input_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.digamma_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_digamma__benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    base = torch.rand(shape, device=flag_gems.device, dtype=torch.float32) * 9.5 + 0.5
    input_tensor = base.to(dtype)

    ref_input = input_tensor.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.digamma_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_digamma_(input_tensor), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"digamma_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
