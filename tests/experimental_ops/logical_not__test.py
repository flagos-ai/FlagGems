# LOGICAL_NOT_ operator test

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
from flag_gems.experimental_ops.logical_not_ import (  # noqa: E402
    logical_not_ as gems_logical_not_,
)


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


@pytest.mark.logical_not_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 2048)])
@pytest.mark.parametrize("dtype", [torch.bool])
@pytest.mark.parametrize("contig", [True, False])
def test_logical_not__tensor(shape, dtype, contig):
    if contig:
        input_tensor = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    else:
        base = torch.randint(0, 2, (shape[1], shape[0]), device=flag_gems.device).to(
            dtype
        )
        input_tensor = base.transpose(0, 1)
    ref_input = to_reference(input_tensor)
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.logical_not_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_logical_not_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logical_not_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 2048)])
@pytest.mark.parametrize("dtype", [torch.bool])
@pytest.mark.parametrize("contig", [True, False])
def test_logical_not__benchmark_tensor(shape, dtype, contig):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    if contig:
        input_tensor = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    else:
        base = torch.randint(0, 2, (shape[1], shape[0]), device=flag_gems.device).to(
            dtype
        )
        input_tensor = base.transpose(0, 1)
    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.logical_not_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_logical_not_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"logical_not_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
