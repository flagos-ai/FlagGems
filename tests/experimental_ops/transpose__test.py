# TRANSPOSE_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.transpose_ import transpose_ as gems_transpose_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.transpose_
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), 0, 1),
        ((128, 256), 1, 0),
        ((512, 512), -1, -2),
        ((4, 5, 6), 0, 2),
        ((4, 5, 6), -1, -3),
        ((3, 4, 5, 6), 1, 3),
        ((3, 4, 5, 6), -2, -4),
        ((1024, 2048), 0, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_transpose__tensor(case, dtype):
    shape, dim0, dim1 = case
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.transpose_(ref_input, dim0, dim1)

    with flag_gems.use_gems():
        act_out = gems_transpose_(act_input, dim0, dim1)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.transpose_
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), 0, 1),
        ((128, 256), 1, 0),
        ((512, 512), -1, -2),
        ((4, 5, 6), 0, 2),
        ((4, 5, 6), -1, -3),
        ((3, 4, 5, 6), 1, 3),
        ((3, 4, 5, 6), -2, -4),
        ((1024, 2048), 0, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_transpose__benchmark_tensor(case, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, dim0, dim1 = case
    ref_input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    act_input = ref_input.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.transpose_(ref_input, dim0, dim1),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_transpose_(act_input, dim0, dim1), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"transpose_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
