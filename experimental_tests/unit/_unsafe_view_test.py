# _UNSAFE_VIEW operator test
import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops._unsafe_view import _unsafe_view as gems__unsafe_view
from flag_gems.experimental_ops._unsafe_view import (
    _unsafe_view_out as gems__unsafe_view_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


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


@pytest.mark.unsafe_view
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), (6,)),
        ((2, 3), (3, 2)),
        ((128, 256), (256, 128)),
        ((64, 64, 8), (128, 256)),
        ((32, 16, 8), (64, 64)),
        ((512,), (256, 2)),
        ((512, 512), (65536, 4)),
        ((1024, 1024), (2048, 512)),
        ((16, 8, 4, 2), (32, 32)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__unsafe_view_tensor(case, dtype):
    shape, size = case
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)

    ref_out = torch.ops.aten._unsafe_view(ref_input, size)

    with flag_gems.use_gems():
        act_out = gems__unsafe_view(input_tensor, size)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.unsafe_view
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), (6,)),
        ((2, 3), (3, 2)),
        ((128, 256), (256, 128)),
        ((64, 64, 8), (128, 256)),
        ((32, 16, 8), (64, 64)),
        ((512,), (256, 2)),
        ((512, 512), (65536, 4)),
        ((1024, 1024), (2048, 512)),
        ((16, 8, 4, 2), (32, 32)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__unsafe_view_out(case, dtype):
    shape, size = case
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)

    ref_out_holder = torch.empty(0, dtype=dtype, device=ref_input.device)
    ref_out = torch.ops.aten._unsafe_view.out(ref_input, size, out=ref_out_holder)

    act_out_holder = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems__unsafe_view_out(input_tensor, size, act_out_holder)

    gems_assert_close(act_out, ref_out, dtype=dtype)
