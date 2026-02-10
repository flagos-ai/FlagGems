# PIXEL_SHUFFLE operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.pixel_shuffle import pixel_shuffle as gems_pixel_shuffle
from flag_gems.experimental_ops.pixel_shuffle import (
    pixel_shuffle_out as gems_pixel_shuffle_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

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


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize(
    "shape_r",
    [
        ((1, 4, 2, 3), 2),
        ((2, 9, 4, 4), 3),
        ((4, 64, 32, 32), 2),
        ((2, 128, 64, 64), 2),
        ((1, 64, 16, 16), 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_shuffle_tensor(shape_r, dtype):
    shape, r = shape_r
    n, c, h, w = shape
    x = torch.randn((n, c, h, w), dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.ops.aten.pixel_shuffle(ref_x, r)

    with flag_gems.use_gems():
        act_out = gems_pixel_shuffle(x, r)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize(
    "shape_r",
    [
        ((1, 4, 2, 3), 2),
        ((2, 9, 4, 4), 3),
        ((4, 64, 32, 32), 2),
        ((2, 128, 64, 64), 2),
        ((1, 64, 16, 16), 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_shuffle_out(shape_r, dtype):
    shape, r = shape_r
    n, c, h, w = shape
    x = torch.randn((n, c, h, w), dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    out_shape = (n, c // (r * r), h * r, w * r)
    ref_out = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
    torch.ops.aten.pixel_shuffle.out(ref_x, r, out=ref_out)

    act_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        gems_pixel_shuffle_out(x, r, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)
