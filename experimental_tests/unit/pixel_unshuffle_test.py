# PIXEL_UNSHUFFLE operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.pixel_unshuffle import (  # noqa: E402
    pixel_unshuffle as gems_pixel_unshuffle,
)
from flag_gems.experimental_ops.pixel_unshuffle import (  # noqa: E402
    pixel_unshuffle_out as gems_pixel_unshuffle_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


@pytest.mark.pixel_unshuffle
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_unshuffle_tensor(shape_factor, dtype):
    shape, downscale_factor = shape_factor
    input_tensor = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(
        dtype
    )

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.pixel_unshuffle(ref_input, downscale_factor)

    with flag_gems.use_gems():
        act_out = gems_pixel_unshuffle(input_tensor, downscale_factor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.pixel_unshuffle
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_unshuffle_out(shape_factor, dtype):
    shape, downscale_factor = shape_factor
    N, C, H, W = shape
    r = downscale_factor
    out_shape = (N, C * (r * r), H // r, W // r)

    input_tensor = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(
        dtype
    )
    ref_input = to_reference(input_tensor)

    out_ref = torch.empty(out_shape, dtype=dtype, device=ref_input.device)
    ref_out = torch.ops.aten.pixel_unshuffle.out(
        ref_input, downscale_factor, out=out_ref
    )

    out_act = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_pixel_unshuffle_out(input_tensor, downscale_factor, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)
