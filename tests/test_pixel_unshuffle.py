import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.pixel_unshuffle
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pixel_unshuffle(shape_factor, dtype):
    shape, downscale_factor = shape_factor
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_out = torch.ops.aten.pixel_unshuffle(ref_input, downscale_factor)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.pixel_unshuffle(input_tensor, downscale_factor)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.pixel_unshuffle_out
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pixel_unshuffle_out(shape_factor, dtype):
    shape, downscale_factor = shape_factor
    N, C, H, W = shape
    r = downscale_factor
    out_shape = (N, C * (r * r), H // r, W // r)

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = utils.to_reference(input_tensor, True)

    out_ref = torch.empty(out_shape, dtype=ref_input.dtype, device=ref_input.device)
    ref_out = torch.ops.aten.pixel_unshuffle.out(
        ref_input, downscale_factor, out=out_ref
    )

    out_act = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.pixel_unshuffle.out(
            input_tensor, downscale_factor, out=out_act
        )

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)
