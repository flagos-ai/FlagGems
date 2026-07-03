import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize(
    "shape_factor",
    [((1, 12, 4, 4), 2), ((2, 18, 3, 3), 3), ((1, 4, 8, 8), 2), ((4, 36, 2, 2), 3)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pixel_shuffle(shape_factor, dtype):
    shape, upscale_factor = shape_factor
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_out = torch.ops.aten.pixel_shuffle(ref_input, upscale_factor)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.pixel_shuffle(input_tensor, upscale_factor)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)

