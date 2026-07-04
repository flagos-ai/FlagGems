import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.channel_shuffle
@pytest.mark.parametrize(
    "shape_groups", [((1, 4, 2, 2), 2), ((2, 8, 4, 4), 4), ((4, 16, 8, 8), 4)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_channel_shuffle(shape_groups, dtype):
    shape, groups = shape_groups
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_out = torch.ops.aten.channel_shuffle(ref_input, groups)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.channel_shuffle(input_tensor, groups)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)
