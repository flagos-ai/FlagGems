import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

TILE_DIMS = [(0,), (2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]


@pytest.mark.tile
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dims", TILE_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tile(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.tile(ref_inp, dims)
    with flag_gems.use_gems():
        res_out = torch.tile(inp, dims)

    utils.gems_assert_close(res_out, ref_out, dtype)
