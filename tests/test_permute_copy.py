import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

PERMUTE_COPY_SHAPES_AND_DIMS = utils.PERMUTE_COPY_SHAPES_AND_DIMS


@pytest.mark.permute_copy
@pytest.mark.parametrize("shape, dims", PERMUTE_COPY_SHAPES_AND_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_permute_copy(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.permute_copy(ref_inp, dims)
    with flag_gems.use_gems():
        res_out = torch.permute_copy(inp, dims)

    utils.gems_assert_close(res_out, ref_out, dtype)
