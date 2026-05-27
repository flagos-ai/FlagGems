import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

PERMUTE_COPY_SHAPES_AND_DIMS = [
    # 1D cases
    ((24,), (0,)),
    ((12,), (0,)),
    # 2D cases
    ((2, 3), (0, 1)),
    ((2, 3), (1, 0)),
    ((3, 4), (1, 0)),
    ((16, 128), (1, 0)),
    # 3D cases (from original PERMUTE_COPY_SHAPES x PERMUTE_COPY_PERMS)
    ((2, 3, 4), (0, 2, 1)),
    ((2, 3, 4), (1, 0, 2)),
    ((2, 3, 4), (2, 1, 0)),
    ((4, 2, 3), (1, 2, 0)),
    ((4, 2, 3), (2, 0, 1)),
]


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
