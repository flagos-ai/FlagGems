import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# argwhere returns int64 indices regardless of input dtype, use common float dtypes for input
ARGWHERE_DTYPES = utils.FLOAT_DTYPES
# Shapes covering 1D through 4D to validate multi-dimensional index decomposition
ARGWHERE_SHAPES = [(4096,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60)]


@pytest.mark.argwhere
@pytest.mark.parametrize("shape", ARGWHERE_SHAPES)
@pytest.mark.parametrize("dtype", ARGWHERE_DTYPES)
def test_argwhere(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.argwhere(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.argwhere(inp)

    utils.gems_assert_equal(res_out, ref_out)
