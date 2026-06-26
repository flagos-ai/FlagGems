import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reduce_prod
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_prod(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.prod(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.prod(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
