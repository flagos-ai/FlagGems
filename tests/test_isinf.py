import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.isinf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isinf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isinf(inp)

    utils.gems_assert_equal(res_out, ref_out)
