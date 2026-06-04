import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fix
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fix(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 10
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.fix(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.fix(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
