import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.log
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
