import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_log1p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log1p(shape, dtype):
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.special.log1p(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.log1p(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
