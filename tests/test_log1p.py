import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.log1p_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_(shape, dtype):
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.log1p_()
    with flag_gems.use_gems():
        res_out = inp.log1p_()
    utils.gems_assert_close(res_out, ref_out, dtype)
