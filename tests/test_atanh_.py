import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.atanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    ref_inp = utils.to_reference(inp, True)

    ref_out = ref_inp.atanh_()

    inp1 = inp.clone()
    with flag_gems.use_gems():
        res_out = inp1.atanh_()

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp1, ref_inp, dtype)
    assert res_out is inp1
