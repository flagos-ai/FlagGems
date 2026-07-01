import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.inplace
@pytest.mark.frac_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_frac_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.frac_()
    with flag_gems.use_gems():
        res_out = inp.frac_()

    utils.gems_assert_equal(res_out, ref_out)
