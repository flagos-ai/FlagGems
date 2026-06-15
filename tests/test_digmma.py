import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.digamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.digamma_()
    with flag_gems.use_gems():
        res_out = inp.digamma_()

    utils.gems_assert_close(res_out, ref_out, dtype)
