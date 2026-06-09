import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.asin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.asin_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asin_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    utils.gems_assert_close(inp, ref_inp, dtype, True)
