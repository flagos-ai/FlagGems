import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.frexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_frexp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_mantissa, ref_exponent = torch.frexp(ref_inp)
    with flag_gems.use_gems():
        res_mantissa, res_exponent = torch.frexp(inp)

    utils.gems_assert_close(res_mantissa, ref_mantissa, dtype)
    utils.gems_assert_equal(res_exponent, ref_exponent)
