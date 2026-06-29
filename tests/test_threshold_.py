import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.threshold_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_threshold_(shape, dtype):
    threshold_val = 0.0
    value_val = -1.0
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.threshold_(ref_inp, threshold_val, value_val)
    with flag_gems.use_gems():
        res_out = torch.threshold_(inp, threshold_val, value_val)

    # Compare return values
    utils.gems_assert_close(res_out, ref_out, dtype)
    # Compare mutated input
    utils.gems_assert_close(inp, ref_inp, dtype)
