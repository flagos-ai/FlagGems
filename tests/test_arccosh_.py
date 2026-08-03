import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arccosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccosh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # arccosh requires input >= 1
    inp = inp.abs() + 1.0
    ref_inp = utils.to_reference(inp, True)
    act_inp = inp.clone()

    ref_inp.arccosh_()
    with flag_gems.use_gems():
        act_inp.arccosh_()

    utils.gems_assert_close(act_inp, ref_inp, dtype)
