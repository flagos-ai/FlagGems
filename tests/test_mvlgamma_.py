import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Tests for mvlgamma_ (multivariate log-gamma function)
MVLGAMMA_SHAPES = [(1024,), (1024, 1024), (16, 32, 64)]


@pytest.mark.mvlgamma_
@pytest.mark.parametrize("shape", MVLGAMMA_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [2, 3, 4])
def test_mvlgamma_(shape, dtype, p):
    # Input must be > (p-1)/2 for mvlgamma_ to be defined
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + (p - 1) / 2 + 0.1
    ref_inp = utils.to_reference(inp)

    ref_out = ref_inp.mvlgamma_(p=p)
    with flag_gems.use_gems():
        res_out = inp.mvlgamma_(p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
