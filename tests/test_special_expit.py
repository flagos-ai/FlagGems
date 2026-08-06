import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_expit
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_expit(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.special.expit(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.expit(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
