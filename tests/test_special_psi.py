import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_psi
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_psi(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.psi(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.psi(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
