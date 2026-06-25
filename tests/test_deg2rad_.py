import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.deg2rad_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 360.0
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.deg2rad_(ref_inp)

    act_inp = inp.clone()
    with flag_gems.use_gems():
        res_out = torch.deg2rad_(act_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(act_inp, ref_inp, dtype)
    assert res_out is act_inp
