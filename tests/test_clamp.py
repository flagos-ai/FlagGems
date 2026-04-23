import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.clamp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", utils.SCALARS)
@pytest.mark.parametrize("mini", utils.SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clamp_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", utils.SCALARS)
@pytest.mark.parametrize("mini", utils.SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.clamp_(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp_(inp, min=mini, max=maxi)

    utils.gems_assert_equal(res_out, ref_out)
