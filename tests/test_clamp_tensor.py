import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.clamp_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_clamp_tensor(shape, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    maxi = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = utils.to_reference(inp)
    ref_maxi = utils.to_reference(maxi)
    ref_mini = utils.to_reference(mini)

    ref_out = torch.clamp(ref_inp, min=ref_mini, max=ref_maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clamp_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_clamp_tensor_(shape, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    maxi = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = utils.to_reference(inp.clone())
    ref_maxi = utils.to_reference(maxi)
    ref_mini = utils.to_reference(mini)

    ref_out = torch.clamp_(ref_inp, min=ref_mini, max=ref_maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp_(inp, min=mini, max=maxi)

    utils.gems_assert_equal(res_out, ref_out)