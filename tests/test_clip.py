import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.clip
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", utils.SCALARS)
@pytest.mark.parametrize("mini", utils.SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clip(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clip(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clip(inp, min=mini, max=maxi)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clip_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", utils.SCALARS)
@pytest.mark.parametrize("mini", utils.SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clip_(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.clip_(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clip_(inp, min=mini, max=maxi)

    utils.gems_assert_equal(res_out, ref_out)