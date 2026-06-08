import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.arctanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_arctanh_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.arctanh_()
    with flag_gems.use_gems():
        res_out = inp.arctanh_()

    utils.gems_assert_close(res_out, ref_out, dtype)
