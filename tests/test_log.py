import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.log
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)