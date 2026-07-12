import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.dot
@pytest.mark.parametrize("shape", utils.UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dot(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.dot(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.dot(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
