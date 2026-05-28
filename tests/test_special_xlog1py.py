import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlog1py(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.xlog1py(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.xlog1py(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
