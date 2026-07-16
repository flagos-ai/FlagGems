import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SCALAR_VALUES = (0, 1.0, -1.0, 0.5)


@pytest.mark.less
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_less(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.less(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.less(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.less_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scalar", SCALAR_VALUES)
def test_less_scalar(shape, dtype, scalar):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.less(ref_inp1, scalar)
    with flag_gems.use_gems():
        res_out = torch.less(inp1, scalar)

    utils.gems_assert_equal(res_out, ref_out)
