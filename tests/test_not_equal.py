import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.not_equal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_not_equal(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.not_equal(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.not_equal(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.not_equal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_not_equal_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.not_equal(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.not_equal(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
