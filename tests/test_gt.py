import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.gt
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_gt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.gt(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gt(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.gt
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_gt_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    inp2 = 0

    ref_out = torch.gt(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.gt(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
