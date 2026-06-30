import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Exclude 1D shapes since inplace op requires matching dimensions
GREATER_EQUAL_SHAPES = [s for s in utils.POINTWISE_SHAPES if s != (1,)]


@pytest.mark.greater_equal_
@pytest.mark.parametrize("shape", GREATER_EQUAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_equal_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_inp1.greater_equal_(ref_inp2)
    with flag_gems.use_gems():
        inp1.greater_equal_(inp2)

    utils.gems_assert_equal(inp1, ref_inp1)
