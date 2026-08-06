import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Exclude 1D shapes since inplace op requires matching dimensions
GE_SHAPES = [s for s in utils.POINTWISE_SHAPES if s != (1,)]


@pytest.mark.ge_
@pytest.mark.parametrize("shape", GE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ge_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_inp1.ge_(ref_inp2)
    with flag_gems.use_gems():
        inp1.ge_(inp2)

    utils.gems_assert_equal(inp1, ref_inp1)
