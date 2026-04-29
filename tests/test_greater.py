import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
    to_reference,
)


@pytest.mark.greater
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_greater(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.greater(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.greater(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.greater
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_greater_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    inp2 = 0

    ref_out = torch.greater(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.greater(inp1, inp2)

    gems_assert_equal(res_out, ref_out)
