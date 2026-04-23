import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
    to_reference,
)


@pytest.mark.isneginf
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isneginf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = to_reference(inp)

    ref_out = torch.isneginf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isneginf(inp)

    gems_assert_equal(res_out, ref_out)
