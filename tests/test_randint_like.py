import pytest
import torch

import flag_gems

from .accuracy_utils import (
    DISTRIBUTION_SHAPES,
    FLOAT_DTYPES,
    gems_assert_equal,
    to_reference,
)


@pytest.mark.randint_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_randint_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.randint_like(x, 10)
    ref_out = to_reference(res_out)
    # Values should be in [0, 10)
    assert (ref_out >= 0).all()
    assert (ref_out < 10).all()
    # Verify output has same shape as input
    gems_assert_equal(
        torch.tensor(list(res_out.shape)),
        torch.tensor(list(x.shape)),
    )
