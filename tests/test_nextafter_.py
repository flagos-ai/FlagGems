import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Kernel uses int32 bitcast which only supports float32; fp16/bf16 would produce incorrect results.
NEXTAFTER_DTYPES = [torch.float32]


@pytest.mark.nextafter_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", NEXTAFTER_DTYPES)
def test_nextafter_(shape, dtype):
    # Test nextafter_: returns the next representable value from x toward y
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x).clone()
    ref_y = utils.to_reference(y)
    ref_x.nextafter_(ref_y)

    with flag_gems.use_gems():
        x_clone = x.clone()
        x_clone.nextafter_(y)
        utils.gems_assert_close(x_clone, ref_x, dtype)
