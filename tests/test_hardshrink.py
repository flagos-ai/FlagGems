import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.hardshrink
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("lambd", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardshrink(shape, dtype, lambd):
    """Test hardshrink with various shapes and lambda values."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_output = torch.ops.aten.hardshrink(ref_x, lambd)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.hardshrink(x, lambd)

    utils.gems_assert_close(res_output, ref_output, dtype)
