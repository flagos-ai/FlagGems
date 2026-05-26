import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.digamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma(shape, dtype):
    # Use positive inputs to avoid extreme values and NaN at poles
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)
    ref_out = torch.digamma(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.digamma(inp)

    # Use more lenient tolerance for digamma due to its wide range of output values
    if dtype == torch.float16:
        # Float16 needs higher tolerance due to precision loss from float32 computation
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2)
    elif dtype == torch.bfloat16:
        # Bfloat16 needs even higher tolerance due to its lower precision
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-1)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
