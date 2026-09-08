import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# torch.special.bessel_y0 supports float32 and float64
FLOAT_DTYPES = [torch.float32, torch.float64]

# Pointwise shapes covering small, medium, and batched 2D tensors
POINTWISE_SHAPES = [(128,), (512, 256), (2, 128, 128)]


@pytest.mark.special_bessel_y0
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_bessel_y0(shape, dtype):
    # bessel_y0 requires positive inputs (returns NaN for negative values)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs() + 0.1
    ref_inp = utils.to_reference(inp)
    ref_out = torch.special.bessel_y0(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_y0(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
