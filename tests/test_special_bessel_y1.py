import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = [torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)


@pytest.mark.special_bessel_y1
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
# special.bessel_y1 only supports float32/float64; float16/bf16 raise RuntimeError
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_bessel_y1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.bessel_y1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_y1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
