import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_bessel_y1
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
# special.bessel_y1 only supports float32/float64; float16/bf16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_bessel_y1(shape, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("Skipping fp64 test on platform without fp64 support")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.bessel_y1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_y1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
