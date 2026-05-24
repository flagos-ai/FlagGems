import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_bessel_y1
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_bessel_y1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.bessel_y1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_y1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
