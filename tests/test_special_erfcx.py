import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_erfcx
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# erfcx only supports float32
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_erfcx(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.erfcx(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.erfcx(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
