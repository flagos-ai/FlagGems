import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_airy_ai
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# airy_ai_cuda does not support Half/BFloat16
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_airy_ai(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.airy_ai(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.airy_ai(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
