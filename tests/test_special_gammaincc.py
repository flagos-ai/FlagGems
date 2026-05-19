import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_gammaincc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# igammac_cuda is not implemented for Half/BFloat16
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_gammaincc(shape, dtype):
    inp_a = torch.abs(torch.randn(shape, dtype=dtype, device=flag_gems.device)) + 0.1
    inp_x = torch.abs(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    ref_a = utils.to_reference(inp_a)
    ref_x = utils.to_reference(inp_x)

    ref_out = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res_out = torch.special.gammaincc(inp_a, inp_x)

    utils.gems_assert_close(res_out, ref_out, dtype)
