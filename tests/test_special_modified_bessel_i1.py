import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_i1
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.modified_bessel_i1 does not support fp16/bf16 (CUDA not implemented for Half)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_modified_bessel_i1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.modified_bessel_i1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.modified_bessel_i1(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
