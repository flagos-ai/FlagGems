import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_scaled_modified_bessel_k0
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
# Only test float32 since PyTorch doesn't support float16/bfloat16 for special operators
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_scaled_modified_bessel_k0(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.special_scaled_modified_bessel_k0(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_scaled_modified_bessel_k0(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True, atol=0.02)
