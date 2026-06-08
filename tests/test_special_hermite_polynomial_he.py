import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="NVIDIA-only CUDA JIT kernel; not supported on other backends",
)
@pytest.mark.special_hermite_polynomial_he
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# CUDA does not support half/bfloat16 for this special function
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_hermite_polynomial_he(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # n is a tensor with small integer values (degree of polynomial)
    inp2 = torch.randint(0, 6, shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.hermite_polynomial_he(ref_inp1.cpu(), ref_inp2.cpu())
    with flag_gems.use_gems():
        res_out = torch.special.hermite_polynomial_he(inp1, inp2)

    utils.gems_assert_close(res_out.cpu(), ref_out, dtype, equal_nan=True)

    # Also test scalar n path
    for n in [0, 1, 2, 3, 4, 5]:
        ref_out = torch.special.hermite_polynomial_he(ref_inp1.cpu(), n)
        with flag_gems.use_gems():
            res_out = torch.special.hermite_polynomial_he(inp1, n)

        utils.gems_assert_close(res_out.cpu(), ref_out, dtype, equal_nan=True)
