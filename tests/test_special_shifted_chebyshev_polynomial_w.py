import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# TODO: Reference GitHub issue for multi-backend support
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="NVIDIA-only CUDA JIT kernel; not supported on other backends",
)
@pytest.mark.special_shifted_chebyshev_polynomial_w
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# PyTorch reference only supports float32 for this operator
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_w(shape, dtype):
    # x: the values to evaluate the polynomial at
    # n: the degree of the shifted Chebyshev polynomial
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Use small positive integers for n to avoid numerical issues
    inp2 = torch.randint(0, 5, shape, dtype=torch.int32, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.shifted_chebyshev_polynomial_w(
        ref_inp1.cpu(), ref_inp2.cpu()
    )
    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_w(inp1, inp2)

    utils.gems_assert_close(res_out.cpu(), ref_out, dtype)
