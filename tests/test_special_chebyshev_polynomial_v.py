import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_chebyshev_polynomial_v
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    # torch reference only supports float32 on CUDA
    [torch.float32],
)
def test_special_chebyshev_polynomial_v(shape, dtype):
    # Clamp x to [-0.99, 0.99] to avoid division by zero at x=-1
    # and numerical instability near x=-1
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device).clamp(-0.99, 0.99)
    # n is the degree of the polynomial, use small positive integers
    n = torch.randint(0, 5, shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_n = utils.to_reference(n, True)

    ref_out = torch.special.chebyshev_polynomial_v(ref_x, ref_n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_v(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)
