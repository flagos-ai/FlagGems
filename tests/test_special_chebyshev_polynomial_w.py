import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# special.* Chebyshev polynomials only support float32 in PyTorch reference
@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
# special.* Chebyshev polynomials: torch ref only supports float32
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w(shape, dtype):
    # x in [-1, 1] (Chebyshev domain); kernel clamps out-of-bound values
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x)
    # Small polynomial degree for accuracy comparison
    n = 3

    ref_out = torch.special.chebyshev_polynomial_w(ref_x, n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_w(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
