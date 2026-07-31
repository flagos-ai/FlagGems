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
    # Cover full supported range n ∈ [0, 10]
    inp2 = torch.randint(0, 11, shape, dtype=torch.int32, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.shifted_chebyshev_polynomial_w(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.shifted_chebyshev_polynomial_w(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="NVIDIA-only CUDA JIT kernel; not supported on other backends",
)
@pytest.mark.special_shifted_chebyshev_polynomial_w
def test_special_shifted_chebyshev_polynomial_w_n_out_of_range():
    # Verify that n > 10 raises an error.
    # The validation lives in the Python wrapper function, not in the Triton kernel.
    # On Nvidia with PyTorch 2.10, torch.special dispatch goes through PyTorch's
    # native CUDA kernel and never reaches our wrapper, so we test the wrapper directly.
    from flag_gems.ops.special_shifted_chebyshev_polynomial_w import (
        special_shifted_chebyshev_polynomial_w,
    )

    x = torch.randn(3, dtype=torch.float32, device=flag_gems.device)
    n = torch.tensor([0, 5, 11], dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(ValueError, match="n must be <= 10"):
        special_shifted_chebyshev_polynomial_w(x, n)
