import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# torch.special.chebyshev_polynomial_u only supports float32 on CUDA
@pytest.mark.special_chebyshev_polynomial_u
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
# torch.special.chebyshev_polynomial_u does not support float16/bfloat16 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_special_chebyshev_polynomial_u(shape, dtype, n):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.special.chebyshev_polynomial_u(ref_x, n)

    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_u(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


# torch.special.chebyshev_polynomial_u only supports float32 on CUDA
@pytest.mark.special_chebyshev_polynomial_u
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
# torch.special.chebyshev_polynomial_u does not support float16/bfloat16 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_u_tensor_n(shape, dtype):
    n = torch.randint(0, 6, shape, dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_n = utils.to_reference(n)
    ref_x = utils.to_reference(x)
    ref_out = torch.special.chebyshev_polynomial_u(ref_x, ref_n)

    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_u(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


# torch.special.chebyshev_polynomial_u only supports float32 on CUDA
@pytest.mark.special_chebyshev_polynomial_u
# torch.special.chebyshev_polynomial_u does not support float16/bfloat16 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("n", [-1, 6, 10])
def test_special_chebyshev_polynomial_u_guard_scalar(dtype, n):
    """Verify that n outside [0, 5] raises ValueError."""
    x = torch.randn(4, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(ValueError, match="must be in \\[0, 5\\]"):
            torch.special.chebyshev_polynomial_u(x, n)


# torch.special.chebyshev_polynomial_u only supports float32 on CUDA
@pytest.mark.special_chebyshev_polynomial_u
# torch.special.chebyshev_polynomial_u does not support float16/bfloat16 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_u_guard_tensor(dtype):
    """Verify that tensor n with values outside [0, 5] raises ValueError."""
    # Fixed shape is sufficient for guard behavior testing
    shape = (8,)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, shape, dtype=torch.int32, device=flag_gems.device)
    # Force at least one value to be out of range
    n[0] = 6
    with flag_gems.use_gems():
        with pytest.raises(ValueError, match="must be in \\[0, 5\\]"):
            torch.special.chebyshev_polynomial_u(x, n)
