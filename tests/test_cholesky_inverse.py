import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for cholesky_inverse: square matrices from small to medium
CHOLESKY_INVERSE_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]

# Batched shapes for cholesky_inverse: (batch, n, n)
CHOLESKY_INVERSE_BATCH_SHAPES = [
    (4, 4, 4),
    (2, 8, 8),
    (3, 16, 16),
]


# cholesky_inverse only supports float32/float64. On the Kunlunxin XPU backend a
# float64 request is silently downgraded to float32, so it can never match a
# true-fp64 reference; skip fp64 on that backend only. Every other vendor keeps
# running float64 exactly as before.
_SKIP_FP64_ON_KUNLUNXIN = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name == "kunlunxin",
    reason="Kunlunxin XPU has no real float64 (silently downgraded to float32)",
)
CHOLESKY_INVERSE_DTYPES = [
    torch.float32,
    pytest.param(torch.float64, marks=_SKIP_FP64_ON_KUNLUNXIN),
]


def _make_positive_definite(shape, dtype, device):
    """Create a positive-definite matrix and return its Cholesky factor."""
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=device)
    A = B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device) * n
    L = torch.linalg.cholesky(A)
    return L


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_SHAPES)
# cholesky_inverse only supports float32/float64
@pytest.mark.parametrize("dtype", CHOLESKY_INVERSE_DTYPES)
def test_cholesky_inverse(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    ref_L = utils.to_reference(L)

    ref_out = torch.cholesky_inverse(ref_L, upper=False)

    with flag_gems.use_gems():
        res_out = torch.cholesky_inverse(L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_SHAPES[:3])
@pytest.mark.parametrize("dtype", CHOLESKY_INVERSE_DTYPES)
def test_cholesky_inverse_upper(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    U = L.transpose(-2, -1).contiguous()
    ref_U = utils.to_reference(U)

    ref_out = torch.cholesky_inverse(ref_U, upper=True)

    with flag_gems.use_gems():
        res_out = torch.cholesky_inverse(U, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", CHOLESKY_INVERSE_DTYPES)
def test_cholesky_inverse_batch(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    ref_L = utils.to_reference(L)

    ref_out = torch.cholesky_inverse(ref_L, upper=False)

    with flag_gems.use_gems():
        res_out = torch.cholesky_inverse(L, upper=False)

    utils.gems_assert_close(res_out, ref_out, dtype)
