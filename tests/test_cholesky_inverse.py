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


def _factor_metadata(factor):
    return (
        factor.device,
        factor.dtype,
        tuple(factor.shape),
        tuple(factor.stride()),
        factor.storage_offset(),
        factor.untyped_storage().data_ptr(),
        factor.requires_grad,
    )


def _snapshot_factor(factor):
    metadata = _factor_metadata(factor)
    values = factor.detach().clone() if factor.numel() else None
    return metadata, values


def _assert_readonly_output_abi(result, factor, snapshot):
    metadata, values = snapshot
    assert _factor_metadata(factor) == metadata
    if values is not None:
        assert torch.equal(factor, values)
    assert isinstance(result, torch.Tensor)
    assert result.device == metadata[0]
    assert result.dtype == metadata[1]
    assert tuple(result.shape) == metadata[2]
    if values is None:
        # Empty tensors can all have data_ptr=0: Python identity is required.
        assert result is factor
    else:
        assert result.untyped_storage().data_ptr() != metadata[5]


def _make_positive_definite(shape, dtype, device):
    """Create a positive-definite matrix and return its Cholesky factor."""
    if (
        dtype == torch.float64
        and torch.device(device).type != "cpu"
        and flag_gems.vendor_name in {"ascend", "iluvatar"}
    ):
        pytest.skip(
            "backend float64 matrix arithmetic/Cholesky cannot construct a valid factor"
        )
    torch.manual_seed(0)
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
    ref_L = utils.to_reference(L, True)

    ref_out = torch.cholesky_inverse(ref_L, upper=False)
    snapshot = _snapshot_factor(L)
    res_out = flag_gems.cholesky_inverse(L, upper=False)
    _assert_readonly_output_abi(res_out, L, snapshot)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_SHAPES[:3])
@pytest.mark.parametrize("dtype", CHOLESKY_INVERSE_DTYPES)
def test_cholesky_inverse_upper(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    U = L.transpose(-2, -1).contiguous()
    ref_U = utils.to_reference(U, True)

    ref_out = torch.cholesky_inverse(ref_U, upper=True)
    snapshot = _snapshot_factor(U)
    res_out = flag_gems.cholesky_inverse(U, upper=True)
    _assert_readonly_output_abi(res_out, U, snapshot)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", CHOLESKY_INVERSE_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", CHOLESKY_INVERSE_DTYPES)
def test_cholesky_inverse_batch(shape, dtype):
    L = _make_positive_definite(shape, dtype, flag_gems.device)
    ref_L = utils.to_reference(L, True)

    ref_out = torch.cholesky_inverse(ref_L, upper=False)
    snapshot = _snapshot_factor(L)
    res_out = flag_gems.cholesky_inverse(L, upper=False)
    _assert_readonly_output_abi(res_out, L, snapshot)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_inverse
@pytest.mark.parametrize("shape", [(0, 0), (2, 0, 0)])
@pytest.mark.parametrize("dtype", CHOLESKY_INVERSE_DTYPES)
def test_cholesky_inverse_empty_alias(shape, dtype):
    factor = torch.empty(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    original_pointer = factor.untyped_storage().data_ptr()

    snapshot = _snapshot_factor(factor)
    result = flag_gems.cholesky_inverse(factor)
    _assert_readonly_output_abi(result, factor, snapshot)

    assert result is factor
    assert result.untyped_storage().data_ptr() == original_pointer
    assert result.shape == factor.shape
    assert result.dtype == factor.dtype
    gradient = torch.autograd.grad(result.sum(), factor)[0]
    assert gradient.device == factor.device
    utils.gems_assert_equal(gradient, utils.to_reference(torch.ones_like(factor)))


@pytest.mark.cholesky_inverse
def test_cholesky_inverse_autograd_rejected():
    factor = _make_positive_definite(
        (4, 4), torch.float32, flag_gems.device
    ).requires_grad_(True)

    snapshot = _snapshot_factor(factor)
    result = flag_gems.cholesky_inverse(factor)
    _assert_readonly_output_abi(result, factor, snapshot)

    # Nonempty inputs run both Triton kernels under torch.no_grad and no custom
    # autograd Function is registered, so gradients are explicitly unsupported.
    assert not result.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(result.sum(), factor)
