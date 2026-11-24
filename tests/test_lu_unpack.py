import pytest
import torch

import flag_gems
from flag_gems.ops.lu_unpack import lu_unpack_backward

from . import accuracy_utils as utils

# lu_factor only supports float32/float64 (half/bfloat16 are not supported).
LU_SHAPES = [
    (3, 3),
    (4, 4),
    (3, 4),
    (4, 3),
    (5, 5),
    (8, 8),
    (64, 64),
    (256, 128),
    (512, 512),
    (768, 768),
    (1024, 1024),
]
# torch.linalg.lu_factor only supports float32/float64; half/bfloat16 not supported.
LU_DTYPES = [torch.float32] + ([torch.float64] if utils.fp64_is_supported else [])


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack(shape, dtype):
    # Create a square or rectangular matrix and compute LU factorization
    m, n = shape
    k = min(m, n)
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    # Get LU factorization
    LU, pivots = torch.linalg.lu_factor(A)

    # cuSOLVER and LAPACK occasionally disagree at pivot-decision points, so
    # independent factorizations of the same matrix can differ structurally.
    # lu_unpack is deterministic given (LU, pivots), so the reference unpacks
    # the same packed factorization.
    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)

    # Unpack using FlagGems
    P, L, U = flag_gems.lu_unpack(LU, pivots)

    # Unpack using reference
    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    # Compare all three outputs
    # LU factorization has accumulated numerical error proportional to k
    utils.gems_assert_close(P, ref_P, dtype)
    utils.gems_assert_close(L, ref_L, dtype, reduce_dim=k)
    utils.gems_assert_close(U, ref_U, dtype, reduce_dim=k)


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_unpack_data_false(shape, dtype):
    # Test with unpack_data=False
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    LU, pivots = torch.linalg.lu_factor(A)

    # cuSOLVER and LAPACK occasionally disagree at pivot-decision points, so
    # independent factorizations of the same matrix can differ structurally.
    # lu_unpack is deterministic given (LU, pivots), so the reference unpacks
    # the same packed factorization.
    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)

    P, L, U = flag_gems.lu_unpack(LU, pivots, unpack_data=False)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(
        ref_LU, ref_pivots, unpack_data=False
    )

    utils.gems_assert_close(P, ref_P, dtype)
    # L and U should be empty tensors
    assert L.numel() == 0
    assert U.numel() == 0


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_unpack_pivots_false(shape, dtype):
    # Test with unpack_pivots=False
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    LU, pivots = torch.linalg.lu_factor(A)

    # cuSOLVER and LAPACK occasionally disagree at pivot-decision points, so
    # independent factorizations of the same matrix can differ structurally.
    # lu_unpack is deterministic given (LU, pivots), so the reference unpacks
    # the same packed factorization.
    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)

    P, L, U = flag_gems.lu_unpack(LU, pivots, unpack_pivots=False)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(
        ref_LU, ref_pivots, unpack_pivots=False
    )

    # P should be empty
    assert P.numel() == 0
    utils.gems_assert_close(L, ref_L, dtype, reduce_dim=min(m, n))
    utils.gems_assert_close(U, ref_U, dtype, reduce_dim=min(m, n))


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_batched(shape, dtype):
    # Test with batched input
    batch_size = 4
    m, n = shape
    k = min(m, n)
    A = torch.randn(batch_size, m, n, dtype=dtype, device=flag_gems.device)

    LU, pivots = torch.linalg.lu_factor(A)

    # cuSOLVER and LAPACK occasionally disagree at pivot-decision points, so
    # independent factorizations of the same matrix can differ structurally.
    # lu_unpack is deterministic given (LU, pivots), so the reference unpacks
    # the same packed factorization.
    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)

    P, L, U = flag_gems.lu_unpack(LU, pivots)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    utils.gems_assert_close(P, ref_P, dtype)
    utils.gems_assert_close(L, ref_L, dtype, reduce_dim=k)
    utils.gems_assert_close(U, ref_U, dtype, reduce_dim=k)


@pytest.mark.lu_unpack
@pytest.mark.lu_unpack_out
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_out(shape, dtype):
    # Test lu_unpack.out with pre-allocated output tensors
    m, n = shape
    k = min(m, n)
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    LU, pivots = torch.linalg.lu_factor(A)

    # cuSOLVER and LAPACK occasionally disagree at pivot-decision points, so
    # independent factorizations of the same matrix can differ structurally.
    # lu_unpack is deterministic given (LU, pivots), so the reference unpacks
    # the same packed factorization.
    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)

    # Pre-allocate output tensors
    P_out = torch.empty(m, m, dtype=dtype, device=flag_gems.device)
    L_out = torch.empty(m, k, dtype=dtype, device=flag_gems.device)
    U_out = torch.empty(k, n, dtype=dtype, device=flag_gems.device)

    flag_gems.lu_unpack_out(LU, pivots, P=P_out, L=L_out, U=U_out)

    ref_P, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    utils.gems_assert_close(P_out, ref_P, dtype)
    utils.gems_assert_close(L_out, ref_L, dtype, reduce_dim=k)
    utils.gems_assert_close(U_out, ref_U, dtype, reduce_dim=k)


# ---------------------------------------------------------------------------
# Backward (`lu_unpack_backward`)
#
# `lu_unpack` is differentiable only through L and U (`output_differentiability:
# [False, True, True]`): P and the integer pivots never receive a gradient.  The
# packed LU gradient is built from grad_L (m, k) and grad_U (k, n), k = min(m, n)
# by `lu_unpack_backward` in torch/csrc/autograd/FunctionsManual.cpp.
# ---------------------------------------------------------------------------
LU_BACKWARD_SHAPES = [
    (3, 3),
    (4, 4),
    (3, 4),
    (4, 3),
    (5, 5),
    (8, 8),
    (64, 64),
    (256, 128),
    (512, 512),
    (1024, 1024),
]
LU_BACKWARD_GRADS = ["both", "L", "U"]


def _gems_lu_unpack_autograd(LU_data, LU_pivots, **kwargs):
    """The real differentiable entry: `torch.lu_unpack` under Gems dispatch."""
    return flag_gems.lu_unpack(LU_data, LU_pivots, **kwargs)


def _reference_lu_unpack_backward(grad_L, grad_U, m, n):
    """Independent reimplementation of the ATen composite backward.

    Written directly from `lu_unpack_backward` in FunctionsManual.cpp so the
    FlagGems kernel is compared against the documented formula, not against the
    same kernel twice.
    """
    k = min(m, n)

    def get_L1(L):
        return L.tril(-1) if m == k else L[..., :k, :].tril(-1)

    def get_L2(L):
        return L[..., k:, :]

    def get_U1(U):
        return U.triu() if n == k else U[..., :, :k].triu()

    def get_U2(U):
        return U[..., :, k:]

    if grad_L is not None:
        if grad_U is not None:
            if m == n:
                return grad_L.tril(-1) + grad_U.triu()
            a1 = get_L1(grad_L) + get_U1(grad_U)
            if m > n:
                return torch.cat((a1, get_L2(grad_L)), dim=-2)
            return torch.cat((a1, get_U2(grad_U)), dim=-1)
        if m >= n:
            return grad_L.tril(-1)
        pad = torch.zeros(
            *grad_L.shape[:-1], n - m, dtype=grad_L.dtype, device=grad_L.device
        )
        return torch.cat((grad_L.tril(-1), pad), dim=-1)
    if n >= m:
        return grad_U.triu()
    pad = torch.zeros(
        *grad_U.shape[:-2],
        m - n,
        grad_U.shape[-1],
        dtype=grad_U.dtype,
        device=grad_U.device,
    )
    return torch.cat((grad_U.triu(), pad), dim=-2)


def _make_grads(grads, L, U, reference=False):
    """Draw the upstream gradients once and map them to the reference dtype."""
    grad_L = torch.randn_like(L)
    grad_U = torch.randn_like(U)
    if reference:
        grad_L = utils.to_reference(grad_L)
        grad_U = utils.to_reference(grad_U)
    return (
        grad_L if grads in ("both", "L") else None,
        grad_U if grads in ("both", "U") else None,
    )


@pytest.mark.lu_unpack_backward
@pytest.mark.parametrize("shape", LU_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
@pytest.mark.parametrize("grads", LU_BACKWARD_GRADS)
def test_lu_unpack_backward(shape, dtype, grads):
    """End-to-end autograd: FlagGems `lu_unpack` vs native `aten::lu_unpack`."""
    m, n = shape
    k = min(m, n)
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    LU, pivots = torch.linalg.lu_factor(A)

    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)
    res_LU = LU.clone().requires_grad_(True)
    ref_LU = ref_LU.clone().requires_grad_(True)

    _, res_L, res_U = _gems_lu_unpack_autograd(res_LU, pivots)
    _, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    res_grad_L, res_grad_U = _make_grads(grads, res_L, res_U)
    ref_grad_L = utils.to_reference(res_grad_L) if res_grad_L is not None else None
    ref_grad_U = utils.to_reference(res_grad_U) if res_grad_U is not None else None

    of_gems = tuple(
        x for x, g in ((res_L, res_grad_L), (res_U, res_grad_U)) if g is not None
    )
    of_ref = tuple(
        x for x, g in ((ref_L, ref_grad_L), (ref_U, ref_grad_U)) if g is not None
    )
    gf_gems = tuple(g for g in (res_grad_L, res_grad_U) if g is not None)
    gf_ref = tuple(g for g in (ref_grad_L, ref_grad_U) if g is not None)

    (res_grad,) = torch.autograd.grad(of_gems, (res_LU,), gf_gems)
    (ref_grad,) = torch.autograd.grad(of_ref, (ref_LU,), gf_ref)

    assert res_grad.shape == (m, n)
    assert res_grad.dtype == dtype
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=k)


@pytest.mark.lu_unpack_backward
@pytest.mark.parametrize("shape", LU_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
@pytest.mark.parametrize("grads", LU_BACKWARD_GRADS)
def test_lu_unpack_backward_entry(shape, dtype, grads):
    """Direct `flag_gems.lu_unpack_backward` against the documented formula."""
    m, n = shape
    k = min(m, n)
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    LU, pivots = torch.linalg.lu_factor(A)
    _, L, U = flag_gems.lu_unpack(LU, pivots)

    grad_L, grad_U = _make_grads(grads, L, U)
    ref_grad_L = utils.to_reference(grad_L, True) if grad_L is not None else None
    ref_grad_U = utils.to_reference(grad_U, True) if grad_U is not None else None

    res_grad = lu_unpack_backward(grad_L, grad_U, m, n)
    ref_grad = _reference_lu_unpack_backward(ref_grad_L, ref_grad_U, m, n)

    assert res_grad.shape == (m, n)
    assert res_grad.dtype == dtype
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=k)


@pytest.mark.lu_unpack_backward
@pytest.mark.parametrize("shape", LU_BACKWARD_SHAPES[:4])
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_backward_batched(shape, dtype):
    """Batched packed gradients keep the leading batch dimensions."""
    batch = 3
    m, n = shape
    k = min(m, n)
    A = torch.randn(batch, m, n, dtype=dtype, device=flag_gems.device)
    LU, pivots = torch.linalg.lu_factor(A)

    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)
    res_LU = LU.clone().requires_grad_(True)
    ref_LU = ref_LU.clone().requires_grad_(True)

    _, res_L, res_U = _gems_lu_unpack_autograd(res_LU, pivots)
    _, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)

    res_grad_L, res_grad_U = _make_grads("both", res_L, res_U)
    ref_grad_L = utils.to_reference(res_grad_L)
    ref_grad_U = utils.to_reference(res_grad_U)

    (res_grad,) = torch.autograd.grad(
        (res_L, res_U), (res_LU,), (res_grad_L, res_grad_U)
    )
    (ref_grad,) = torch.autograd.grad(
        (ref_L, ref_U), (ref_LU,), (ref_grad_L, ref_grad_U)
    )

    assert res_grad.shape == (batch, m, n)
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=k)


@pytest.mark.lu_unpack_backward
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_backward_nondifferentiable_outputs(dtype):
    """P and the integer pivots are non-differentiable, exactly like ATen."""
    m, n = 8, 6
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    LU, pivots = torch.linalg.lu_factor(A)
    LU = LU.clone().requires_grad_(True)

    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)
    ref_P, _, _ = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)
    P, L, U = _gems_lu_unpack_autograd(LU, pivots)

    assert not pivots.requires_grad
    assert not P.requires_grad and P.grad_fn is None
    assert ref_P.grad_fn is None
    assert L.requires_grad and U.requires_grad

    with pytest.raises(RuntimeError):
        torch.autograd.grad(P.sum(), (LU,))

    # unpack_data=False still returns the (empty) differentiable outputs. A
    # backward through them yields no packed gradient instead of fabricating
    # one; ATen 2.8 raises "triu: input tensor must have at least 2 dimensions"
    # on this degenerate path, which FlagGems handles gracefully.
    P_empty, L_empty, U_empty = _gems_lu_unpack_autograd(LU, pivots, unpack_data=False)
    assert P_empty.shape == (m, m)
    assert L_empty.numel() == 0 and U_empty.numel() == 0
    if L_empty.requires_grad:
        empty_grads = (
            torch.empty_like(L_empty, requires_grad=False),
            torch.empty_like(U_empty, requires_grad=False),
        )
        assert (
            torch.autograd.grad(
                (L_empty, U_empty), (LU,), empty_grads, allow_unused=True
            )[0]
            is None
        )


@pytest.mark.lu_unpack_backward
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_backward_dispatch(dtype):
    """`torch.lu_unpack` under `use_gems()` runs the FlagGems backward."""
    m, n = 32, 16
    k = min(m, n)
    A = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    LU, pivots = torch.linalg.lu_factor(A)

    ref_LU = utils.to_reference(LU)
    ref_pivots = utils.to_reference(pivots)
    ref_LU = ref_LU.clone().requires_grad_(True)
    _, ref_L, ref_U = torch.ops.aten.lu_unpack(ref_LU, ref_pivots)
    ref_grad_L = utils.to_reference(torch.randn_like(ref_L))
    ref_grad_U = utils.to_reference(torch.randn_like(ref_U))
    (ref_grad,) = torch.autograd.grad(
        (ref_L, ref_U), (ref_LU,), (ref_grad_L, ref_grad_U)
    )

    LU = LU.clone().requires_grad_(True)
    P, L, U = flag_gems.lu_unpack(LU, pivots)
    grad_L = ref_grad_L.to(dtype=dtype, device=flag_gems.device)
    grad_U = ref_grad_U.to(dtype=dtype, device=flag_gems.device)
    (res_grad,) = torch.autograd.grad((L, U), (LU,), (grad_L, grad_U))

    assert "LuUnpackFunction" in type(L.grad_fn).__name__
    assert P.grad_fn is None
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=k)
