import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for linalg_tensorinv tests
# Each shape must satisfy prod(shape[:ind]) == prod(shape[ind:])
TENSORINV_SHAPES_IND2 = [
    (4, 6, 8, 3),  # 4*6=24, 8*3=24 -> output (8, 3, 4, 6)
    (2, 8, 4, 4),  # 2*8=16, 4*4=16 -> output (4, 4, 2, 8)
    (3, 4, 6, 2),  # 3*4=12, 6*2=12 -> output (6, 2, 3, 4)
    (1, 1, 1, 1),  # 1*1=1, 1*1=1 -> output (1, 1, 1, 1)
]

TENSORINV_SHAPES_IND1 = [
    (2, 2),  # 2x2 matrix -> output (2, 2)
    (4, 4),  # 4x4 matrix -> output (4, 4)
    (8, 8),  # 8x8 matrix -> output (8, 8)
    (3, 3),  # 3x3 matrix -> output (3, 3)
]

# Only float32 and float16 are covered: float32 exercises PyTorch's native
# tensorinv reference, while float16 exercises the manual fp32 reference path.
TENSORINV_DTYPES = [torch.float32, torch.float16]


def _prod(dims):
    out = 1
    for d in dims:
        out *= d
    return out


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("shape", TENSORINV_SHAPES_IND2)
# torch.linalg.tensorinv requires a float32 reference path for lower precision inputs.
@pytest.mark.parametrize("dtype", TENSORINV_DTYPES)
def test_linalg_tensorinv_ind2(shape, dtype):
    """Test linalg_tensorinv with ind=2"""
    # Generate a random invertible matrix by using A = L @ L.T + I where L is random
    # This ensures the matrix is positive definite and invertible
    ind = 2
    m = shape[0] * shape[1]
    n = shape[2] * shape[3]
    assert m == n, f"Shape {shape} invalid for ind={ind}"

    # Create a random matrix and ensure it's well-conditioned
    # For float16, we need to compute in float32 and then convert
    if dtype == torch.float16:
        # Create in float32 and convert to float16
        A = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        A_flat = A.reshape(m, n)
        A_flat = (
            A_flat @ A_flat.T
            + torch.eye(m, dtype=torch.float32, device=flag_gems.device) * 0.1
        )
        A = A_flat.reshape(shape).to(torch.float16)
    else:
        A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Make it invertible by adding a large identity-like term
        A_flat = A.reshape(m, n)
        A_flat = (
            A_flat @ A_flat.T + torch.eye(m, dtype=dtype, device=flag_gems.device) * 0.1
        )
        A = A_flat.reshape(shape)

    ref_A = utils.to_reference(A)

    # Compute reference in float32 (since PyTorch's tensorinv doesn't support float16)
    if dtype == torch.float16:
        ref_A_fp32 = ref_A.to(torch.float32)
        ref_out = torch.linalg.tensorinv(ref_A_fp32, ind=ind).to(torch.float16)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("shape", TENSORINV_SHAPES_IND1)
# torch.linalg.tensorinv requires a float32 reference path for lower precision inputs.
@pytest.mark.parametrize("dtype", TENSORINV_DTYPES)
def test_linalg_tensorinv_ind1(shape, dtype):
    """Test linalg_tensorinv with ind=1 (equivalent to matrix inverse)"""
    ind = 1
    m = shape[0]
    n = shape[1]
    assert m == n, f"Shape {shape} must be square for ind={ind}"

    # Create a random invertible matrix
    # For float16, we need to compute in float32 and then convert
    if dtype == torch.float16:
        A = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        A = A @ A.T + torch.eye(m, dtype=torch.float32, device=flag_gems.device) * 0.1
        A = A.to(torch.float16)
    else:
        A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Make it invertible by adding a identity-like term
        A = A @ A.T + torch.eye(m, dtype=dtype, device=flag_gems.device) * 0.1

    ref_A = utils.to_reference(A)

    # Compute reference in float32 (since PyTorch's tensorinv doesn't support float16)
    if dtype == torch.float16:
        ref_A_fp32 = ref_A.to(torch.float32)
        ref_out = torch.linalg.tensorinv(ref_A_fp32, ind=ind).to(torch.float16)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_tensorinv
# Non-SPD inputs exercise partial pivoting.
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (8, 8), (16, 16), (4, 6, 8, 3)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorinv_non_spd(shape, dtype):
    """General (non symmetric / non positive-definite) invertible inputs."""
    if len(shape) == 2:
        ind = 1
    else:
        ind = 2
    m = 1
    for i in range(ind):
        m *= shape[i]
    n = 1
    for i in range(ind, len(shape)):
        n *= shape[i]
    assert m == n

    # Plain randn: general, non-SPD, invertible with high probability.  Scale
    # to keep cond(A) moderate so float32 reference comparison is meaningful.
    A = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    A = A.reshape(shape)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    # Looser tolerance: general randn matrices can be moderately conditioned.
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linalg_tensorinv_zero_diagonal_pivot(dtype):
    """A permutation matrix: the diagonal is zero but a valid pivot exists
    off the diagonal, so partial pivoting must still produce the correct
    inverse.
    """
    A = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=flag_gems.device)
    ind = 1

    ref_A = utils.to_reference(A)
    if dtype == torch.float16:
        ref_A_fp32 = ref_A.to(torch.float32)
        ref_out = torch.linalg.tensorinv(ref_A_fp32, ind=ind).to(torch.float16)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    assert not torch.isnan(res_out).any(), "tensorinv produced NaN on permutation"
    assert not torch.isinf(res_out).any(), "tensorinv produced Inf on permutation"
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv_blocked_path():
    """Exercise the blocked (> _TENSORINV_BLOCK_MAX=64) dispatch path with a
    non-SPD matrix, which both covers the larger-size code route and the
    partial-pivoting logic in the blocked kernel.
    """
    dtype = torch.float32
    n = 80  # > 64 -> blocked kernel
    A = torch.randn(n, n, dtype=dtype, device=flag_gems.device) * 2.0
    A = A @ A.mT + 0.1 * torch.eye(n, dtype=dtype, device=flag_gems.device)
    ind = 1

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorinv_ind3(dtype):
    """Cover ind=3 (prod(shape[:3]) == prod(shape[3:])), exercising the
    matrix-size computation for ind values beyond {1, 2}.
    """
    shape = (2, 2, 2, 8)  # prod(shape[:3]) = 8 == prod(shape[3:]) = 8
    ind = 3
    m = 1
    for i in range(ind):
        m *= shape[i]

    A = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    A = A.reshape(shape)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    assert tuple(res_out.shape) == shape[ind:] + shape[:ind]
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorinv_ind1_higher_dim(dtype):
    """ind=1 on a >2D input: only the first dim is the rows axis, the rest
    flatten to cols (prod must match). Covers the higher-dimensional reshape
    path.
    """
    shape = (8, 2, 2, 2)  # prod(shape[:1]) = 8 == prod(shape[1:]) = 8
    ind = 1
    m = shape[0]

    A = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    A = A.reshape(shape)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    assert tuple(res_out.shape) == shape[ind:] + shape[:ind]
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize(
    "A_cpu",
    [
        torch.zeros(4, 4),  # exact-zero pivot -> rank 0
        torch.tensor([[1.0, 2.0], [2.0, 4.0]]),  # exact-zero pivot -> rank 1
    ],
)
def test_linalg_tensorinv_singular(A_cpu):
    """A singular input with an exact-zero pivot must return inf/nan, not a
    finite wrong inverse. (With partial pivoting a zero pivot means the whole
    remaining column is zero.) Only structurally-exact zero pivots are
    asserted; tiny-but-nonzero pivots from round-off are a tolerance question
    outside this kernel's scope.
    """
    A = A_cpu.to(flag_gems.device)
    res_out = flag_gems.linalg_tensorinv(A, ind=1)

    assert torch.isnan(res_out).any() or torch.isinf(res_out).any(), (
        "tensorinv on a singular (exact-zero-pivot) matrix must return "
        "inf/nan, not a finite wrong inverse"
    )


@pytest.mark.linalg_tensorinv_out
@pytest.mark.parametrize("shape", TENSORINV_SHAPES_IND2)
# torch.linalg.tensorinv requires a float32 reference path for lower precision inputs.
@pytest.mark.parametrize("dtype", TENSORINV_DTYPES)
def test_linalg_tensorinv_out(shape, dtype):
    """Out-of-place variant: the result is written into the provided out
    tensor and returned from the call.
    """
    ind = 2
    m = shape[0] * shape[1]
    n = shape[2] * shape[3]
    assert m == n, f"Shape {shape} invalid for ind={ind}"

    if dtype == torch.float16:
        A = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        A_flat = A.reshape(m, n)
        A_flat = (
            A_flat @ A_flat.T
            + torch.eye(m, dtype=torch.float32, device=flag_gems.device) * 0.1
        )
        A = A_flat.reshape(shape).to(torch.float16)
    else:
        A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        A_flat = A.reshape(m, n)
        A_flat = (
            A_flat @ A_flat.T + torch.eye(m, dtype=dtype, device=flag_gems.device) * 0.1
        )
        A = A_flat.reshape(shape)

    ref_A = utils.to_reference(A)
    if dtype == torch.float16:
        ref_A_fp32 = ref_A.to(torch.float32)
        ref_out = torch.linalg.tensorinv(ref_A_fp32, ind=ind).to(torch.float16)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    out = torch.empty(
        A.shape[ind:] + A.shape[:ind], dtype=dtype, device=flag_gems.device
    )
    res_out = flag_gems.linalg_tensorinv(A, ind=ind, out=out)

    assert res_out is out, "linalg_tensorinv(out=) must return the provided out tensor"
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize(
    "ind",
    [0, -1, -2],
)
def test_linalg_tensorinv_non_positive_ind(ind):
    """A non-positive ind must raise, matching torch.linalg.tensorinv.

    Regression guard: with the check ordered after the shape checks, ind=0
    passed validation on a 1x1 input (where prod(shape[:0]) == prod(shape[0:])
    holds trivially) and returned a wrong result instead of an error.
    """
    A = torch.randn(1, 1, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.linalg.tensorinv(utils.to_reference(A), ind=ind)

    with pytest.raises(RuntimeError):
        flag_gems.linalg_tensorinv(A, ind=ind)


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv_invalid_input():
    """Invalid shapes must raise RuntimeError rather than produce a result."""
    A_1d = torch.randn(4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_tensorinv(A_1d, ind=2)

    # prod(shape[:2]) = 6 != prod(shape[2:]) = 4
    A_mismatch = torch.randn(2, 3, 2, 2, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_tensorinv(A_mismatch, ind=2)

    # ind > A.dim()
    with pytest.raises(RuntimeError):
        flag_gems.linalg_tensorinv(A_1d, ind=3)


@pytest.mark.linalg_tensorinv
@pytest.mark.parametrize("shape", [(2, 2), (8, 8), (4, 6, 8, 3)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_tensorinv_non_contiguous(shape, dtype):
    """Non-contiguous inputs must give the same result as contiguous ones:
    the implementation works on a contiguous copy, and losing that copy would
    silently produce a wrong result on strided input.
    """
    if len(shape) == 2:
        ind = 1
        m = shape[0]
    else:
        ind = 2
        m = shape[0] * shape[1]

    # Deliberately asymmetric: a symmetric generator (A @ A.T) would make the
    # transposed view numerically equal to its own transpose, so a stride bug
    # would go unnoticed.
    base = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    base = base + m * torch.eye(m, dtype=dtype, device=flag_gems.device)

    # A stride-2 slice of a half-empty buffer keeps `shape` unchanged (so the
    # prod(shape[:ind]) == prod(shape[ind:]) requirement still holds) while
    # being non-contiguous.
    buf = torch.empty(
        (shape[0] * 2,) + tuple(shape[1:]), dtype=dtype, device=flag_gems.device
    )
    buf[::2].copy_(base.reshape(shape))
    A_stride = buf[::2]

    assert tuple(A_stride.shape) == shape
    assert not A_stride.is_contiguous()

    # A dense transposed view: for a 2D square input `A.t()` keeps stride
    # (1, n), which survives both reshape and a plain clone(), so this is the
    # input that actually depends on the contiguous-copy guarantee.  For the
    # 4D case, transpose the last two dims so the row/col product split (and
    # hence prod(shape[:ind]) == prod(shape[ind:])) is preserved, while the
    # flattened stride still ends up transposed.
    A_dense = (
        base.reshape(shape).transpose(0, 1)
        if len(shape) == 2
        else base.reshape(shape).transpose(-1, -2)
    )
    assert not A_dense.is_contiguous()
    assert _prod(A_dense.shape[:ind]) == _prod(A_dense.shape[ind:])
    if len(shape) == 2:
        assert not torch.equal(A_dense, base.reshape(shape))

    # Contiguous inputs must remain correct too.
    for A_view in (A_stride, A_dense, base.reshape(shape)):
        ref_out = torch.linalg.tensorinv(utils.to_reference(A_view), ind=ind)
        res_out = flag_gems.linalg_tensorinv(A_view, ind=ind)

        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv_non_spd_blocked_path():
    """The blocked (>64) path with a general non-SPD matrix.

    The existing blocked-path test uses an SPD matrix, for which the
    column-maximum pivot is always the diagonal element, so its row-swap
    branch never runs. Plain randn has no such guarantee and exercises the
    swap logic in the blocked kernel.
    """
    dtype = torch.float32
    n = 80  # > _TENSORINV_BLOCK_MAX -> blocked kernel
    torch.manual_seed(42)
    A = torch.randn(n, n, dtype=dtype, device=flag_gems.device) * 2.0
    ind = 1

    ref_out = torch.linalg.tensorinv(utils.to_reference(A), ind=ind)
    res_out = flag_gems.linalg_tensorinv(A, ind=ind)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=1)


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv_blocked_path_singular():
    """A singular exact-zero-pivot matrix on the blocked path must propagate
    inf/nan, same as the register path.
    """
    dtype = torch.float32
    n = 80
    A = torch.zeros(n, n, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.linalg_tensorinv(A, ind=1)

    assert torch.isnan(res_out).any() or torch.isinf(res_out).any(), (
        "tensorinv on a singular matrix must return inf/nan on the blocked "
        "path too, not a finite wrong inverse"
    )


@pytest.mark.linalg_tensorinv
def test_linalg_tensorinv_input_not_mutated():
    """The in-place Gauss-Jordan works on a clone, so the caller's tensor must
    come back unchanged.
    """
    dtype = torch.float32
    A = torch.randn(8, 8, dtype=dtype, device=flag_gems.device) * 2.0
    A0 = A.clone()

    flag_gems.linalg_tensorinv(A, ind=1)

    assert torch.equal(A, A0), "linalg_tensorinv must not mutate its input"


@pytest.mark.linalg_tensorinv_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linalg_tensorinv_out_overwrites_and_non_contiguous(dtype):
    """out= must be fully overwritten (stale contents incl. inf/nan must not
    survive) and must accept a non-contiguous input.
    """
    ind = 1
    m = 4
    if dtype == torch.float16:
        A = (torch.randn(m, m, dtype=torch.float32, device=flag_gems.device) * 2.0).to(
            dtype
        )
    else:
        A = torch.randn(m, m, dtype=dtype, device=flag_gems.device) * 2.0
    A = A @ A.mT + 0.1 * torch.eye(m, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    if dtype == torch.float16:
        ref_out = torch.linalg.tensorinv(ref_A.to(torch.float32), ind=ind).to(dtype)
    else:
        ref_out = torch.linalg.tensorinv(ref_A, ind=ind)

    # Pre-fill with values that would be obviously wrong if left in place.
    out = torch.full((m, m), float("nan"), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.linalg_tensorinv(A, ind=ind, out=out)

    assert res_out is out
    assert not torch.isnan(out).any(), "out= must be fully overwritten"
    utils.gems_assert_close(out, ref_out, dtype, atol=1e-2, reduce_dim=1)

    # Non-contiguous input through the out= path.
    A_view = A.t()
    assert not A_view.is_contiguous()
    out2 = torch.empty_like(A)
    ref_view_A = utils.to_reference(A_view)
    if dtype == torch.float16:
        ref_view = torch.linalg.tensorinv(ref_view_A.to(torch.float32), ind=ind).to(
            dtype
        )
    else:
        ref_view = torch.linalg.tensorinv(ref_view_A, ind=ind)
    res_view = flag_gems.linalg_tensorinv(A_view, ind=ind, out=out2)

    assert res_view is out2
    utils.gems_assert_close(out2, ref_view, dtype, atol=1e-2, reduce_dim=1)
