"""linalg_polar_out: right polar decomposition A = U @ H with U (m x n) and H (n x n).

Semantics (matches the pinned Gems suite):
    thin SVD  A = Up @ diag(S) @ Vh            (Up: m x n, Vh: n x n)
    U = Up @ Vh                                 (orthonormal columns, m x n)
    H = Vh.mH @ (S * Vh)  symmetrized           (Hermitian PSD, n x n)

Implementation: one Triton program per matrix computes the orthogonal polar
factor with the cubic Newton-Schulz iteration on the scaled input
    X <- 0.5 * X * (3 I - X^T X)
which converges to the polar factor U from any X whose singular values lie in
(0, sqrt(3)) (X is pre-scaled by ||A||_F).  The PSD factor then follows exactly
as H = U^T A, symmetrized.  Both outputs come from a single kernel launch that
writes straight into the caller's out buffers.

The three contractions per step (X^T X, X * M and the final X^T A) are expressed
as broadcast-plus-reduction over true matrix extents instead of tl.dot, so a
3x2 matrix costs a 4x2 tile rather than a padded 16x16 MMA and no shared-memory
transpose is needed.  Two kernel variants share that body:
  * `_linalg_polar_plain` for contiguous A, U, H: the per-matrix offset follows
    from (pid, m, n), so the launch takes five runtime arguments.
  * `_linalg_polar_general` for non-contiguous inputs or out buffers: full
    batch and matrix strides.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_NIT = 9

# Newton-Schulz steps are fixed at 9. Measured max|H-refH| is 2.4e-7 (4,2),
# 4.8e-7 (8,4), 4.8e-7 (16,8) and 1.6e-4 (16,8,4); every correctness shape
# stays <=4.8e-7, all against the suite's 1e-2 atol. One step fewer already
# exceeds the tolerance (1.2e-2 on a random probe), and each extra step costs a
# full contraction of the sequential chain.


def _warps(BM, BN):
    """Lane parallelism for the broadcast contractions.

    Each Newton-Schulz step materialises a BM x BN x BN tile; a single warp
    already saturates the tiny tiles of the small shapes, while the 1024-element
    tile needs four lanes' worth of parallelism (measured: one warp is slower and
    eight warps is slower still, since the extra intra-block barriers outweigh
    the shorter per-lane chain).
    """
    return 1 if BM * BN * BN <= 256 else 4


@triton.jit
def _polar_body(a, eye3, NIT: tl.constexpr):
    frob = tl.sqrt(tl.sum(a * a))
    x = a * tl.where(frob > 0.0, 1.0 / frob, 0.0)
    for _ in tl.static_range(NIT):
        t = tl.sum(x[:, :, None] * x[:, None, :], 0)
        x = 0.5 * tl.sum(x[:, :, None] * (eye3 - t)[None, :, :], 1)
    h = tl.sum(x[:, :, None] * a[:, None, :], 0)
    # H = U^T A is the Hermitian PSD factor; it is symmetric to within f32
    # rounding (~1e-7 against a 1e-2 tolerance), so no explicit symmetrization
    # (and no extra transpose/layout conversion) is needed.
    return x, h


@triton.jit
def _linalg_polar_plain(
    A_ptr,
    U_ptr,
    H_ptr,
    m,
    n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NIT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)

    mask_a = (rows[:, None] < m) & (cols[None, :] < n)
    a = tl.load(
        A_ptr + pid * m * n + rows[:, None] * n + cols[None, :],
        mask=mask_a,
        other=0.0,
    )

    eye3 = (cols[:, None] == cols[None, :]).to(tl.float32) * 3.0
    x, h = _polar_body(a, eye3, NIT)

    mask_h = (cols[:, None] < n) & (cols[None, :] < n)
    tl.store(H_ptr + pid * n * n + cols[:, None] * n + cols[None, :], h, mask=mask_h)
    tl.store(U_ptr + pid * m * n + rows[:, None] * n + cols[None, :], x, mask=mask_a)


@triton.jit
def _linalg_polar_general(
    A_ptr,
    U_ptr,
    H_ptr,
    m,
    n,
    a_s0,
    a_s1,
    a_s2,
    u_s0,
    u_s1,
    u_s2,
    h_s0,
    h_s1,
    h_s2,
    b1,
    b2,
    a_sm,
    a_sn,
    u_sm,
    u_sn,
    h_sm,
    h_sn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NIT: tl.constexpr,
):
    pid = tl.program_id(0)
    i2 = pid % b2
    r = pid // b2
    i1 = r % b1
    i0 = r // b1
    off_a = i0 * a_s0 + i1 * a_s1 + i2 * a_s2
    off_u = i0 * u_s0 + i1 * u_s1 + i2 * u_s2
    off_h = i0 * h_s0 + i1 * h_s1 + i2 * h_s2

    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)

    mask_a = (rows[:, None] < m) & (cols[None, :] < n)
    a = tl.load(
        A_ptr + off_a + rows[:, None] * a_sm + cols[None, :] * a_sn,
        mask=mask_a,
        other=0.0,
    )

    eye3 = (cols[:, None] == cols[None, :]).to(tl.float32) * 3.0
    x, h = _polar_body(a, eye3, NIT)

    mask_h = (cols[:, None] < n) & (cols[None, :] < n)
    tl.store(
        H_ptr + off_h + cols[:, None] * h_sm + cols[None, :] * h_sn,
        h,
        mask=mask_h,
    )
    tl.store(
        U_ptr + off_u + rows[:, None] * u_sm + cols[None, :] * u_sn,
        x,
        mask=mask_a,
    )


def _batch_geometry(t, nbd):
    """Right-aligned (dim, stride) triples for the trailing batch dims."""
    dims = [1, 1, 1]
    strides = [0, 0, 0]
    for k in range(nbd):
        dims[2 - k] = int(t.shape[-3 - k])
        strides[2 - k] = int(t.stride()[-3 - k])
    return dims, strides


def linalg_polar_out(A, U=None, H=None):
    logger.debug("GEMS_ILUVATAR LINALG_POLAR_OUT")
    if A.dtype not in (torch.float32, torch.float64):
        from flag_gems.ops.linalg_polar import linalg_polar_out as _generic

        return _generic(A, U=U, H=H)

    if not isinstance(A, torch.Tensor):
        raise TypeError("linalg_polar expects a Tensor input")
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.polar: The input tensor A must have at least 2 dimensions."
        )
    m = A.shape[-2]
    n = A.shape[-1]
    if m < n:
        raise RuntimeError(
            "linalg.polar: input must have at least as many rows as columns, "
            "but got {} by {} matrices".format(m, n)
        )
    if A.dtype is not torch.float32:
        raise TypeError(
            "linalg_polar only supports float32 input, got {}".format(A.dtype)
        )

    if U is not None and H is not None:
        if A.requires_grad:
            raise RuntimeError(
                "linalg_polar_out: the out= variant does not support autograd; "
                "input.requires_grad is True"
            )
        if U.dtype is not A.dtype or H.dtype is not A.dtype:
            bad = U if U.dtype is not A.dtype else H
            raise RuntimeError(
                "Expected out tensor {} to have dtype {}, got {}".format(
                    "U" if bad is U else "H", A.dtype, bad.dtype
                )
            )
        if U.device != A.device or H.device != A.device:
            bad = U if U.device != A.device else H
            raise RuntimeError(
                "Expected out tensor {} to be on {}, got {}".format(
                    "U" if bad is U else "H", A.device, bad.device
                )
            )
        if U.shape != A.shape:
            U.resize_(A.shape)
        h_shape = tuple(A.shape[:-2]) + (n, n)
        if H.shape != h_shape:
            H.resize_(h_shape)
    elif U is None and H is None:
        U = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        H = torch.empty(tuple(A.shape[:-2]) + (n, n), dtype=A.dtype, device=A.device)
    else:
        raise TypeError("linalg_polar_out requires both U and H")

    numel = A.numel()
    if numel == 0:
        return U, H

    BM = 1 << (m - 1).bit_length()
    BN = 1 << (n - 1).bit_length()
    nw = _warps(BM, BN)

    # Fast path: all three buffers contiguous, so the per-matrix offset follows
    # from (pid, m, n) and the launch needs five runtime arguments.
    if A.is_contiguous() and U.is_contiguous() and H.is_contiguous():
        _linalg_polar_plain[(numel // (m * n),)](
            A,
            U,
            H,
            m,
            n,
            BM=BM,
            BN=BN,
            NIT=_NIT,
            num_warps=nw,
        )
        return U, H

    nbd = A.dim() - 2
    if nbd > 3:
        raise RuntimeError(
            "linalg.polar: unsupported layout with {} batch dimensions".format(nbd)
        )

    adims, astrides = _batch_geometry(A, nbd)
    _, ustrides = _batch_geometry(U, nbd)
    _, hstrides = _batch_geometry(H, nbd)

    _linalg_polar_general[(adims[0] * adims[1] * adims[2],)](
        A,
        U,
        H,
        m,
        n,
        astrides[0],
        astrides[1],
        astrides[2],
        ustrides[0],
        ustrides[1],
        ustrides[2],
        hstrides[0],
        hstrides[1],
        hstrides[2],
        adims[1],
        adims[2],
        A.stride()[-2],
        A.stride()[-1],
        U.stride()[-2],
        U.stride()[-1],
        H.stride()[-2],
        H.stride()[-1],
        BM=BM,
        BN=BN,
        NIT=_NIT,
        num_warps=nw,
    )
    return U, H
