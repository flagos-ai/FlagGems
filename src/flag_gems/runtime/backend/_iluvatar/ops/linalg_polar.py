"""Right polar decomposition A = U @ H via a Newton-Schulz iteration on A.

The whole numerical result is produced by one Triton kernel per batch element:
  X0 = A / ||A||_F                      (spectral norm of X0 <= 1)
  X_{k+1} = 0.5 * X_k * (3I - X_k^H X_k) converging to the orthogonal polar
            factor U (all singular values are driven to 1)
  H = sym(U^H A)                        (Hermitian PSD square root of A^H A)

Working in A-space (instead of forming the Gram matrix A^H A) keeps the
relative accuracy of the small singular values, so exactly rank-deficient
inputs still get a fully orthogonal U, matching torch.linalg.polar.

The iteration is a serial dependency chain in a single CTA, so per-iteration
latency dominates. For the small tiles this operator sees, tl.dot costs a
shared-memory round trip for its operand layout and forces a 16x16 minimum
tile even when n is 2. The DOT=False path therefore evaluates the same three
matrix products as in-register broadcast-and-reduce products at the natural
power-of-two tile size, which keeps every operand in registers and removes the
padding waste. Larger matrices keep the tensor-core path.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _polar_ns_kernel(
    A_ptr,
    U_ptr,
    H_ptr,
    m,
    n,
    sA_m,
    sA_n,
    s0,
    d0,
    s1,
    d1,
    s2,
    d2,
    s3,
    d3,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NBDIM: tl.constexpr,
    MAX_IT: tl.constexpr,
    EPS: tl.constexpr,
    DOT: tl.constexpr,
):
    b = tl.program_id(0)

    # ---- flat batch index -> element offset inside A (arbitrary strides) ----
    rem = b
    off = 0
    if NBDIM > 0:
        i0 = rem // d0
        rem = rem - i0 * d0
        off += i0 * s0
    if NBDIM > 1:
        i1 = rem // d1
        rem = rem - i1 * d1
        off += i1 * s1
    if NBDIM > 2:
        i2 = rem // d2
        rem = rem - i2 * d2
        off += i2 * s2
    if NBDIM > 3:
        i3 = rem // d3
        rem = rem - i3 * d3
        off += i3 * s3

    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mask_a = (rm[:, None] < m) & (rn[None, :] < n)
    a_ptrs = A_ptr + off + rm[:, None] * sA_m + rn[None, :] * sA_n
    A32 = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)

    # ---- Frobenius scaling so that sigma_max(X0) <= 1 ----
    fro2 = tl.sum(tl.sum(A32 * A32, axis=1), axis=0)
    scale = tl.maximum(tl.sqrt(fro2), 1e-30)
    X = A32 / scale

    # Identity restricted to the valid n x n block: the zero padding columns
    # (j >= n) then contribute exactly 0 to eye - P (their rows/columns of P are
    # zero), so the convergence residual needs no mask or select, and the
    # padding columns of the iterate are kept exactly zero by the polynomial
    # M = (15I - 10P + 3P^2)/8 because their whole row of M is zero. Including
    # the padding on the diagonal instead would pin the residual at 1.0 and
    # force every call to run the full MAX_IT budget.
    eye = tl.where((rn[:, None] == rn[None, :]) & (rn[:, None] < n), 1.0, 0.0)

    # ---- polar iteration towards the orthogonal factor ----
    # Third-order (quartic) Newton-Schulz step X <- X (15I - 10P + 3P^2) / 8
    # with P = X^H X.  Applied to a singular value sigma it gives
    # sigma <- sigma (15 - 10 s^2 + 3 s^4) / 8, a monotone map of [0,1] onto
    # itself with gain 15/8 = 1.875 near 0 (vs 1.5 for the plain step) and
    # cubic convergence 1 - delta -> 1 - 2.5 delta^3 near 1, so it needs
    # noticeably fewer iterations than the standard 2nd-order step for the
    # same residual.  It keeps the same fixed points (0 stays 0, 1 stays 1),
    # hence the same rank-deficient behaviour and the same orthogonality.
    res = 1e30
    it = 0
    while (it < MAX_IT) & (res > EPS):
        if DOT:
            P = tl.dot(tl.trans(X), X, input_precision="ieee")
        else:
            P = tl.sum(tl.expand_dims(X, 2) * tl.expand_dims(X, 1), axis=0)
        res = tl.max(tl.abs(eye - P))
        if DOT:
            PP = tl.dot(P, P, input_precision="ieee")
        else:
            PP = tl.sum(tl.expand_dims(P, 1) * tl.expand_dims(P, 0), axis=2)
        M = 0.125 * (15.0 * eye - 10.0 * P + 3.0 * PP)
        if DOT:
            X = tl.dot(X, M, input_precision="ieee")
        else:
            X = tl.sum(tl.expand_dims(X, 2) * tl.expand_dims(M, 0), axis=1)
        it += 1

    # ---- H = sym(U^H A) ----
    if DOT:
        Hm = tl.dot(tl.trans(X), A32, input_precision="ieee")
    else:
        Hm = tl.sum(tl.expand_dims(X, 2) * tl.expand_dims(A32, 1), axis=0)
    Hm = 0.5 * (Hm + tl.trans(Hm))

    # ---- stores (U and H are freshly allocated, contiguous) ----
    mask_h = (rn[:, None] < n) & (rn[None, :] < n)
    u_ptrs = U_ptr + b * (m * n) + rm[:, None] * n + rn[None, :]
    tl.store(u_ptrs, X.to(U_ptr.dtype.element_ty), mask=mask_a)
    h_ptrs = H_ptr + b * (n * n) + rn[:, None] * n + rn[None, :]
    tl.store(h_ptrs, Hm.to(H_ptr.dtype.element_ty), mask=mask_h)


_MAX_IT = 64
_EPS = 1e-6
# Upper bound on the elements of one broadcast product tile (BM*BN*BN) for the
# in-register path; above it the tensor-core dot path is used instead.
_SMALL_TILE = 512
_POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def _next_pow2(v):
    for p in _POW2:
        if p >= v:
            return p
    return 1 << (int(v - 1).bit_length())


def _check(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError(
            "linalg_polar: Expected a Tensor as input, got %s" % type(A).__name__
        )
    if A.dim() < 2:
        raise RuntimeError(
            "linalg_polar: The input tensor A must have at least 2 dimensions, "
            "but it has %d." % A.dim()
        )
    if A.dtype != torch.float32:
        raise RuntimeError(
            "linalg_polar: Expected a float32 tensor as input, got %s" % str(A.dtype)
        )
    if A.shape[-2] < A.shape[-1]:
        raise RuntimeError(
            "linalg_polar: Expected A.shape[-2] >= A.shape[-1], got A of shape %s"
            % (tuple(A.shape),)
        )


def linalg_polar(A):
    logger.debug("GEMS_ILUVATAR LINALG_POLAR")
    if A.dtype not in (torch.float32, torch.float64):
        from flag_gems.ops.linalg_polar import linalg_polar as _generic

        return _generic(A)

    _check(A)

    m = int(A.shape[-2])
    n = int(A.shape[-1])
    lead = tuple(int(v) for v in A.shape[:-2])
    hshape = lead + (n, n)

    if A.numel() == 0 or n == 0 or m == 0:
        U = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        if m == 0 and n > 0:
            H = torch.zeros(hshape, dtype=A.dtype, device=A.device)
        else:
            H = torch.empty(hshape, dtype=A.dtype, device=A.device)
        return U, H

    B = 1
    for v in lead:
        B *= v

    U = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    H = torch.empty(hshape, dtype=A.dtype, device=A.device)

    lstr = tuple(int(s) for s in A.stride()[:-2])
    nbd = len(lead)
    divs = [1] * 4
    for k in range(min(nbd, 4)):
        p = 1
        for q in range(k + 1, nbd):
            p *= lead[q]
        divs[k] = p
    strides = list(lstr[:4]) + [0] * (4 - min(nbd, 4))

    bm = _next_pow2(m)
    bn = _next_pow2(n)
    dot = bm * bn * bn > _SMALL_TILE
    if dot:
        BM = max(16, bm)
        BN = max(16, bn)
        # one warp keeps the residual reduction intra-warp; extra warps only
        # add cross-warp reduction traffic for a tile this small
        num_warps = 1 if BM * BN <= 256 else (4 if BM * BN <= 4096 else 8)
    else:
        BM = bm
        BN = bn
        # keep roughly <= 32 tile elements per lane so nothing spills
        num_warps = max(1, min(8, (BM * BN * BN + 1023) // 1024))

    _polar_ns_kernel[(B,)](
        A,
        U,
        H,
        m,
        n,
        A.stride(-2),
        A.stride(-1),
        strides[0],
        divs[0],
        strides[1],
        divs[1],
        strides[2],
        divs[2],
        strides[3],
        divs[3],
        BM=BM,
        BN=BN,
        NBDIM=nbd,
        MAX_IT=_MAX_IT,
        EPS=_EPS,
        DOT=dot,
        num_warps=num_warps,
    )
    return U, H
