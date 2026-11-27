"""Triton implementation of torch.linalg.svd (Gems entry point: main.py::run).

Algorithm: level-parallel one-sided (Hestenes) Jacobi on the columns of A.
All math is done inside @triton.jit kernels; the wrapper only allocates
outputs/scratch, reads metadata and launches.

Stage 1 (_svd_jacobi_kernel): one program per matrix, W = A (m x n) and the
accumulated rotation V (n x n) in registers.  Instead of touching one column
pair per step (which needs one block-wide reduction per pair), each level
processes a whole *matching* of disjoint column pairs at once: the partner map
is the classic round-robin (Brent-Luk / Latin square) over the odd domain
[0, BLOCK_N-1), where column k is paired with (s-k) mod (BLOCK_N-1) and the one
self-paired column of each level is paired with the last column.  Over
s = 0..BLOCK_N-2 every unordered pair is visited exactly once, so a sweep is
complete.  Per level the partner columns are fetched with tl.gather, alpha/beta/
gamma of all pairs cost three column reductions, and both partners are updated
together with one elementwise blend, so the reduction/exchange count per sweep
drops from O(n^2) to O(n).  The half-angle rotation is the numerically stable
form; sweeps stop early once max |gamma| / sqrt(alpha beta) < TOL.
Stage 2 (_svd_sort_kernel): selection-sorts sigma descending and gathers the
matching columns of U / rows of Vh into the (possibly batched) outputs.
Stage 3 (_svd_compl_kernel): only for full_matrices=True with m > n; completes
the columns of U to an orthonormal basis by modified Gram-Schmidt on standard
basis vectors (two re-orthogonalisation passes).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _batch_offset(pid, bt0, bs0, bt1, bs1, bt2, bs2, bt3, bs3):
    r = pid
    i3 = r % bt3
    r = r // bt3
    i2 = r % bt2
    r = r // bt2
    i1 = r % bt1
    r = r // bt1
    return i3 * bs3 + i2 * bs2 + i1 * bs1 + r * bs0


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["m", "n"],
)
@triton.jit
def _svd_jacobi_kernel(
    a_ptr,
    ur_ptr,
    vr_ptr,
    sr_ptr,
    bt0,
    bs0,
    bt1,
    bs1,
    bt2,
    bs2,
    bt3,
    bs3,
    m,
    n,
    nlev,
    stride_am,
    stride_an,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_SWEEPS: tl.constexpr,
    TOL: tl.constexpr,
):
    pid = tl.program_id(0)
    a_base = a_ptr + _batch_offset(pid, bt0, bs0, bt1, bs1, bt2, bs2, bt3, bs3)

    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    ndom: tl.constexpr = BLOCK_N - 1

    wmask = (rows[:, None] < m) & (cols[None, :] < n)
    w = tl.load(
        a_base + rows[:, None] * stride_am + cols[None, :] * stride_an,
        mask=wmask,
        other=0.0,
    ).to(tl.float32)

    vmask = (cols[:, None] < n) & (cols[None, :] < n)
    v = tl.where((cols[:, None] == cols[None, :]) & vmask, 1.0, 0.0)

    if BLOCK_N > 1:
        zero_rows = rows[:, None] * 0
        zero_cols = cols[:, None] * 0
        # partner map p = (s - cols) mod ndom, maintained incrementally so the
        # level loop carries no per-level integer division (modulo is emulated)
        pvec = tl.where(cols == 0, 0, ndom - cols)
        sweep = 0
        offmax = 1.0
        while (sweep < MAX_SWEEPS) & (offmax > TOL):
            offvec = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for s in range(0, nlev):
                k0 = (s * (ndom + 1) // 2) % ndom
                jvec = tl.where(cols == ndom, k0, pvec)
                jvec = tl.where(cols == k0, ndom, jvec)
                pvec = tl.where(pvec >= ndom - 1, 0, pvec + 1)
                idx = zero_rows + jvec[None, :]
                idxv = zero_cols + jvec[None, :]

                ws = tl.gather(w, idx, 1)
                # alpha and gamma from ONE cross-warp exchange (packed along a
                # trailing axis of size two), and beta taken as alpha permuted by
                # the partner map: beta[k] = ||w_jvec[k]||^2 = alpha[jvec[k]]
                # exactly, because ws is a pure column permutation of w.
                ag = tl.sum(tl.join(w * w, w * ws), axis=0)
                alpha, gamma = tl.split(ag)
                beta = tl.reshape(
                    tl.gather(tl.reshape(alpha, (1, BLOCK_N)), jvec[None, :], 1),
                    (BLOCK_N,),
                )

                nz = tl.abs(gamma) > 1e-30
                den = tl.where(nz, 2.0 * gamma, 1.0)
                tau = tl.where(nz, (beta - alpha) / den, 0.0)
                sg = tl.where(tau >= 0.0, 1.0, -1.0)
                t = tl.where(nz, sg / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
                c = 1.0 / tl.sqrt(1.0 + t * t)
                sv = c * t

                first = cols < jvec
                second = cols > jvec
                ca = tl.where(first | second, c, 1.0)
                # both partners of a pair use (c, -s): the partner tile supplies
                # the other column, so W'_k = c W_k - s W_partner(k) is the same
                # orthogonal rotation seen from either side.
                cb = tl.where(first | second, -sv, 0.0)

                ratio = tl.abs(gamma) / (tl.sqrt(alpha * beta) + 1e-30)
                offvec = tl.maximum(offvec, tl.where(first, ratio, 0.0))

                w = w * ca[None, :] + ws * cb[None, :]
                vs = tl.gather(v, idxv, 1)
                v = v * ca[None, :] + vs * cb[None, :]
            offmax = tl.max(offvec, axis=0)
            sweep += 1

    sig = tl.sqrt(tl.sum(w * w, axis=0))
    ok = (sig > 0.0) & (cols < n)
    inv = tl.where(ok, 1.0 / tl.where(ok, sig, 1.0), 0.0)
    uvals = w * inv[None, :]

    tl.store(
        ur_ptr + pid * m * n + rows[:, None] * n + cols[None, :], uvals, mask=wmask
    )
    tl.store(vr_ptr + pid * n * n + cols[:, None] * n + cols[None, :], v, mask=vmask)
    tl.store(sr_ptr + pid * n + cols, sig, mask=cols < n)


@triton.jit
def _svd_sort_kernel(
    ur_ptr,
    vr_ptr,
    sr_ptr,
    u_ptr,
    s_ptr,
    vh_ptr,
    m,
    n,
    k,
    u_per,
    s_per,
    vh_per,
    stride_um,
    stride_un,
    stride_vm,
    stride_vn,
    n_vh_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    sig = tl.load(sr_ptr + pid * n + cols, mask=cols < n, other=-1.0)
    sig = tl.where(cols < n, sig, -1.0)

    # descending rank (stable on equal sigmas by column index): one O(n^2)
    # comparison pass instead of k serial argmax reductions, then scatter each
    # source column straight to its sorted position.
    cmp = (sig[:, None] > sig[None, :]) | (
        (sig[:, None] == sig[None, :]) & (cols[:, None] < cols[None, :])
    )
    rank = tl.sum(cmp.to(tl.int32), axis=0)

    # all three source tiles are read into registers BEFORE anything is written,
    # so this kernel may permute the outputs in place (the wrapper passes the
    # output buffers as the source when k == n, which removes every scratch
    # allocation); the barrier makes that ordering explicit at block scope.
    uvals = tl.load(
        ur_ptr + pid * m * n + rows[:, None] * n + cols[None, :],
        mask=(rows[:, None] < m) & (cols[None, :] < n),
        other=0.0,
    )
    vhvals = tl.load(
        vr_ptr + pid * n * n + cols[:, None] * n + cols[None, :],
        mask=(cols[:, None] < n) & (cols[None, :] < n),
        other=0.0,
    )
    tl.debug_barrier()

    tl.store(s_ptr + pid * s_per + rank, sig, mask=(cols < n) & (rank < k))

    tl.store(
        u_ptr + pid * u_per + rows[:, None] * stride_um + rank[None, :] * stride_un,
        uvals,
        mask=(rows[:, None] < m) & (cols[None, :] < n) & (rank[None, :] < k),
    )
    tl.store(
        vh_ptr + pid * vh_per + rank[None, :] * stride_vm + cols[:, None] * stride_vn,
        vhvals,
        mask=(cols[:, None] < n) & (cols[None, :] < n) & (rank[None, :] < n_vh_rows),
    )


@triton.jit
def _svd_compl_kernel(
    u_ptr,
    m,
    n,
    u_per,
    stride_um,
    stride_un,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    base = u_ptr + pid * u_per
    rows = tl.arange(0, BLOCK_M)
    cc = tl.arange(0, BLOCK_N)

    u = tl.load(
        base + rows[:, None] * stride_um + cc[None, :] * stride_un,
        mask=(rows[:, None] < m) & (cc[None, :] < n),
        other=0.0,
    )

    nc = m - n
    for c in range(0, nc):
        v = tl.where(rows == (n + c), 1.0, 0.0)
        coef = tl.sum(u * v[:, None], axis=0)
        v = v - tl.sum(coef[None, :] * u, axis=1)
        coef = tl.sum(u * v[:, None], axis=0)
        v = v - tl.sum(coef[None, :] * u, axis=1)
        cp = tl.load(
            base + rows[:, None] * stride_um + (n + cc)[None, :] * stride_un,
            mask=(rows[:, None] < m) & (cc[None, :] < c),
            other=0.0,
        )
        coef2 = tl.sum(cp * v[:, None], axis=0)
        v = v - tl.sum(coef2[None, :] * cp, axis=1)
        coef2 = tl.sum(cp * v[:, None], axis=0)
        v = v - tl.sum(coef2[None, :] * cp, axis=1)
        nrm = tl.sqrt(tl.sum(v * v, axis=0))
        v = v / tl.where(nrm > 1e-8, nrm, 1.0)
        tl.store(base + rows * stride_um + (n + c) * stride_un, v, mask=rows < m)


def _num_warps(block_m, block_n):
    elems = block_m * block_n
    if elems <= 256:
        return 1
    if elems <= 1024:
        return 2
    if elems <= 4096:
        return 4
    return 8


def _pad4(vals, fill):
    out = list(vals) + [fill] * (4 - len(vals))
    return out


def linalg_svd(A, full_matrices=True, *, driver=None):
    logger.debug("GEMS_ILUVATAR LINALG_SVD")
    if A.dtype not in (torch.float32, torch.float64):
        from flag_gems.ops.linalg_svd import linalg_svd as _generic

        return _generic(A, full_matrices=full_matrices, driver=driver)

    m = int(A.shape[-2])
    n = int(A.shape[-1])
    batch = tuple(int(s) for s in A.shape[:-2])
    B = 1
    for sz in batch:
        B *= sz
    k = min(m, n)
    full = bool(full_matrices)
    dev = A.device
    dt = A.dtype

    u_shape = batch + ((m, m) if full else (m, k))
    vh_shape = batch + ((n, n) if full else (k, n))
    U = torch.empty(u_shape, device=dev, dtype=dt)
    S = torch.empty(batch + (k,), device=dev, dtype=dt)
    Vh = torch.empty(vh_shape, device=dev, dtype=dt)

    if B == 0 or m == 0 or n == 0:
        return U, S, Vh

    BLOCK_M = triton.next_power_of_2(m)
    BLOCK_N = triton.next_power_of_2(n)

    # When k == n the unsorted U (..., m, n), Vh (..., n, n) and S (..., n) fit
    # exactly in the output buffers, so the sort kernel permutes them in place
    # and no scratch allocation is needed at all.  Requires U's row stride to be
    # n, which rules out full_matrices with m > n (U is (m, m) there).
    inplace = (k == n) and (dt == torch.float32) and ((not full) or m == n)
    if inplace:
        ur, vr, sr = U, Vh, S
    else:
        ur = torch.empty((B, m, n), device=dev, dtype=torch.float32)
        vr = torch.empty((B, n, n), device=dev, dtype=torch.float32)
        sr = torch.empty((B, n), device=dev, dtype=torch.float32)

    if len(batch) == 0:
        bsz = (1, 1, 1, 1)
        bst = (0, 0, 0, 0)
    else:
        bsz = _pad4(batch, 1)
        bst = _pad4(A.stride()[:-2], 0)

    n_warps = _num_warps(BLOCK_M, BLOCK_N)
    n_vh_rows = n if full else k

    _svd_jacobi_kernel[(B,)](
        A,
        ur,
        vr,
        sr,
        bsz[3],
        bst[3],
        bsz[2],
        bst[2],
        bsz[1],
        bst[1],
        bsz[0],
        bst[0],
        m,
        n,
        BLOCK_N - 1,
        A.stride(-2),
        A.stride(-1),
        BLOCK_M,
        BLOCK_N,
        14,
        1e-4,
    )

    _svd_sort_kernel[(B,)](
        ur,
        vr,
        sr,
        U,
        S,
        Vh,
        m,
        n,
        k,
        m * (m if full else k),
        k,
        (n * n) if full else (k * n),
        U.stride(-2),
        U.stride(-1),
        Vh.stride(-2),
        Vh.stride(-1),
        n_vh_rows,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=n_warps,
    )

    if full and m > n:
        _svd_compl_kernel[(B,)](
            U,
            m,
            n,
            m * m,
            U.stride(-2),
            U.stride(-1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=n_warps,
        )

    return U, S, Vh
