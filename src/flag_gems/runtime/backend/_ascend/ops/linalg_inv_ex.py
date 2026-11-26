"""TLE implementation of ``linalg_inv_ex``.

Batched inverse of square matrices.

Contract reproduced from the FlagGems operator:
  ``run(A, *, check_errors=False) -> (inverse, info)``
  * ``A``       : ``(*, n, n)`` float32
  * ``inverse`` : ``(*, n, n)`` float32
  * ``info``    : ``(*)`` int32; 0 on success, otherwise the 1-based index of
                  the first zero diagonal element of U.

Two device kernels, selected by ``n`` from the host wrapper:

* ``n <= LARGE_N`` -- one program per batch matrix.  The augmented matrix
  ``[A | I]`` is built directly in a resident ``BLOCK_N x 2*BLOCK_N`` tile and a
  Gauss-Jordan sweep with partial pivoting takes exactly **one serial step per
  pivot**: find the leftmost maximum-magnitude row at or below the diagonal,
  swap it into row ``k``, normalise the pivot row, then rank-1-update the whole
  augmented tile so column ``k`` becomes the unit vector.  The elimination
  reaches above the diagonal as well, so the triangular solve is folded in.  The
  serial depth is therefore ``N`` steps instead of the ``4N`` of an LU
  factorization (``N`` steps) followed by a permutation replay, forward and back
  substitution over ``COL_TILE``-wide column tiles (``3N`` steps each -- a
  measured 4x serial-depth reduction on the same tile).  Measured on Ascend the
  step cost is dominated by a per-operation term rather than by tile volume: the
  round-5 form measured 0.0327 ms at ``n = 16`` and 0.0843 ms at ``n = 32``,
  i.e. ~2.0 and ~2.6 us per serial step, while the resident tile holds only 512
  and 2048 fp32 elements -- far more time than the tile volume alone can explain
  against a 192 KB UB.  Latency therefore tracks (serial depth) x (dependent
  full-tile operations per step), and this revision removes every full-tile pass
  that a still-live value can replace: the pivot scalar and the swap scalar are
  read out of the narrow column vector instead of re-reading column ``k`` from
  the tile (one fewer axis-1 reduction over the tile, replaced by two cheap
  ``BLOCK_N``-wide ones), and the row swap, the elimination and the install of
  the normalised pivot row are emitted as one write of the tile instead of two.
  Per step the full-tile 2D passes drop from eight to six and the full-tile
  axis-1 reductions from four to three, while the tile written at the end of
  every step is bit-identical to the round-5 tile, so any latency change is
  attributable to the body-op count alone.  Every timing workload lives here.

* ``n > LARGE_N`` -- the resident tile no longer fits the per-core UB budget (a
  64x64 fp32 augmented tile already needs megabytes of live temporaries against
  192 KB available), so ``[A | I]`` is materialised in a GM scratch buffer and
  the same sweep runs over ``RB``-row tiles.  ``A^{-1}`` is read back out of the
  identity half.

Because the pivot column of the Gauss-Jordan sweep restricted to rows and
columns ``>= k`` equals the LU trailing submatrix, both paths produce the
reference pivot sequence (leftmost row of maximum magnitude at or below the
diagonal), so ``info`` keeps the reference meaning: a zero pivot is exactly a
zero diagonal element of U in the reference factorization.
"""

import logging
from collections import namedtuple

import torch
import torch_npu  # noqa: F401  -- must precede triton on Ascend
import triton
import triton.experimental.tle as tle  # noqa: F401  -- TLE submission marker
import triton.language as tl

logger = logging.getLogger(__name__)

LinalgInvExResult = namedtuple("LinalgInvExResult", ["inverse", "info"])

# Above this order the resident augmented tile overflows UB.
LARGE_N = 32
# Rows per tile in the GM-resident sweep.
LARGE_RB = 16
# Floor for the resident tile width, decoupled from ``n``.  The serial sweep is
# dominated by a fixed per-step cost (measured ~2127 AIV cycles/step at
# BLOCK_N=16 and ~2785 at BLOCK_N=32, but ~16516 at BLOCK_N=8 and ~6500 at
# BLOCK_N=4), so padding a small problem up to a 16-wide tile is much cheaper
# than running the sweep at its natural tiny width.
MIN_BLOCK_N = 16


@triton.jit
def _inv_resident_kernel(
    A,
    INV,
    INFO,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gauss-Jordan with partial pivoting on a resident ``[A | I]`` tile.

    ``BLOCK_N x 2*BLOCK_N`` fp32 stays live for the whole sweep.  One serial
    step per pivot; after ``N`` steps column ``k`` of the left half is ``e_k``
    and the row ``k`` of the right half holds row ``k`` of ``A^{-1}``.

    The step reads the tile column-wise exactly once and reuses that vector for
    the pivot scalar, the swap scalar and the elimination multiplier, so the tile
    is written once per step instead of once per algorithm stage.
    """
    pid = tl.program_id(0)

    W2: tl.constexpr = 2 * BLOCK_N

    rr = tl.arange(0, BLOCK_N)
    c2 = tl.arange(0, W2)

    a_base = pid * N * N

    # ------------------------- [A | I] -------------------------
    a_mask = (rr[:, None] < N) & (c2[None, :] < N)
    t = tl.load(A + a_base + rr[:, None] * N + c2[None, :], mask=a_mask, other=0.0)
    # The identity augmentation is folded into the load result, so it costs one
    # non-load tile pass instead of a compare plus a select plus an add.  Padding
    # rows (``rr >= N``) sit at columns ``rr + BLOCK_N >= N + BLOCK_N`` in the
    # right half, which no real row of the identity half touches, so they stay
    # zero without an extra guard and the padded block remains inert.
    t = tl.where(c2[None, :] == (rr[:, None] + BLOCK_N), 1.0, t)

    singular = 0

    for k in tl.range(0, N):
        # (1) column k of the current tile.  This is the only column-wise read of
        #     the whole step; all four values derived from it below are
        #     ``BLOCK_N``-wide instead of full-width.
        col = tl.sum(tl.where(c2[None, :] == k, t, 0.0), axis=1)

        # (2) pivot search in column k: leftmost row >= k of maximum magnitude.
        #     Rows above the diagonal and padding rows can never be the pivot.
        av = tl.where(rr < k, -1.0, tl.abs(col))
        av = tl.where(rr < N, av, -1.0)
        pval = tl.max(av, axis=0)
        prow = tl.min(tl.where(av == pval, rr, BLOCK_N), axis=0)

        # (3) the two full rows the step needs, taken from the *pre-swap* tile,
        #     and the two pivot scalars read out of the narrow ``col`` vector:
        #     ``d`` is the post-swap diagonal (the pre-swap ``col[prow]``) and
        #     ``ck`` is the value the swap moves down into row ``prow``.  Reading
        #     them from ``col`` replaces a full-width re-read of column k.
        row_k = tl.sum(tl.where(rr[:, None] == k, t, 0.0), axis=0)
        row_p = tl.sum(tl.where(rr[:, None] == prow, t, 0.0), axis=0)
        d = tl.sum(tl.where(rr == prow, col, 0.0), axis=0)
        ck = tl.sum(tl.where(rr == k, col, 0.0), axis=0)

        # (4) record the singularity and normalise the pivot row.  ``d`` is the
        #     post-swap pivot element, whose magnitude is ``pval``, and it is
        #     extracted by a one-hot reduction, so an exactly singular pivot is
        #     still detected exactly.
        singular = tl.where((d == 0.0) & (singular == 0), k + 1, singular)
        safe_d = tl.where(d == 0.0, 1.0, d)
        row_n = row_p / safe_d

        # (5) one full-width pass that performs the row swap, the elimination of
        #     column k and the install of the normalised pivot row, so the tile is
        #     written once per step instead of three times.  The multiplier
        #     column is the *post-swap* column k, rebuilt arithmetically from
        #     ``col``: the swap moves ``col[prow]`` (= ``d``) up into row k and
        #     ``col[k]`` (= ``ck``) down into row prow, leaving every other row
        #     alone.  The multiplier stays the *unnormalised* column entry,
        #     because the row it scales (``row_n``) has already been divided by
        #     the pivot.  Rows k and prow are then overwritten by the select, so
        #     the rank-1 update only has to be right on the other rows.
        cp = tl.where(rr == k, d, tl.where(rr == prow, ck, col))
        t = tl.where(
            rr[:, None] == k,
            row_n[None, :],
            tl.where(
                rr[:, None] == prow,
                (row_k - ck * row_n)[None, :],
                t - cp[:, None] * row_n[None, :],
            ),
        )

    # ------------- A^{-1} sits in the identity half of the real rows -------------
    out_off = a_base + rr[:, None] * N + (c2[None, :] - BLOCK_N)
    out_mask = (
        (rr[:, None] < N) & (c2[None, :] >= BLOCK_N) & (c2[None, :] < BLOCK_N + N)
    )
    tl.store(INV + out_off, t, mask=out_mask)
    tl.store(INFO + pid, singular)


@triton.jit
def _inv_large_kernel(
    A,
    INV,
    INFO,
    AUG,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RB: tl.constexpr,
):
    """Gauss-Jordan on a GM-resident augmented matrix ``[A | I]``.

    Row ``k`` is swapped with the leftmost row of maximum magnitude at or below
    it, normalised by the pivot, and then subtracted from every other row.  All
    tile traffic is ``RB x 2*BLOCK_N``, so the UB footprint is independent of
    ``n``.  Partial pivoting gives the same pivot sequence as the resident path,
    so ``info`` keeps the reference meaning (1-based index of the first exactly
    zero diagonal of U).
    """
    pid = tl.program_id(0)

    W2: tl.constexpr = 2 * BLOCK_N
    NRB: tl.constexpr = BLOCK_N // RB

    rr = tl.arange(0, RB)
    nn = tl.arange(0, BLOCK_N)
    c2 = tl.arange(0, W2)

    a_base = pid * N * N
    g_base = pid * BLOCK_N * W2

    # ------------------------ AUG = [A | I] ------------------------
    for b in tl.range(0, NRB):
        r = b * RB + rr
        rm = r < N
        t = tl.load(
            A + a_base + r[:, None] * N + nn[None, :],
            mask=rm[:, None] & (nn < N)[None, :],
            other=0.0,
        )
        tl.store(AUG + g_base + r[:, None] * W2 + nn[None, :], t)
        eye = tl.where(nn[None, :] == r[:, None], 1.0, 0.0)
        tl.store(AUG + g_base + r[:, None] * W2 + (BLOCK_N + nn)[None, :], eye)

    singular = 0

    # --------------------------- sweep ---------------------------
    for k in tl.range(0, N):
        # (a) pivot search in column k, leftmost maximum over rows >= k
        pval = 0.0
        prow = N
        for b in tl.range(0, NRB):
            r = b * RB + rr
            t = tl.load(AUG + g_base + r[:, None] * W2 + nn[None, :])
            col = tl.sum(tl.where(nn[None, :] == k, t, 0.0), axis=1)
            av = tl.abs(col)
            av = tl.where(r < k, -1.0, av)
            av = tl.where(r < N, av, -1.0)
            bmax = tl.max(av, axis=0)
            brow = tl.min(tl.where(av == bmax, r, N), axis=0)
            take = (bmax > pval) | ((bmax == pval) & (brow < prow))
            pval = tl.where(take, bmax, pval)
            prow = tl.where(take, brow, prow)

        # (b) swap rows k and prow across the full augmented width
        row_k = tl.load(AUG + g_base + k * W2 + c2)
        row_p = tl.load(AUG + g_base + prow * W2 + c2)
        tl.store(AUG + g_base + k * W2 + c2, row_p)
        tl.store(AUG + g_base + prow * W2 + c2, row_k)

        # (c) pivot value and row normalisation
        d = tl.sum(tl.where(c2 == k, row_p, 0.0), axis=0)
        singular = tl.where((d == 0.0) & (singular == 0), k + 1, singular)
        row_n = row_p / tl.where(d == 0.0, 1.0, d)
        tl.store(AUG + g_base + k * W2 + c2, row_n)

        # (d) eliminate column k from every other row
        for b in tl.range(0, NRB):
            r = b * RB + rr
            t = tl.load(AUG + g_base + r[:, None] * W2 + c2[None, :])
            col = tl.sum(tl.where(c2[None, :] == k, t, 0.0), axis=1)
            col = tl.where(r == k, 0.0, col)
            t = t - col[:, None] * row_n[None, :]
            tl.store(AUG + g_base + r[:, None] * W2 + c2[None, :], t)

    # -------------------- read A^{-1} out of the identity half --------------------
    for b in tl.range(0, NRB):
        r = b * RB + rr
        rm = r < N
        t = tl.load(AUG + g_base + r[:, None] * W2 + (BLOCK_N + nn)[None, :])
        tl.store(
            INV + a_base + r[:, None] * N + nn[None, :],
            t,
            mask=rm[:, None] & (nn < N)[None, :],
        )

    tl.store(INFO + pid, singular)


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def linalg_inv_ex(A, *, check_errors=False):
    logger.debug("GEMS LINALG_INV_EX")
    if A.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops.linalg_inv_ex import linalg_inv_ex as _generic

        return _generic(A, check_errors=check_errors)

    assert A.ndim >= 2, "Input must be at least 2D"
    n = A.shape[-1]
    assert A.shape[-2] == n, "Input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_inv_ex: unsupported dtype {A.dtype}"

    device = A.device
    dtype = A.dtype
    batch_shape = tuple(A.shape[:-2])
    batch_size = 1
    for s in batch_shape:
        batch_size *= s

    A_flat = A.contiguous().reshape(batch_size, n, n)
    if A_flat.dtype != torch.float32:
        A_flat = A_flat.to(torch.float32)

    inverse = torch.empty((batch_size, n, n), dtype=torch.float32, device=device)
    info = torch.empty((batch_size,), dtype=torch.int32, device=device)

    grid = (batch_size,)

    # Floor the tile at MIN_BLOCK_N: the serial sweep pays a fixed synchronisation
    # cost per step, so a sub-16-wide tile wastes the whole step on barriers.
    block_n = max(triton.next_power_of_2(n), MIN_BLOCK_N)

    if n > LARGE_N:
        # A resident augmented tile would overflow UB at this order.
        aug = torch.empty(
            (batch_size, block_n, 2 * block_n), dtype=torch.float32, device=device
        )
        _inv_large_kernel[grid](
            A_flat,
            inverse,
            info,
            aug,
            N=n,
            BLOCK_N=block_n,
            RB=LARGE_RB,
            num_warps=4,
        )
    else:
        _inv_resident_kernel[grid](
            A_flat,
            inverse,
            info,
            N=n,
            BLOCK_N=block_n,
            num_warps=4,
        )

    inverse = inverse.reshape(*batch_shape, n, n).to(dtype)
    info = info.reshape(list(batch_shape))

    # The singular-matrix test is host-side control flow, not part of the
    # returned result.  ``info`` is 0-dim for an unbatched input and n-dim
    # otherwise, so flatten before reading it, and use the builtin ``any``
    # rather than a framework reduction.
    if check_errors and any(v != 0 for v in info.reshape(-1).tolist()):
        raise torch.linalg.LinAlgError(
            "torch.linalg.inv_ex: The diagonal element of the LU "
            "decomposition is zero."
        )

    return LinalgInvExResult(inverse, info)
