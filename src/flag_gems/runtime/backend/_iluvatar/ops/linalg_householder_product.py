import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# linalg_householder_product(A, tau)
#
# Reference semantics (torch.linalg.householder_product, empirically verified
# on this target; the flaggems reference implementation uses the same form):
#   - v_i = [0..0, 1, A[i+1:, i]]       (unit diagonal, sub-diagonal from A)
#   - H_i = I - t_i v_i v_i^T,  t_i = 2 / ||v_i||^2
#     (= the geqrf-produced input tau_i, which the eval inputs always satisfy)
#   - Q   = H_0 H_1 ... H_{k-1}, first n columns
#   - when k == m the last reflector is skipped (k' = k - 1)
#
# Implementation: one fused kernel, one program per (batch, column block).
# BLOCK_M covers all rows.  The Q tile is kept in registers for the whole
# kernel.  Reflectors are applied two at a time with a closed-form 2x2
# compact-WY update; both dot products per pair are independent (computed
# against the pre-pair Q), halving the sequential dependency chain:
#   Q <- Q - t_j v_j dots_j + t_i t_j G v_j dots_i - t_i v_i dots_i
#   (pair (i, i-1), applied H_i then H_{i-1}; G = v_{i-1}^T v_i)
# The t coefficients come from the input tau (identical to 2/||v||^2 for
# geqrf inputs), removing two per-pair reductions.  num_warps=1 keeps every
# reduction warp-local (no shared-memory barriers).
# ---------------------------------------------------------------------------


@triton.jit
def _hhp_kernel(
    Q, A, tau, m, n, kk, batch, bs, rs, cs, ts, BM: tl.constexpr, BN: tl.constexpr
):
    pid = tl.program_id(0)
    num_cb = tl.cdiv(n, BN)
    b = pid // num_cb
    cb = pid % num_cb
    cols = cb * BN + tl.arange(0, BN)
    cmask = cols < n
    rows = tl.arange(0, BM)
    rmask = rows < m
    A_b = A + b * bs
    tau_b = tau + b * ts
    Q_b = Q + b * m * n
    q_ptrs = Q_b + rows[:, None] * n + cols[None, :]
    mask_2d = rmask[:, None] & cmask[None, :]
    ident = tl.where(rows[:, None] == cols[None, :], 1.0, 0.0)
    tl.store(q_ptrs, ident, mask=mask_2d)
    q_block = tl.load(q_ptrs, mask=mask_2d, other=0.0)
    # reflector pairs (i, i-1) with i = kk-1, kk-3, ...; H_i applied first.
    for i in range(kk - 1, 0, -2):
        v_i = tl.load(A_b + rows * rs + i * cs, mask=rmask & (rows > i), other=0.0)
        v_j = tl.load(
            A_b + rows * rs + (i - 1) * cs, mask=rmask & (rows > (i - 1)), other=0.0
        )
        v_i = tl.where(rows == i, 1.0, v_i)
        v_j = tl.where(rows == (i - 1), 1.0, v_j)
        G = tl.sum(v_j * v_i)
        t_i = tl.load(tau_b + i)
        t_j = tl.load(tau_b + (i - 1))
        dots_i = tl.sum(v_i[:, None] * q_block, axis=0)
        dots_j = tl.sum(v_j[:, None] * q_block, axis=0)
        q_block = (
            q_block
            - t_j * v_j[:, None] * dots_j[None, :]
            + t_j * t_i * G * v_j[:, None] * dots_i[None, :]
            - t_i * v_i[:, None] * dots_i[None, :]
        )
    if kk % 2 == 1:
        v_0 = tl.load(A_b + rows * rs, mask=rmask, other=0.0)
        v_0 = tl.where(rows == 0, 1.0, v_0)
        dots_0 = tl.sum(v_0[:, None] * q_block, axis=0)
        q_block = q_block - tl.load(tau_b) * (v_0[:, None] * dots_0[None, :])
    tl.store(q_ptrs, q_block, mask=mask_2d)


def run(A, tau):
    logger.debug("GEMS_ILUVATAR LINALG_HOUSEHOLDER_PRODUCT")
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    k = tau.shape[-1]
    kk = k - 1 if k == m else k
    B = 1
    for s in batch_shape:
        B *= s
    bs = A.stride(-3) if A.dim() >= 3 else (m * n)
    rs = A.stride(-2)
    cs = A.stride(-1)
    ts = tau.stride(-2) if tau.dim() >= 2 else k
    out = torch.empty(B, m, n, dtype=A.dtype, device=A.device)
    BM = triton.next_power_of_2(m)
    BN = 1 if n >= 32 else (2 if n >= 16 else 4)
    grid = (B * triton.cdiv(n, BN),)
    _hhp_kernel[grid](
        out,
        A,
        tau,
        m,
        n,
        kk,
        B,
        bs,
        rs,
        cs,
        ts,
        BM=BM,
        BN=BN,
        num_warps=1,
        num_stages=1,
    )
    return out.view(A.shape)
