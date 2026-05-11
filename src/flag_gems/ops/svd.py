import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

SVDResult = namedtuple("SVDResult", ["U", "S", "V"])

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)

_GRAM_CONDITION_GUARD_MAX_BATCH = 16
_GRAM_CONDITION_GUARD_MAX_K = 32
_GRAM_CONDITION_EIGEN_RATIO = 1.0e-8


def _fallback_svd(input, some=True, compute_uv=True):
    return torch.ops.aten.svd.default.redispatch(
        _FALLBACK_KEYSET, input, some, compute_uv
    )


def _aten_bmm(left, right, out_shape):
    out = torch.ops.aten.bmm.default.redispatch(_FALLBACK_KEYSET, left, right)
    return out.reshape(out_shape)


def _svd_shape(input):
    if input.dim() < 2:
        return 0, 0, 0
    m = input.shape[-2]
    n = input.shape[-1]
    batch = 1
    for dim in input.shape[:-2]:
        batch *= dim
    return batch, m, n


def _should_guard_gram_spectrum(batch, k):
    return batch <= _GRAM_CONDITION_GUARD_MAX_BATCH and k <= _GRAM_CONDITION_GUARD_MAX_K


def _gram_spectrum_needs_fallback(vals):
    if vals.numel() == 0:
        return False

    largest = vals[..., -1]
    smallest = vals[..., 0]
    suspicious = (
        (~torch.isfinite(largest))
        | (~torch.isfinite(smallest))
        | (largest <= 0)
        | (smallest <= largest * _GRAM_CONDITION_EIGEN_RATIO)
    )
    return bool(torch.any(suspicious).item())


def _is_float32_cuda_matrix(input):
    return input.is_cuda and input.dtype == torch.float32 and input.dim() >= 2


def _is_iluvatar_backend():
    return device.vendor_name == "iluvatar"


def _can_use_rank1_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return _is_float32_cuda_matrix(input) and some and compute_uv and min(m, n) == 1


def _can_use_rank2_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and min(m, n) == 2
        and max(m, n) <= 1024
    )


def _can_use_2x2_kernel(input):
    _, m, n = _svd_shape(input)
    return _can_use_rank2_kernel(input, True, True) and m == 2 and n == 2


def _can_use_4x4_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return _is_float32_cuda_matrix(input) and some and compute_uv and m == 4 and n == 4


def _can_use_small_jacobi_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and not _is_iluvatar_backend()
        and min(m, n) <= 16
        and max(m, n) <= 1024
    )


@libentry()
@triton.jit
def _small_jacobi_svd_kernel(
    A,
    A_WORK,
    V_WORK,
    U,
    S,
    V,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWEEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    col_mask = cols < K
    eps = 1.0e-20

    a_base = A + pid * M * N
    aw_base = A_WORK + pid * K * ROWS
    vw_base = V_WORK + pid * K * K

    for j in tl.static_range(0, K):
        if TALL:
            vals = tl.load(a_base + rows * N + j, mask=row_mask, other=0.0).to(
                tl.float32
            )
        else:
            vals = tl.load(a_base + j * N + rows, mask=row_mask, other=0.0).to(
                tl.float32
            )
        tl.store(aw_base + j * ROWS + rows, vals, mask=row_mask)
        ident_col = tl.where(cols == j, 1.0, 0.0)
        tl.store(vw_base + j * K + cols, ident_col, mask=col_mask)

    for _ in tl.static_range(0, SWEEPS):
        for p in tl.static_range(0, K):
            for q in tl.static_range(p + 1, K):
                ap = tl.load(aw_base + p * ROWS + rows, mask=row_mask, other=0.0)
                aq = tl.load(aw_base + q * ROWS + rows, mask=row_mask, other=0.0)
                alpha = tl.sum(ap * ap)
                beta = tl.sum(aq * aq)
                gamma = tl.sum(ap * aq)
                abs_gamma = tl.abs(gamma)
                threshold = 1.0e-7 * tl.sqrt(alpha * beta + eps)
                active = abs_gamma > threshold

                safe_gamma = tl.where(active, gamma, 1.0)
                tau = (beta - alpha) / (2.0 * safe_gamma)
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                c = tl.rsqrt(1.0 + t * t)
                s_rot = t * c
                c = tl.where(active, c, 1.0)
                s_rot = tl.where(active, s_rot, 0.0)

                new_ap = c * ap - s_rot * aq
                new_aq = s_rot * ap + c * aq
                tl.store(aw_base + p * ROWS + rows, new_ap, mask=row_mask)
                tl.store(aw_base + q * ROWS + rows, new_aq, mask=row_mask)

                vp = tl.load(vw_base + p * K + cols, mask=col_mask, other=0.0)
                vq = tl.load(vw_base + q * K + cols, mask=col_mask, other=0.0)
                new_vp = c * vp - s_rot * vq
                new_vq = s_rot * vp + c * vq
                tl.store(vw_base + p * K + cols, new_vp, mask=col_mask)
                tl.store(vw_base + q * K + cols, new_vq, mask=col_mask)

    s_idx = tl.arange(0, BLOCK_K)
    s_vals = tl.full((BLOCK_K,), 0.0, dtype=tl.float32)
    for j in tl.static_range(0, K):
        col = tl.load(aw_base + j * ROWS + rows, mask=row_mask, other=0.0)
        norm = tl.sqrt(tl.sum(col * col))
        s_vals = tl.where(s_idx == j, norm, s_vals)

    ranks = tl.zeros((BLOCK_K,), dtype=tl.int32)
    for i in tl.static_range(0, K):
        si = tl.sum(tl.where(s_idx == i, s_vals, 0.0))
        beats = ((si > s_vals) | ((si == s_vals) & (i < s_idx))) & (s_idx < K)
        ranks = ranks + beats.to(tl.int32)

    for j in tl.static_range(0, K):
        col = tl.load(aw_base + j * ROWS + rows, mask=row_mask, other=0.0)
        norm = tl.sum(tl.where(s_idx == j, s_vals, 0.0))
        rank = tl.sum(tl.where(s_idx == j, ranks, 0))
        inv_norm = tl.where(norm > eps, 1.0 / norm, 0.0)
        tl.store(S + pid * K + rank, norm)

        basis = tl.load(vw_base + j * K + cols, mask=col_mask, other=0.0)
        if TALL:
            tl.store(U + pid * M * K + rows * K + rank, col * inv_norm, mask=row_mask)
            tl.store(V + pid * N * K + cols * K + rank, basis, mask=col_mask)
        else:
            tl.store(U + pid * M * K + cols * K + rank, basis, mask=col_mask)
            tl.store(V + pid * N * K + rows * K + rank, col * inv_norm, mask=row_mask)


def _can_use_streaming_jacobi_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and 16 < min(m, n) <= 64
        and max(m, n) <= 1024
    )


def _can_use_gram_eigh_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return _is_float32_cuda_matrix(input) and some and compute_uv and min(m, n) <= 1024


@triton.jit
def _rotate_pair_4(ap, aq, vp, vq):
    eps = 1.0e-20
    alpha = tl.sum(ap * ap, axis=1)
    beta = tl.sum(aq * aq, axis=1)
    gamma = tl.sum(ap * aq, axis=1)
    abs_gamma = tl.abs(gamma)
    threshold = 1.0e-7 * tl.sqrt(alpha * beta + eps)
    active = abs_gamma > threshold
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = tl.rsqrt(1.0 + t * t)
    s_rot = t * c
    c = tl.where(active, c, 1.0)
    s_rot = tl.where(active, s_rot, 0.0)
    new_ap = c[:, None] * ap - s_rot[:, None] * aq
    new_aq = s_rot[:, None] * ap + c[:, None] * aq
    new_vp = c[:, None] * vp - s_rot[:, None] * vq
    new_vq = s_rot[:, None] * vp + c[:, None] * vq
    return new_ap, new_aq, new_vp, new_vq


@libentry()
@triton.jit
def _small4_square_svd_kernel(
    A,
    U,
    S,
    V,
    BATCH: tl.constexpr,
    BLOCK_B: tl.constexpr,
    SWEEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    r = tl.arange(0, 4)
    bb = b[:, None]
    rr = r[None, :]
    mask = b < BATCH
    full_mask = (bb < BATCH) & (rr < 4)
    base = A + bb * 16 + rr * 4

    c0 = tl.load(base, mask=full_mask, other=0.0).to(tl.float32)
    c1 = tl.load(base + 1, mask=full_mask, other=0.0).to(tl.float32)
    c2 = tl.load(base + 2, mask=full_mask, other=0.0).to(tl.float32)
    c3 = tl.load(base + 3, mask=full_mask, other=0.0).to(tl.float32)

    v0 = tl.where(rr == 0, 1.0, 0.0)
    v1 = tl.where(rr == 1, 1.0, 0.0)
    v2 = tl.where(rr == 2, 1.0, 0.0)
    v3 = tl.where(rr == 3, 1.0, 0.0)

    for _ in tl.static_range(0, SWEEPS):
        c0, c1, v0, v1 = _rotate_pair_4(c0, c1, v0, v1)
        c0, c2, v0, v2 = _rotate_pair_4(c0, c2, v0, v2)
        c0, c3, v0, v3 = _rotate_pair_4(c0, c3, v0, v3)
        c1, c2, v1, v2 = _rotate_pair_4(c1, c2, v1, v2)
        c1, c3, v1, v3 = _rotate_pair_4(c1, c3, v1, v3)
        c2, c3, v2, v3 = _rotate_pair_4(c2, c3, v2, v3)

    s0 = tl.sqrt(tl.sum(c0 * c0, axis=1))
    s1 = tl.sqrt(tl.sum(c1 * c1, axis=1))
    s2 = tl.sqrt(tl.sum(c2 * c2, axis=1))
    s3 = tl.sqrt(tl.sum(c3 * c3, axis=1))
    r0 = (s1 > s0).to(tl.int32) + (s2 > s0).to(tl.int32) + (s3 > s0).to(tl.int32)
    r1 = ((s0 >= s1).to(tl.int32)) + (s2 > s1).to(tl.int32) + (s3 > s1).to(tl.int32)
    r2 = ((s0 >= s2).to(tl.int32)) + ((s1 >= s2).to(tl.int32)) + (s3 > s2).to(tl.int32)
    r3 = (
        ((s0 >= s3).to(tl.int32))
        + ((s1 >= s3).to(tl.int32))
        + ((s2 >= s3).to(tl.int32))
    )
    eps = 1.0e-20

    tl.store(S + b * 4 + r0, s0, mask=mask)
    tl.store(S + b * 4 + r1, s1, mask=mask)
    tl.store(S + b * 4 + r2, s2, mask=mask)
    tl.store(S + b * 4 + r3, s3, mask=mask)

    tl.store(
        U + bb * 16 + rr * 4 + r0[:, None],
        c0 / tl.maximum(s0[:, None], eps),
        mask=full_mask,
    )
    tl.store(
        U + bb * 16 + rr * 4 + r1[:, None],
        c1 / tl.maximum(s1[:, None], eps),
        mask=full_mask,
    )
    tl.store(
        U + bb * 16 + rr * 4 + r2[:, None],
        c2 / tl.maximum(s2[:, None], eps),
        mask=full_mask,
    )
    tl.store(
        U + bb * 16 + rr * 4 + r3[:, None],
        c3 / tl.maximum(s3[:, None], eps),
        mask=full_mask,
    )

    tl.store(V + bb * 16 + rr * 4 + r0[:, None], v0, mask=full_mask)
    tl.store(V + bb * 16 + rr * 4 + r1[:, None], v1, mask=full_mask)
    tl.store(V + bb * 16 + rr * 4 + r2[:, None], v2, mask=full_mask)
    tl.store(V + bb * 16 + rr * 4 + r3[:, None], v3, mask=full_mask)


@libentry()
@triton.jit
def _rank2_svd_tiny_kernel(
    A,
    U,
    S,
    V,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    r = tl.arange(0, BLOCK_R)
    bb = b[:, None]
    rr = r[None, :]
    bmask = b < BATCH
    eps = 1.0e-20

    if TALL:
        mask = (bb < BATCH) & (rr < M)
        base = A + bb * M * N + rr * N
        x = tl.load(base, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(base + 1, mask=mask, other=0.0).to(tl.float32)
    else:
        mask = (bb < BATCH) & (rr < N)
        base = A + bb * M * N + rr
        x = tl.load(base, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(base + N, mask=mask, other=0.0).to(tl.float32)

    aa = tl.sum(x * x, axis=1)
    bbv = tl.sum(y * y, axis=1)
    ab = tl.sum(x * y, axis=1)
    diff = aa - bbv
    root = tl.sqrt(diff * diff + 4.0 * ab * ab)
    l0 = tl.maximum(0.0, 0.5 * (aa + bbv + root))
    l1 = tl.maximum(0.0, 0.5 * (aa + bbv - root))
    s0 = tl.sqrt(l0)
    s1 = tl.sqrt(l1)

    ab_abs = tl.abs(ab)
    aa_ge_bb = aa >= bbv
    vx0 = tl.where(ab_abs > eps, ab, tl.where(aa_ge_bb, 1.0, 0.0))
    vy0 = tl.where(ab_abs > eps, l0 - aa, tl.where(aa_ge_bb, 0.0, 1.0))
    inv_norm = tl.rsqrt(vx0 * vx0 + vy0 * vy0 + eps)
    vx0 = vx0 * inv_norm
    vy0 = vy0 * inv_norm
    vx1 = -vy0
    vy1 = vx0

    tl.store(S + b * 2, s0, mask=bmask)
    tl.store(S + b * 2 + 1, s1, mask=bmask)
    inv_s0 = tl.where(s0 > eps, 1.0 / s0, 0.0)
    inv_s1 = tl.where(s1 > eps, 1.0 / s1, 0.0)

    if TALL:
        u0 = (x * vx0[:, None] + y * vy0[:, None]) * inv_s0[:, None]
        u1 = (x * vx1[:, None] + y * vy1[:, None]) * inv_s1[:, None]
        ubase = U + bb * M * 2 + rr * 2
        tl.store(ubase, u0, mask=mask)
        tl.store(ubase + 1, u1, mask=mask)
        vbase = V + b * 4
        tl.store(vbase, vx0, mask=bmask)
        tl.store(vbase + 1, vx1, mask=bmask)
        tl.store(vbase + 2, vy0, mask=bmask)
        tl.store(vbase + 3, vy1, mask=bmask)
    else:
        ubase = U + b * 4
        tl.store(ubase, vx0, mask=bmask)
        tl.store(ubase + 1, vx1, mask=bmask)
        tl.store(ubase + 2, vy0, mask=bmask)
        tl.store(ubase + 3, vy1, mask=bmask)
        v0 = (x * vx0[:, None] + y * vy0[:, None]) * inv_s0[:, None]
        v1 = (x * vx1[:, None] + y * vy1[:, None]) * inv_s1[:, None]
        vbase = V + bb * N * 2 + rr * 2
        tl.store(vbase, v0, mask=mask)
        tl.store(vbase + 1, v1, mask=mask)


@libentry()
@triton.jit
def _rank2_svd_kernel(
    A,
    U,
    S,
    V,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_R)
    eps = 1.0e-20

    if TALL:
        mask = offs < M
        base = A + pid * M * N
        x = tl.load(base + offs * N, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(base + offs * N + 1, mask=mask, other=0.0).to(tl.float32)
    else:
        mask = offs < N
        base = A + pid * M * N
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(base + N + offs, mask=mask, other=0.0).to(tl.float32)

    aa = tl.sum(x * x)
    bb = tl.sum(y * y)
    ab = tl.sum(x * y)
    diff = aa - bb
    root = tl.sqrt(diff * diff + 4.0 * ab * ab)
    l0 = tl.maximum(0.0, 0.5 * (aa + bb + root))
    l1 = tl.maximum(0.0, 0.5 * (aa + bb - root))
    s0 = tl.sqrt(l0)
    s1 = tl.sqrt(l1)

    ab_abs = tl.abs(ab)
    aa_ge_bb = aa >= bb
    vx0 = tl.where(ab_abs > eps, ab, tl.where(aa_ge_bb, 1.0, 0.0))
    vy0 = tl.where(ab_abs > eps, l0 - aa, tl.where(aa_ge_bb, 0.0, 1.0))
    inv_norm = tl.rsqrt(vx0 * vx0 + vy0 * vy0 + eps)
    vx0 = vx0 * inv_norm
    vy0 = vy0 * inv_norm
    vx1 = -vy0
    vy1 = vx0

    sbase = S + pid * 2
    tl.store(sbase, s0)
    tl.store(sbase + 1, s1)

    inv_s0 = tl.where(s0 > eps, 1.0 / s0, 0.0)
    inv_s1 = tl.where(s1 > eps, 1.0 / s1, 0.0)

    if TALL:
        ubase = U + pid * M * 2
        u0 = (x * vx0 + y * vy0) * inv_s0
        u1 = (x * vx1 + y * vy1) * inv_s1
        tl.store(ubase + offs * 2, u0, mask=mask)
        tl.store(ubase + offs * 2 + 1, u1, mask=mask)

        vbase = V + pid * 4
        tl.store(vbase, vx0)
        tl.store(vbase + 1, vx1)
        tl.store(vbase + 2, vy0)
        tl.store(vbase + 3, vy1)
    else:
        ubase = U + pid * 4
        tl.store(ubase, vx0)
        tl.store(ubase + 1, vx1)
        tl.store(ubase + 2, vy0)
        tl.store(ubase + 3, vy1)

        vbase = V + pid * N * 2
        v0 = (x * vx0 + y * vy0) * inv_s0
        v1 = (x * vx1 + y * vy1) * inv_s1
        tl.store(vbase + offs * 2, v0, mask=mask)
        tl.store(vbase + offs * 2 + 1, v1, mask=mask)


def _rank2_svd(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    u = torch.empty((batch, m, 2), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, 2), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, 2), dtype=input.dtype, device=input.device)
    block_r = triton.next_power_of_2(max(m, n))
    with torch_device_fn.device(input.device):
        if max(m, n) <= 16 and batch >= 16:
            block_b = 16
            _rank2_svd_tiny_kernel[(triton.cdiv(batch, block_b),)](
                a,
                u,
                s,
                v,
                BATCH=batch,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_B=block_b,
                BLOCK_R=block_r,
                num_warps=1,
            )
        else:
            _rank2_svd_kernel[(batch,)](
                a,
                u,
                s,
                v,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_R=block_r,
                num_warps=1 if block_r <= 64 else 4,
            )
    return (
        u.reshape(*input.shape[:-2], m, 2),
        s.reshape(*input.shape[:-2], 2),
        v.reshape(*input.shape[:-2], n, 2),
    )


def _small_jacobi_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    v_work = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    u = torch.empty((batch, m, k), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, k), dtype=input.dtype, device=input.device)
    block_r = triton.next_power_of_2(rows)
    block_k = triton.next_power_of_2(k)
    sweeps = 3 if k <= 4 else 6
    with torch_device_fn.device(input.device):
        _small_jacobi_svd_kernel[(batch,)](
            a,
            a_work,
            v_work,
            u,
            s,
            v,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=m >= n,
            BLOCK_R=block_r,
            BLOCK_K=block_k,
            SWEEPS=sweeps,
            num_warps=1 if block_r <= 64 else 4,
        )
    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


def _small4_square_svd(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    u = torch.empty((batch, 4, 4), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, 4), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, 4, 4), dtype=input.dtype, device=input.device)
    block_b = 16
    with torch_device_fn.device(input.device):
        _small4_square_svd_kernel[(triton.cdiv(batch, block_b),)](
            a, u, s, v, BATCH=batch, BLOCK_B=block_b, SWEEPS=4, num_warps=1
        )
    return (
        u.reshape(*input.shape[:-2], 4, 4),
        s.reshape(*input.shape[:-2], 4),
        v.reshape(*input.shape[:-2], 4, 4),
    )


def _rank1_svd(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    if n == 1:
        col = a[..., :, 0]
        s = torch.linalg.vector_norm(col, dim=-1, keepdim=True)
        u = col.unsqueeze(-1) / s.clamp_min(torch.finfo(input.dtype).eps).unsqueeze(-1)
        v = torch.ones((batch, 1, 1), dtype=input.dtype, device=input.device)
    else:
        row = a[..., 0, :]
        s = torch.linalg.vector_norm(row, dim=-1, keepdim=True)
        u = torch.ones((batch, 1, 1), dtype=input.dtype, device=input.device)
        v = row.unsqueeze(-1) / s.clamp_min(torch.finfo(input.dtype).eps).unsqueeze(-1)
    return (
        u.reshape(*input.shape[:-2], m, 1),
        s.reshape(*input.shape[:-2], 1),
        v.reshape(*input.shape[:-2], n, 1),
    )


def _gram_svd(input):
    a = input.contiguous()
    m = a.shape[-2]
    n = a.shape[-1]
    batch = 1
    for dim in a.shape[:-2]:
        batch *= dim
    finfo = torch.finfo(a.dtype)
    if m >= n:
        a_3d = a.reshape(batch, m, n)
        at_3d = a.transpose(-2, -1).reshape(batch, n, m)
        gram = _aten_bmm(at_3d, a_3d, (*a.shape[:-2], n, n))
        vals, v = torch.linalg.eigh(gram)
        if _should_guard_gram_spectrum(batch, n) and _gram_spectrum_needs_fallback(
            vals
        ):
            return _fallback_svd(input, True, True)
        vals = vals.flip(-1).clamp_min_(0.0)
        v = v.flip(-1)
        s = torch.sqrt(vals)
        u = _aten_bmm(
            a_3d,
            v.reshape(batch, n, n),
            (*a.shape[:-2], m, n),
        ) / s.clamp_min(finfo.eps).unsqueeze(-2)
        return u, s, v

    a_3d = a.reshape(batch, m, n)
    at_3d = a.transpose(-2, -1).reshape(batch, n, m)
    gram = _aten_bmm(a_3d, at_3d, (*a.shape[:-2], m, m))
    vals, u = torch.linalg.eigh(gram)
    if _should_guard_gram_spectrum(batch, m) and _gram_spectrum_needs_fallback(vals):
        return _fallback_svd(input, True, True)
    vals = vals.flip(-1).clamp_min_(0.0)
    u = u.flip(-1)
    s = torch.sqrt(vals)
    v = _aten_bmm(
        at_3d,
        u.reshape(batch, m, m),
        (*a.shape[:-2], n, m),
    ) / s.clamp_min(
        finfo.eps
    ).unsqueeze(-2)
    return u, s, v


@libentry()
@triton.jit
def _gram16_finalize_kernel(
    A,
    EVALS,
    EVECS,
    U,
    S,
    V,
    M: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    EVECS_BATCH_STRIDE: tl.constexpr,
    EVECS_ROW_STRIDE: tl.constexpr,
    EVECS_COL_STRIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, 16)
    src_cols = 15 - cols
    row_mask = rows < ROWS
    eps = 1.0e-20

    vals = tl.load(EVALS + batch * 16 + src_cols)
    s_vals = tl.sqrt(tl.maximum(vals, 0.0))
    inv_s = tl.where(s_vals > eps, 1.0 / s_vals, 0.0)

    acc = tl.zeros((BLOCK_R, 16), dtype=tl.float32)
    a_base = A + batch * M * N
    e_base = EVECS + batch * EVECS_BATCH_STRIDE
    for k in tl.static_range(0, 16):
        eig = tl.load(e_base + k * EVECS_ROW_STRIDE + src_cols * EVECS_COL_STRIDE)
        if TALL:
            a_vals = tl.load(
                a_base + rows * N + k,
                mask=row_mask,
                other=0.0,
            )
        else:
            a_vals = tl.load(
                a_base + k * N + rows,
                mask=row_mask,
                other=0.0,
            )
        acc += a_vals[:, None] * eig[None, :]

    projected = acc * inv_s[None, :]
    if TALL:
        tl.store(
            U + batch * M * 16 + rows[:, None] * 16 + cols[None, :],
            projected,
            mask=row_mask[:, None],
        )
    else:
        tl.store(
            V + batch * N * 16 + rows[:, None] * 16 + cols[None, :],
            projected,
            mask=row_mask[:, None],
        )

    head_mask = row_block == 0
    tl.store(S + batch * 16 + cols, s_vals, mask=head_mask)

    basis_rows = tl.arange(0, 16)
    basis_cols = tl.arange(0, 16)
    basis_src_cols = 15 - basis_cols
    basis = tl.load(
        e_base
        + basis_rows[:, None] * EVECS_ROW_STRIDE
        + basis_src_cols[None, :] * EVECS_COL_STRIDE
    )
    if TALL:
        tl.store(
            V + batch * N * 16 + basis_rows[:, None] * 16 + basis_cols[None, :],
            basis,
            mask=head_mask,
        )
    else:
        tl.store(
            U + batch * M * 16 + basis_rows[:, None] * 16 + basis_cols[None, :],
            basis,
            mask=head_mask,
        )


def _gram16_svd(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    u = torch.empty((batch, m, 16), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, 16), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, 16), dtype=input.dtype, device=input.device)
    rows = max(m, n)
    if m >= n:
        at_3d = a.transpose(-2, -1).reshape(batch, n, m)
        gram = _aten_bmm(at_3d, a, (batch, n, n))
    else:
        at_3d = a.transpose(-2, -1).reshape(batch, n, m)
        gram = _aten_bmm(a, at_3d, (batch, m, m))
    vals, basis = torch.linalg.eigh(gram)
    if _should_guard_gram_spectrum(batch, 16) and _gram_spectrum_needs_fallback(vals):
        return _fallback_svd(input, True, True)

    block_r = 64 if m >= n else 128
    with torch_device_fn.device(input.device):
        _gram16_finalize_kernel[(batch, triton.cdiv(rows, block_r))](
            a,
            vals,
            basis,
            u,
            s,
            v,
            M=m,
            N=n,
            ROWS=rows,
            TALL=m >= n,
            EVECS_BATCH_STRIDE=basis.stride(0),
            EVECS_ROW_STRIDE=basis.stride(-2),
            EVECS_COL_STRIDE=basis.stride(-1),
            BLOCK_R=block_r,
            num_warps=4,
        )
    return (
        u.reshape(*input.shape[:-2], m, 16),
        s.reshape(*input.shape[:-2], 16),
        v.reshape(*input.shape[:-2], n, 16),
    )


def _gesvda_svd(input):
    u, s, vh = torch.linalg.svd(
        input.contiguous(), full_matrices=False, driver="gesvda"
    )
    return u, s, vh.mH


def _should_use_gram16(batch, m, n):
    return batch >= 16 and min(m, n) == 16 and max(m, n) <= 1024


def _should_use_gram(batch, m, n):
    k = min(m, n)
    largest = max(m, n)
    if k <= 32:
        return True
    if batch <= 4 and m == n and m <= 256:
        return True
    if (m, n) == (1024, 1024):
        return True
    if batch >= 128 and k <= 64 and largest <= 1024:
        return False
    return False


def svd(input, some=True, compute_uv=True):
    logger.debug("GEMS SVD")
    if not _is_float32_cuda_matrix(input) or not some or not compute_uv:
        return SVDResult(*_fallback_svd(input, some, compute_uv))
    if 0 in input.shape[-2:]:
        return SVDResult(*_fallback_svd(input, some, compute_uv))

    batch, m, n = _svd_shape(input)
    k = min(m, n)
    try:
        if k == 1:
            return SVDResult(*_rank1_svd(input))
        if k == 2 and max(m, n) <= 1024:
            return SVDResult(*_rank2_svd(input))
        if k == 4 and m == 4 and n == 4 and batch >= 16:
            return SVDResult(*_small4_square_svd(input))
        if k <= 8 and max(m, n) <= 1024:
            if _is_iluvatar_backend():
                return SVDResult(*_fallback_svd(input, some, compute_uv))
            return SVDResult(*_small_jacobi_svd(input))
        if _should_use_gram16(batch, m, n):
            return SVDResult(*_gram16_svd(input))
        if _should_use_gram(batch, m, n):
            return SVDResult(*_gram_svd(input))
        return SVDResult(*_gesvda_svd(input))
    except RuntimeError:
        return SVDResult(*_fallback_svd(input, some, compute_uv))
