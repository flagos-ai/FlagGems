import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

SVDResult = namedtuple("SVDResult", ["U", "S", "V"])

_GRAM_CONDITION_GUARD_MAX_BATCH = 16
_GRAM_CONDITION_GUARD_MAX_K = 32
_GRAM_CONDITION_EIGEN_RATIO = 1.0e-8
_GRAM_TALL_WIDE_MAX_K = 32
_GRAM_TALL_WIDE_MAX_ROWS = 1024
_RANK1_BLOCK_R_MAX = 1024
_RANK2_BLOCK_R_MAX = 2048
_TSQR_CHOLESKY_MAX_BATCH = 32
_TSQR_CHOLESKY_MAX_K = 128
_TSQR_CHOLESKY_MAX_ROWS = 1024


def _unsupported_svd(input, some=True, compute_uv=True, reason=None):
    batch, m, n = _svd_shape(input)
    suffix = "" if reason is None else f" {reason}"
    raise NotImplementedError(
        "FlagGems native SVD currently supports float32 CUDA matrices with "
        "some=True, compute_uv=True, non-empty inputs, and native Triton "
        f"rank/Jacobi shape coverage; got batch={batch}, m={m}, n={n}, "
        f"dtype={input.dtype}, device={input.device}, some={some}, "
        f"compute_uv={compute_uv}.{suffix}"
    )


def _is_iluvatar_backend():
    return device.vendor_name == "iluvatar"


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


def _is_float32_cuda_matrix(input):
    return input.is_cuda and input.dtype == torch.float32 and input.dim() >= 2


def _is_low_precision_cuda_matrix(input):
    return (
        input.is_cuda
        and input.dtype in (torch.float16, torch.bfloat16)
        and input.dim() >= 2
    )


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
        and max(m, n) <= _RANK2_BLOCK_R_MAX
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


def _can_use_cyclic_jacobi_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    k = min(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and 16 <= k <= 64
        and max(m, n) <= 1024
    )


def _can_use_gram_jacobi_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    k = min(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and 16 <= k <= 32
        and max(m, n) <= 64
    )


def _can_use_tall_wide_gram_jacobi_kernel(input, some=True, compute_uv=True):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and batch >= 128
        and 16 <= k <= _GRAM_TALL_WIDE_MAX_K
        and rows <= _GRAM_TALL_WIDE_MAX_ROWS
        and rows >= 2 * k
    )


def _can_use_tsqr_cholesky_kernel(input, some=True, compute_uv=True):
    # Input-dependent TSQR safety needs a native device-side guard before dispatch.
    return False


def _can_use_blocked_jacobi_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    k = min(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and 64 < k <= 512
        and max(m, n) <= 1024
    )


def _can_use_blocked_square_project_kernel(input, some=True, compute_uv=True):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and batch == 1
        and m == n
        and 128 <= k <= 512
    )


def _can_use_hier_block_square_project_kernel(input, some=True, compute_uv=True):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and batch <= 2
        and m == n
        and k in (256, 512)
    )


def _can_use_projected_jacobi_kernel(input, some=True, compute_uv=True):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and 4 <= batch <= 32
        and k == 64
        and max(m, n) <= 128
    )


def _can_use_singular_values_only(input, some=True, compute_uv=False):
    _, m, n = _svd_shape(input)
    k = min(m, n)
    return (
        _is_float32_cuda_matrix(input)
        and not compute_uv
        and k <= 512
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


@libentry()
@triton.jit
def _small_jacobi_svals_kernel(
    A,
    A_WORK,
    S,
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
    s_idx = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    eps = 1.0e-20

    a_base = A + pid * M * N
    aw_base = A_WORK + pid * K * ROWS

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
        norm = tl.sum(tl.where(s_idx == j, s_vals, 0.0))
        rank = tl.sum(tl.where(s_idx == j, ranks, 0))
        tl.store(S + pid * K + rank, norm)


def _can_use_streaming_jacobi_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return (
        _is_float32_cuda_matrix(input)
        and some
        and compute_uv
        and 16 < min(m, n) <= 64
        and max(m, n) <= 1024
    )


def _can_use_gram_kernel(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    return _is_float32_cuda_matrix(input) and some and compute_uv and min(m, n) <= 1024


@libentry()
@triton.jit
def _triton_bmm_kernel(
    A,
    B,
    C,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A + batch * stride_ab
    b_base = B + batch * stride_bb
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        a = tl.load(
            a_base + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_base + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        C + batch * M * N + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _triton_bmm(left, right, out_shape):
    batch, m, k = left.shape
    right_batch, right_k, n = right.shape
    assert batch == right_batch, "Batch dim mismatch"
    assert k == right_k, "K dim mismatch"
    out = torch.empty((batch, m, n), dtype=left.dtype, device=left.device)
    block_m = 16 if m <= 16 else 32
    block_n = 16 if n <= 16 else 32
    block_k = 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), batch)
    with torch_device_fn.device(left.device):
        _triton_bmm_kernel[grid](
            left,
            right,
            out,
            left.stride(0),
            left.stride(1),
            left.stride(2),
            right.stride(0),
            right.stride(1),
            right.stride(2),
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=1 if block_m == 16 and block_n == 16 else 4,
        )
    return out.reshape(out_shape)


@libentry()
@triton.jit
def _gram_build_tiled_kernel(
    A,
    GRAM,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    tile_i = tl.program_id(0)
    tile_j = tl.program_id(1)
    batch = tl.program_id(2)
    offs_i = tile_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)
    rows = tl.arange(0, BLOCK_R)
    i_mask = offs_i < K
    j_mask = offs_j < K
    a_base = A + batch * M * N
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    for row_start in range(0, ROWS, BLOCK_R):
        chunk_rows = row_start + rows
        row_mask = chunk_rows < ROWS
        if TALL:
            lhs = tl.load(
                a_base + chunk_rows[None, :] * N + offs_i[:, None],
                mask=i_mask[:, None] & row_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            rhs = tl.load(
                a_base + chunk_rows[:, None] * N + offs_j[None, :],
                mask=row_mask[:, None] & j_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            lhs = tl.load(
                a_base + offs_i[:, None] * N + chunk_rows[None, :],
                mask=i_mask[:, None] & row_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            rhs = tl.load(
                a_base + offs_j[None, :] * N + chunk_rows[:, None],
                mask=row_mask[:, None] & j_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        acc += tl.dot(lhs, rhs, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        GRAM + batch * K * K + offs_i[:, None] * K + offs_j[None, :],
        acc,
        mask=i_mask[:, None] & j_mask[None, :],
    )


@libentry()
@triton.jit
def _cholesky_upper_kernel(
    GRAM,
    R,
    STATUS,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    base_g = GRAM + batch * K * K
    base_r = R + batch * K * K

    diag_vals = tl.load(
        base_g + cols * K + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    max_diag = tl.max(tl.abs(diag_vals), axis=0)
    tol = tl.maximum(max_diag * 1.0e-8, 1.0e-20)
    status = tl.full((), 0, dtype=tl.int32)
    finite_limit = 3.4028234663852886e38

    j = 0
    while j < K:
        row_mask = col_mask & (cols >= j)
        gram_row = tl.load(
            base_g + j * K + cols,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        diag = tl.load(base_g + j * K + j).to(tl.float32)

        p = 0
        while p < j:
            r_pj = tl.load(base_r + p * K + j).to(tl.float32)
            r_pcols = tl.load(
                base_r + p * K + cols,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            gram_row -= r_pj * r_pcols
            diag -= r_pj * r_pj
            p += 1

        good_diag = (diag == diag) & (tl.abs(diag) < finite_limit) & (diag > tol)
        pivot = tl.sqrt(tl.maximum(diag, tol))
        r_vals = gram_row / pivot
        r_vals = tl.where(cols == j, pivot, r_vals)
        r_vals = tl.where(row_mask, r_vals, 0.0)
        bad_vals = tl.sum(
            (((r_vals != r_vals) | (tl.abs(r_vals) >= finite_limit)) & row_mask).to(
                tl.int32
            ),
            axis=0,
        )
        status = tl.where(good_diag & (bad_vals == 0), status, 1)
        tl.store(base_r + j * K + cols, r_vals, mask=col_mask)
        j += 1

    tl.store(STATUS + batch, status)


def _tsqr_guard_fallback_svd(input):
    _, m, n = _svd_shape(input)
    k = min(m, n)
    if 16 <= k <= 64 and max(m, n) <= 1024:
        return _cyclic_jacobi_svd(input)
    if 64 < k <= 512 and max(m, n) <= 1024:
        return _blocked_jacobi_svd(input)
    return _unsupported_svd(
        input,
        True,
        True,
        "TSQR/Cholesky guard could not find a native Jacobi fallback.",
    )


def _tsqr_cholesky_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    gram = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    r = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    status = torch.empty((batch,), dtype=torch.int32, device=input.device)
    block_k = triton.next_power_of_2(k)
    block_tile = 32
    block_r = 64

    with torch_device_fn.device(input.device):
        _gram_build_tiled_kernel[
            (
                triton.cdiv(k, block_tile),
                triton.cdiv(k, block_tile),
                batch,
            )
        ](
            a,
            gram,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=tall,
            BLOCK_I=block_tile,
            BLOCK_J=block_tile,
            BLOCK_R=block_r,
            num_warps=4,
        )
        _cholesky_upper_kernel[(batch,)](
            gram,
            r,
            status,
            K=k,
            BLOCK_K=block_k,
            num_warps=4,
        )

    _, s, basis = svd(r, some=True, compute_uv=True)
    basis = basis.reshape(batch, k, k)
    s = s.reshape(batch, k)

    if tall:
        u = _triton_bmm(a, basis, (batch, m, k))
        v = basis
        projected = u
        projected_rows = m
    else:
        u = basis
        v = _triton_bmm(a.transpose(1, 2).contiguous(), basis, (batch, n, k))
        projected = v
        projected_rows = n

    with torch_device_fn.device(input.device):
        _normalize_projection_kernel[(batch, k)](
            projected,
            s,
            ROWS=projected_rows,
            K=k,
            BLOCK_R=triton.next_power_of_2(projected_rows),
            num_warps=1 if projected_rows <= 64 else 4,
        )

    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


@libentry()
@triton.jit
def _gram_build_kernel(
    A,
    GRAM,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    i = tl.arange(0, BLOCK_K)
    j = tl.arange(0, BLOCK_K)
    rows = tl.arange(0, BLOCK_R)
    k_mask = i < K
    a_base = A + batch * M * N
    acc = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)

    for row_start in range(0, ROWS, BLOCK_R):
        chunk_rows = row_start + rows
        row_mask = chunk_rows < ROWS

        if TALL:
            lhs = tl.load(
                a_base + chunk_rows[None, :] * N + i[:, None],
                mask=k_mask[:, None] & row_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            rhs = tl.load(
                a_base + chunk_rows[:, None] * N + j[None, :],
                mask=row_mask[:, None] & (j[None, :] < K),
                other=0.0,
            ).to(tl.float32)
        else:
            lhs = tl.load(
                a_base + i[:, None] * N + chunk_rows[None, :],
                mask=k_mask[:, None] & row_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            rhs = tl.load(
                a_base + j[None, :] * N + chunk_rows[:, None],
                mask=row_mask[:, None] & (j[None, :] < K),
                other=0.0,
            ).to(tl.float32)

        acc += tl.dot(lhs, rhs, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        GRAM + batch * K * K + i[:, None] * K + j[None, :],
        acc,
        mask=k_mask[:, None] & (j[None, :] < K),
    )


@libentry()
@triton.jit
def _gram_jacobi_sym_kernel(
    GRAM,
    EVECS,
    EVALS,
    K,
    SWEEPS,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    r = tl.arange(0, BLOCK_K)
    cidx = tl.arange(0, BLOCK_K)
    rr = r[:, None]
    cc = cidx[None, :]
    mask = (rr < K) & (cc < K)
    base = GRAM + batch * K * K

    g = tl.load(base + rr * K + cc, mask=mask, other=0.0).to(tl.float32)
    v = tl.where((rr == cc) & mask, 1.0, 0.0)
    eps = 1.0e-20

    sweep = 0
    while sweep < SWEEPS:
        p = 0
        while p < K - 1:
            q = p + 1
            while q < K:
                diag_p = tl.sum(tl.where((rr == p) & (cc == p), g, 0.0))
                diag_q = tl.sum(tl.where((rr == q) & (cc == q), g, 0.0))
                off = tl.sum(tl.where((rr == p) & (cc == q), g, 0.0))
                abs_off = tl.abs(off)
                threshold = 1.0e-7 * tl.sqrt(tl.abs(diag_p * diag_q) + eps)
                active = abs_off > threshold
                safe_off = tl.where(active, off, 1.0)
                tau = (diag_q - diag_p) / (2.0 * safe_off)
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                crot = tl.rsqrt(1.0 + t * t)
                srot = t * crot
                crot = tl.where(active, crot, 1.0)
                srot = tl.where(active, srot, 0.0)

                col_p = tl.sum(tl.where(cc == p, g, 0.0), axis=1)
                col_q = tl.sum(tl.where(cc == q, g, 0.0), axis=1)
                row_p = tl.sum(tl.where(rr == p, g, 0.0), axis=0)
                row_q = tl.sum(tl.where(rr == q, g, 0.0), axis=0)

                new_col_p = crot * col_p - srot * col_q
                new_col_q = srot * col_p + crot * col_q
                new_row_p = crot * row_p - srot * row_q
                new_row_q = srot * row_p + crot * row_q
                g = tl.where(cc == p, new_col_p[:, None], g)
                g = tl.where(cc == q, new_col_q[:, None], g)
                g = tl.where(rr == p, new_row_p[None, :], g)
                g = tl.where(rr == q, new_row_q[None, :], g)

                new_pp = (
                    crot * crot * diag_p
                    - 2.0 * crot * srot * off
                    + srot * srot * diag_q
                )
                new_qq = (
                    srot * srot * diag_p
                    + 2.0 * crot * srot * off
                    + crot * crot * diag_q
                )
                g = tl.where((rr == p) & (cc == p), new_pp, g)
                g = tl.where((rr == q) & (cc == q), new_qq, g)
                g = tl.where(((rr == p) & (cc == q)) | ((rr == q) & (cc == p)), 0.0, g)

                vec_p = tl.sum(tl.where(cc == p, v, 0.0), axis=1)
                vec_q = tl.sum(tl.where(cc == q, v, 0.0), axis=1)
                new_vec_p = crot * vec_p - srot * vec_q
                new_vec_q = srot * vec_p + crot * vec_q
                v = tl.where(cc == p, new_vec_p[:, None], v)
                v = tl.where(cc == q, new_vec_q[:, None], v)
                q += 1
            p += 1
        sweep += 1

    diag = tl.sum(tl.where(rr == cc, g, 0.0), axis=1)
    tl.store(EVALS + batch * K + r, diag, mask=r < K)
    tl.store(EVECS + batch * K * K + rr * K + cc, v, mask=mask)


@libentry()
@triton.jit
def _gram_sort_basis_kernel(
    EVALS,
    EVECS,
    BASIS,
    S,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_K)
    row_mask = rows < K
    eval_col = tl.maximum(tl.load(EVALS + batch * K + col), 0.0)
    rank = tl.full((), 0, dtype=tl.int32)
    for other in tl.static_range(0, K):
        eval_other = tl.maximum(tl.load(EVALS + batch * K + other), 0.0)
        rank += (
            (eval_other > eval_col) | ((eval_other == eval_col) & (other < col))
        ).to(tl.int32)

    vec = tl.load(
        EVECS + batch * K * K + rows * K + col,
        mask=row_mask,
        other=0.0,
    )
    tl.store(S + batch * K + rank, tl.sqrt(eval_col))
    tl.store(
        BASIS + batch * K * K + rows * K + rank,
        vec,
        mask=row_mask,
    )


@libentry()
@triton.jit
def _normalize_projection_kernel(
    Q,
    S,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    mask = rows < ROWS
    eps = 1.0e-20
    sval = tl.load(S + batch * K + col)
    vals = tl.load(Q + batch * ROWS * K + rows * K + col, mask=mask, other=0.0)
    vals = vals / tl.maximum(sval, eps)
    tl.store(Q + batch * ROWS * K + rows * K + col, vals, mask=mask)


@libentry()
@triton.jit
def _renorm_projection_update_s_kernel(
    Q,
    S,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    mask = rows < ROWS
    vals = tl.load(Q + batch * ROWS * K + rows * K + col, mask=mask, other=0.0)
    vals_f32 = vals.to(tl.float32)
    norm = tl.sqrt(tl.sum(vals_f32 * vals_f32, axis=0))
    inv_norm = tl.rsqrt(tl.maximum(norm * norm, 1.0e-40))
    basis = tl.where(rows == col, 1.0, 0.0)
    vals = tl.where(norm <= 1.0e-20, basis, vals * inv_norm)
    tl.store(S + batch * K + col, norm)
    tl.store(Q + batch * ROWS * K + rows * K + col, vals, mask=mask)


@libentry()
@triton.jit
def _complete_zero_projection_kernel(
    Q,
    S,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    mask = rows < ROWS
    eps = 1.0e-12
    sval = tl.load(S + batch * K + col)
    basis = tl.where(rows == col, 1.0, 0.0)
    old = tl.load(Q + batch * ROWS * K + rows * K + col, mask=mask, other=0.0)
    vals = tl.where(sval <= eps, basis, old)
    tl.store(Q + batch * ROWS * K + rows * K + col, vals, mask=mask)


def _gram_jacobi_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    gram = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    eigvecs = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    evals = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    basis = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)
    block_k = triton.next_power_of_2(k)
    block_r = min(triton.next_power_of_2(rows), 64 if k > 32 else 128)
    sweeps = 12 if k <= 17 else 10

    with torch_device_fn.device(input.device):
        _gram_build_kernel[(batch,)](
            a,
            gram,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=tall,
            BLOCK_K=block_k,
            BLOCK_R=block_r,
            num_warps=4,
        )
        _gram_jacobi_sym_kernel[(batch,)](
            gram,
            eigvecs,
            evals,
            k,
            sweeps,
            BLOCK_K=block_k,
            num_warps=4,
        )
    with torch_device_fn.device(input.device):
        _gram_sort_basis_kernel[(batch, k)](
            evals,
            eigvecs,
            basis,
            s,
            K=k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    if tall:
        u = _triton_bmm(a, basis, (batch, m, k))
        v = basis
        proj_rows = m
    else:
        a_t = a.transpose(1, 2).contiguous()
        v = _triton_bmm(a_t, basis, (batch, n, k))
        u = basis
        proj_rows = n

    with torch_device_fn.device(input.device):
        _renorm_projection_update_s_kernel[(batch, k)](
            u if tall else v,
            s,
            ROWS=proj_rows,
            K=k,
            BLOCK_R=triton.next_power_of_2(proj_rows),
            num_warps=1 if proj_rows <= 64 else 4,
        )
        _complete_zero_projection_kernel[(batch, k)](
            u if tall else v,
            s,
            ROWS=proj_rows,
            K=k,
            BLOCK_R=triton.next_power_of_2(proj_rows),
            num_warps=1 if proj_rows <= 64 else 4,
        )
        if k <= _GRAM_TALL_WIDE_MAX_K:
            _thin_reorthogonalize_kernel[(batch,)](
                u if tall else v,
                ROWS=proj_rows,
                K=k,
                BLOCK_R=triton.next_power_of_2(proj_rows),
                num_warps=1 if proj_rows <= 64 else 4,
            )

    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


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
    det = tl.maximum(0.0, aa * bbv - ab * ab)
    l1 = tl.where(l0 > eps, det / l0, 0.0)
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
def _rank2_svals_tiny_kernel(
    A,
    S,
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
    det = tl.maximum(0.0, aa * bbv - ab * ab)
    l1 = tl.where(l0 > 1.0e-20, det / l0, 0.0)
    tl.store(S + b * 2, tl.sqrt(l0), mask=bmask)
    tl.store(S + b * 2 + 1, tl.sqrt(l1), mask=bmask)


@libentry()
@triton.jit
def _rank2_svals_kernel(
    A,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_R)

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
    det = tl.maximum(0.0, aa * bb - ab * ab)
    l1 = tl.where(l0 > 1.0e-20, det / l0, 0.0)

    sbase = S + pid * 2
    tl.store(sbase, tl.sqrt(l0))
    tl.store(sbase + 1, tl.sqrt(l1))


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
    det = tl.maximum(0.0, aa * bb - ab * ab)
    l1 = tl.where(l0 > eps, det / l0, 0.0)
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
        basis0 = tl.where(offs == 0, 1.0, 0.0)
        basis1 = tl.where(offs == 1, 1.0, 0.0)
        u0 = tl.where(s0 > eps, u0, basis0)

        u1 = (x * vx1 + y * vy1) * inv_s1
        u0_first = tl.sum(tl.where(offs == 0, u0, 0.0))
        anchor = tl.where(tl.abs(u0_first) < 0.70710678, basis0, basis1)
        dot = tl.sum(anchor * u0)
        fallback_u1 = anchor - dot * u0
        fallback_norm = tl.sum(fallback_u1 * fallback_u1)
        fallback_u1 = fallback_u1 * tl.rsqrt(fallback_norm + eps)
        u1 = tl.where(s1 > s0 * 5.0e-4, u1, fallback_u1)
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
        basis0 = tl.where(offs == 0, 1.0, 0.0)
        basis1 = tl.where(offs == 1, 1.0, 0.0)
        v0 = tl.where(s0 > eps, v0, basis0)

        v1 = (x * vx1 + y * vy1) * inv_s1
        v0_first = tl.sum(tl.where(offs == 0, v0, 0.0))
        anchor = tl.where(tl.abs(v0_first) < 0.70710678, basis0, basis1)
        dot = tl.sum(anchor * v0)
        fallback_v1 = anchor - dot * v0
        fallback_norm = tl.sum(fallback_v1 * fallback_v1)
        fallback_v1 = fallback_v1 * tl.rsqrt(fallback_norm + eps)
        v1 = tl.where(s1 > s0 * 5.0e-4, v1, fallback_v1)
        tl.store(vbase + offs * 2, v0, mask=mask)
        tl.store(vbase + offs * 2 + 1, v1, mask=mask)


def _rank2_svd(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    u = torch.empty((batch, m, 2), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, 2), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, 2), dtype=input.dtype, device=input.device)
    largest = max(m, n)
    block_r = triton.next_power_of_2(largest)
    with torch_device_fn.device(input.device):
        if largest <= 16 and batch >= 16:
            if largest <= 2:
                block_b = 8
            elif largest == 16:
                block_b = 2 if m >= n else 8
            else:
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


def _rank2_singular_values(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    s = torch.empty((batch, 2), dtype=input.dtype, device=input.device)
    largest = max(m, n)
    block_r = triton.next_power_of_2(largest)
    with torch_device_fn.device(input.device):
        if largest <= 16 and batch >= 16:
            if largest <= 2:
                block_b = 8
            elif largest == 16:
                block_b = 2 if m >= n else 8
            else:
                block_b = 16
            _rank2_svals_tiny_kernel[(triton.cdiv(batch, block_b),)](
                a,
                s,
                BATCH=batch,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_B=block_b,
                BLOCK_R=block_r,
                num_warps=1,
            )
        else:
            _rank2_svals_kernel[(batch,)](
                a,
                s,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_R=block_r,
                num_warps=1 if block_r <= 64 else 4,
            )
    return s.reshape(*input.shape[:-2], 2)


def _small_jacobi_singular_values(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)
    block_r = triton.next_power_of_2(rows)
    block_k = triton.next_power_of_2(k)
    sweeps = 3 if k <= 4 else 5
    with torch_device_fn.device(input.device):
        _small_jacobi_svals_kernel[(batch,)](
            a,
            a_work,
            s,
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
    return s.reshape(*input.shape[:-2], k)


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
    sweeps = 3 if k <= 4 else 5
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


@libentry()
@triton.jit
def _cyclic_jacobi_init_a_kernel(
    A,
    A_WORK,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    a_base = A + batch * M * N
    aw_base = A_WORK + batch * K * ROWS

    if TALL:
        vals = tl.load(a_base + rows * N + col, mask=row_mask, other=0.0).to(tl.float32)
    else:
        vals = tl.load(a_base + col * N + rows, mask=row_mask, other=0.0).to(tl.float32)
    tl.store(aw_base + col * ROWS + rows, vals, mask=row_mask)


@libentry()
@triton.jit
def _cyclic_jacobi_init_kernel(
    A,
    A_WORK,
    V_WORK,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    basis_cols = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    basis_mask = basis_cols < K
    a_base = A + batch * M * N
    aw_base = A_WORK + batch * K * ROWS
    vw_base = V_WORK + batch * K * K

    if TALL:
        vals = tl.load(a_base + rows * N + col, mask=row_mask, other=0.0).to(tl.float32)
    else:
        vals = tl.load(a_base + col * N + rows, mask=row_mask, other=0.0).to(tl.float32)
    tl.store(aw_base + col * ROWS + rows, vals, mask=row_mask)

    ident = tl.where(basis_cols == col, 1.0, 0.0)
    tl.store(vw_base + col * K + basis_cols, ident, mask=basis_mask)


@libentry()
@triton.jit
def _cyclic_jacobi_pair_kernel(
    A_WORK,
    V_WORK,
    STEP,
    K: tl.constexpr,
    ROUND: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    pair = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_K)
    ring = ROUND - 1

    pos_p = pair
    pos_q = ROUND - 1 - pair
    p = tl.where(pos_p == 0, 0, ((pos_p + ring - STEP - 1) % ring) + 1)
    q = tl.where(pos_q == 0, 0, ((pos_q + ring - STEP - 1) % ring) + 1)
    valid_pair = (p < K) & (q < K)
    swap = p > q
    p2 = tl.where(swap, q, p)
    q2 = tl.where(swap, p, q)
    row_mask = (rows < ROWS) & valid_pair
    col_mask = (cols < K) & valid_pair

    aw_base = A_WORK + batch * K * ROWS
    vw_base = V_WORK + batch * K * K
    ap = tl.load(aw_base + p2 * ROWS + rows, mask=row_mask, other=0.0)
    aq = tl.load(aw_base + q2 * ROWS + rows, mask=row_mask, other=0.0)
    alpha = tl.sum(ap * ap)
    beta = tl.sum(aq * aq)
    gamma = tl.sum(ap * aq)
    eps = 1.0e-20
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
    tl.store(aw_base + p2 * ROWS + rows, new_ap, mask=row_mask)
    tl.store(aw_base + q2 * ROWS + rows, new_aq, mask=row_mask)

    vp = tl.load(vw_base + p2 * K + cols, mask=col_mask, other=0.0)
    vq = tl.load(vw_base + q2 * K + cols, mask=col_mask, other=0.0)
    new_vp = c * vp - s_rot * vq
    new_vq = s_rot * vp + c * vq
    tl.store(vw_base + p2 * K + cols, new_vp, mask=col_mask)
    tl.store(vw_base + q2 * K + cols, new_vq, mask=col_mask)


@libentry()
@triton.jit
def _serial_cyclic_jacobi_kernel(
    A_WORK,
    V_WORK,
    K,
    ROUND,
    ROWS: tl.constexpr,
    SWEEPS,
    TAIL_STEPS,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_K)
    row_base_mask = rows < ROWS
    col_base_mask = cols < K
    aw_base = A_WORK + batch * K * ROWS
    vw_base = V_WORK + batch * K * K
    eps = 1.0e-20
    ring = ROUND - 1
    half_round = ROUND // 2

    sweep = 0
    while sweep < SWEEPS:
        step = 0
        while step < ROUND - 1:
            pair = 0
            while pair < half_round:
                pos_p = pair
                pos_q = ROUND - 1 - pair
                p = tl.where(pos_p == 0, 0, ((pos_p + ring - step - 1) % ring) + 1)
                q = tl.where(pos_q == 0, 0, ((pos_q + ring - step - 1) % ring) + 1)
                valid_pair = (p < K) & (q < K)
                swap = p > q
                p2 = tl.where(swap, q, p)
                q2 = tl.where(swap, p, q)
                row_mask = row_base_mask & valid_pair
                col_mask = col_base_mask & valid_pair

                ap = tl.load(aw_base + p2 * ROWS + rows, mask=row_mask, other=0.0)
                aq = tl.load(aw_base + q2 * ROWS + rows, mask=row_mask, other=0.0)
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
                tl.store(aw_base + p2 * ROWS + rows, new_ap, mask=row_mask)
                tl.store(aw_base + q2 * ROWS + rows, new_aq, mask=row_mask)

                vp = tl.load(vw_base + p2 * K + cols, mask=col_mask, other=0.0)
                vq = tl.load(vw_base + q2 * K + cols, mask=col_mask, other=0.0)
                new_vp = c * vp - s_rot * vq
                new_vq = s_rot * vp + c * vq
                tl.store(vw_base + p2 * K + cols, new_vp, mask=col_mask)
                tl.store(vw_base + q2 * K + cols, new_vq, mask=col_mask)
                pair += 1
            step += 1
        sweep += 1

    step = 0
    while step < TAIL_STEPS:
        pair = 0
        while pair < half_round:
            pos_p = pair
            pos_q = ROUND - 1 - pair
            p = tl.where(pos_p == 0, 0, ((pos_p + ring - step - 1) % ring) + 1)
            q = tl.where(pos_q == 0, 0, ((pos_q + ring - step - 1) % ring) + 1)
            valid_pair = (p < K) & (q < K)
            swap = p > q
            p2 = tl.where(swap, q, p)
            q2 = tl.where(swap, p, q)
            row_mask = row_base_mask & valid_pair
            col_mask = col_base_mask & valid_pair

            ap = tl.load(aw_base + p2 * ROWS + rows, mask=row_mask, other=0.0)
            aq = tl.load(aw_base + q2 * ROWS + rows, mask=row_mask, other=0.0)
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
            tl.store(aw_base + p2 * ROWS + rows, new_ap, mask=row_mask)
            tl.store(aw_base + q2 * ROWS + rows, new_aq, mask=row_mask)

            vp = tl.load(vw_base + p2 * K + cols, mask=col_mask, other=0.0)
            vq = tl.load(vw_base + q2 * K + cols, mask=col_mask, other=0.0)
            new_vp = c * vp - s_rot * vq
            new_vq = s_rot * vp + c * vq
            tl.store(vw_base + p2 * K + cols, new_vp, mask=col_mask)
            tl.store(vw_base + q2 * K + cols, new_vq, mask=col_mask)
            pair += 1
        step += 1


@libentry()
@triton.jit
def _cyclic_jacobi_norm_kernel(
    A_WORK,
    S_WORK,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    mask = rows < ROWS
    aw_base = A_WORK + batch * K * ROWS
    vals = tl.load(aw_base + col * ROWS + rows, mask=mask, other=0.0)
    norm = tl.sqrt(tl.sum(vals * vals))
    tl.store(S_WORK + batch * K + col, norm)


@libentry()
@triton.jit
def _cyclic_jacobi_finalize_kernel(
    A_WORK,
    V_WORK,
    S_WORK,
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
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    basis_cols = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    basis_mask = basis_cols < K
    eps = 1.0e-20

    s_col = tl.load(S_WORK + batch * K + col)
    rank = tl.full((), 0, dtype=tl.int32)
    for other in tl.static_range(0, K):
        s_other = tl.load(S_WORK + batch * K + other)
        rank += ((s_other > s_col) | ((s_other == s_col) & (other < col))).to(tl.int32)

    aw_base = A_WORK + batch * K * ROWS
    vw_base = V_WORK + batch * K * K
    col_vals = tl.load(aw_base + col * ROWS + rows, mask=row_mask, other=0.0)
    inv_norm = tl.where(s_col > eps, 1.0 / s_col, 0.0)
    basis = tl.load(vw_base + col * K + basis_cols, mask=basis_mask, other=0.0)
    tl.store(S + batch * K + rank, s_col)

    if TALL:
        tl.store(
            U + batch * M * K + rows * K + rank,
            col_vals * inv_norm,
            mask=row_mask,
        )
        tl.store(
            V + batch * N * K + basis_cols * K + rank,
            basis,
            mask=basis_mask,
        )
    else:
        tl.store(
            U + batch * M * K + basis_cols * K + rank,
            basis,
            mask=basis_mask,
        )
        tl.store(
            V + batch * N * K + rows * K + rank,
            col_vals * inv_norm,
            mask=row_mask,
        )


@libentry()
@triton.jit
def _blocked_jacobi_pair_svals_kernel(
    A_WORK,
    STEP,
    K: tl.constexpr,
    ROUND: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    pair = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    ring = ROUND - 1

    pos_p = pair
    pos_q = ROUND - 1 - pair
    p = tl.where(pos_p == 0, 0, ((pos_p + ring - STEP - 1) % ring) + 1)
    q = tl.where(pos_q == 0, 0, ((pos_q + ring - STEP - 1) % ring) + 1)
    valid_pair = (p < K) & (q < K)
    swap = p > q
    p2 = tl.where(swap, q, p)
    q2 = tl.where(swap, p, q)
    row_mask = (rows < ROWS) & valid_pair

    aw_base = A_WORK + batch * K * ROWS
    ap = tl.load(aw_base + p2 * ROWS + rows, mask=row_mask, other=0.0)
    aq = tl.load(aw_base + q2 * ROWS + rows, mask=row_mask, other=0.0)
    alpha = tl.sum(ap * ap)
    beta = tl.sum(aq * aq)
    gamma = tl.sum(ap * aq)
    eps = 1.0e-20
    abs_gamma = tl.abs(gamma)
    threshold = 1.0e-7 * tl.sqrt(alpha * beta + eps)
    active = abs_gamma > threshold
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = tl.rsqrt(1.0 + t * t)
    s_rot = t * c
    c = tl.where(active & valid_pair, c, 1.0)
    s_rot = tl.where(active & valid_pair, s_rot, 0.0)

    new_ap = c * ap - s_rot * aq
    new_aq = s_rot * ap + c * aq
    tl.store(aw_base + p2 * ROWS + rows, new_ap, mask=row_mask)
    tl.store(aw_base + q2 * ROWS + rows, new_aq, mask=row_mask)


@libentry()
@triton.jit
def _blocked_jacobi_pair_a_kernel(
    A_WORK,
    ROT_C,
    ROT_S,
    STEP,
    K: tl.constexpr,
    ROUND: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    pair = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    ring = ROUND - 1

    pos_p = pair
    pos_q = ROUND - 1 - pair
    p = tl.where(pos_p == 0, 0, ((pos_p + ring - STEP - 1) % ring) + 1)
    q = tl.where(pos_q == 0, 0, ((pos_q + ring - STEP - 1) % ring) + 1)
    valid_pair = (p < K) & (q < K)
    swap = p > q
    p2 = tl.where(swap, q, p)
    q2 = tl.where(swap, p, q)
    row_mask = (rows < ROWS) & valid_pair

    aw_base = A_WORK + batch * K * ROWS
    ap = tl.load(aw_base + p2 * ROWS + rows, mask=row_mask, other=0.0)
    aq = tl.load(aw_base + q2 * ROWS + rows, mask=row_mask, other=0.0)
    alpha = tl.sum(ap * ap)
    beta = tl.sum(aq * aq)
    gamma = tl.sum(ap * aq)
    eps = 1.0e-20
    abs_gamma = tl.abs(gamma)
    threshold = 1.0e-7 * tl.sqrt(alpha * beta + eps)
    active = abs_gamma > threshold
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = tl.rsqrt(1.0 + t * t)
    s_rot = t * c
    c = tl.where(active & valid_pair, c, 1.0)
    s_rot = tl.where(active & valid_pair, s_rot, 0.0)

    new_ap = c * ap - s_rot * aq
    new_aq = s_rot * ap + c * aq
    tl.store(aw_base + p2 * ROWS + rows, new_ap, mask=row_mask)
    tl.store(aw_base + q2 * ROWS + rows, new_aq, mask=row_mask)

    rot_base = batch * (ROUND // 2) + pair
    tl.store(ROT_C + rot_base, c)
    tl.store(ROT_S + rot_base, s_rot)


@libentry()
@triton.jit
def _hier_block_jacobi_pair_a_kernel(
    A_WORK,
    STEP,
    K: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    ROUND_BLOCKS: tl.constexpr,
    ROWS: tl.constexpr,
    TILE_B: tl.constexpr,
    TILE_COLS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOCAL_SWEEPS: tl.constexpr,
):
    batch = tl.program_id(0)
    pair = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    local_cols = tl.arange(0, TILE_COLS)
    ring = ROUND_BLOCKS - 1

    pos_p = pair
    pos_q = ROUND_BLOCKS - 1 - pair
    p_block = tl.where(pos_p == 0, 0, ((pos_p + ring - STEP - 1) % ring) + 1)
    q_block = tl.where(pos_q == 0, 0, ((pos_q + ring - STEP - 1) % ring) + 1)
    valid_pair = (p_block < K_BLOCKS) & (q_block < K_BLOCKS)
    p2 = tl.minimum(p_block, q_block)
    q2 = tl.maximum(p_block, q_block)

    col_ids = tl.where(
        local_cols < TILE_B,
        p2 * TILE_B + local_cols,
        q2 * TILE_B + local_cols - TILE_B,
    )
    row_mask = rows < ROWS
    col_mask = (col_ids < K) & valid_pair
    aw_base = A_WORK + batch * K * ROWS
    vals = tl.load(
        aw_base + col_ids[:, None] * ROWS + rows[None, :],
        mask=col_mask[:, None] & row_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    col_axis = local_cols[:, None]
    eps = 1.0e-20

    for _ in tl.static_range(0, LOCAL_SWEEPS):
        for p in tl.static_range(0, TILE_COLS):
            for q in tl.static_range(p + 1, TILE_COLS):
                ap = tl.sum(tl.where(col_axis == p, vals, 0.0), axis=0)
                aq = tl.sum(tl.where(col_axis == q, vals, 0.0), axis=0)
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
                c = tl.where(active & valid_pair, c, 1.0)
                s_rot = tl.where(active & valid_pair, s_rot, 0.0)

                new_ap = c * ap - s_rot * aq
                new_aq = s_rot * ap + c * aq
                vals = tl.where(col_axis == p, new_ap[None, :], vals)
                vals = tl.where(col_axis == q, new_aq[None, :], vals)

    tl.store(
        aw_base + col_ids[:, None] * ROWS + rows[None, :],
        vals,
        mask=col_mask[:, None] & row_mask[None, :],
    )


@libentry()
@triton.jit
def _blocked_jacobi_apply_v_kernel(
    V_WORK,
    ROT_C,
    ROT_S,
    STEP,
    K: tl.constexpr,
    ROUND: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    batch = tl.program_id(0)
    pair = tl.program_id(1)
    block = tl.program_id(2)
    cols = block * BLOCK_V + tl.arange(0, BLOCK_V)
    ring = ROUND - 1

    pos_p = pair
    pos_q = ROUND - 1 - pair
    p = tl.where(pos_p == 0, 0, ((pos_p + ring - STEP - 1) % ring) + 1)
    q = tl.where(pos_q == 0, 0, ((pos_q + ring - STEP - 1) % ring) + 1)
    valid_pair = (p < K) & (q < K)
    swap = p > q
    p2 = tl.where(swap, q, p)
    q2 = tl.where(swap, p, q)
    mask = (cols < K) & valid_pair

    rot_base = batch * (ROUND // 2) + pair
    c = tl.load(ROT_C + rot_base)
    s_rot = tl.load(ROT_S + rot_base)
    vw_base = V_WORK + batch * K * K
    vp = tl.load(vw_base + p2 * K + cols, mask=mask, other=0.0)
    vq = tl.load(vw_base + q2 * K + cols, mask=mask, other=0.0)
    new_vp = c * vp - s_rot * vq
    new_vq = s_rot * vp + c * vq
    tl.store(vw_base + p2 * K + cols, new_vp, mask=mask)
    tl.store(vw_base + q2 * K + cols, new_vq, mask=mask)


@libentry()
@triton.jit
def _blocked_jacobi_rank_kernel(
    S_WORK,
    RANKS,
    S,
    K,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    s_col = tl.load(S_WORK + batch * K + col)
    rank = tl.full((), 0, dtype=tl.int32)
    other = 0
    while other < K:
        s_other = tl.load(S_WORK + batch * K + other)
        rank += ((s_other > s_col) | ((s_other == s_col) & (other < col))).to(tl.int32)
        other += 1
    tl.store(RANKS + batch * K + col, rank)
    tl.store(S + batch * K + rank, s_col)


@libentry()
@triton.jit
def _blocked_jacobi_store_projected_kernel(
    A_WORK,
    S_WORK,
    RANKS,
    PROJECTED,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    block = tl.program_id(2)
    rows = block * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = rows < OUT_ROWS
    rank = tl.load(RANKS + batch * K + col)
    s_col = tl.load(S_WORK + batch * K + col)
    eps = 1.0e-20
    vals = tl.load(
        A_WORK + batch * K * ROWS + col * ROWS + rows,
        mask=mask,
        other=0.0,
    )
    vals = vals / tl.maximum(s_col, eps)
    basis = tl.where(rows == rank, 1.0, 0.0)
    vals = tl.where(s_col <= eps, basis, vals)
    tl.store(
        PROJECTED + batch * OUT_ROWS * K + rows * K + rank,
        vals,
        mask=mask,
    )


@libentry()
@triton.jit
def _blocked_jacobi_store_basis_kernel(
    V_WORK,
    RANKS,
    BASIS,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    block = tl.program_id(2)
    rows = block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = rows < K
    rank = tl.load(RANKS + batch * K + col)
    vals = tl.load(
        V_WORK + batch * K * K + col * K + rows,
        mask=mask,
        other=0.0,
    )
    tl.store(
        BASIS + batch * K * K + rows * K + rank,
        vals,
        mask=mask,
    )


@libentry()
@triton.jit
def _thin_reorthogonalize_kernel(
    Q,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < ROWS
    base = Q + batch * ROWS * K
    eps = 1.0e-20

    for j in tl.static_range(0, K):
        vec = tl.load(base + rows * K + j, mask=row_mask, other=0.0).to(tl.float32)

        for prev in tl.static_range(0, K):
            if prev < j:
                q_prev = tl.load(base + rows * K + prev, mask=row_mask, other=0.0).to(
                    tl.float32
                )
                coeff = tl.sum(vec * q_prev)
                vec = vec - coeff * q_prev

        for prev in tl.static_range(0, K):
            if prev < j:
                q_prev = tl.load(base + rows * K + prev, mask=row_mask, other=0.0).to(
                    tl.float32
                )
                coeff = tl.sum(vec * q_prev)
                vec = vec - coeff * q_prev

        norm = tl.sqrt(tl.sum(vec * vec))
        basis = tl.where(rows == j, 1.0, 0.0)
        vec = tl.where(norm > eps, vec, basis)

        for prev in tl.static_range(0, K):
            if prev < j:
                q_prev = tl.load(base + rows * K + prev, mask=row_mask, other=0.0).to(
                    tl.float32
                )
                coeff = tl.sum(vec * q_prev)
                vec = vec - coeff * q_prev

        norm = tl.sqrt(tl.sum(vec * vec))
        vec = vec / tl.maximum(norm, eps)
        tl.store(base + rows * K + j, vec, mask=row_mask)


def _cyclic_jacobi_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    v_work = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    s_work = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    u = torch.empty((batch, m, k), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, k), dtype=input.dtype, device=input.device)
    block_r = triton.next_power_of_2(rows)
    block_k = triton.next_power_of_2(k)
    sweeps = 6 if k == 32 else 8 if k < 32 else 12
    tail_steps = 20 if k == 32 else 0
    round_size = k if k % 2 == 0 else k + 1
    serial_medium = 16 <= k <= 32 and rows <= 64 and batch <= 32
    with torch_device_fn.device(input.device):
        _cyclic_jacobi_init_kernel[(batch, k)](
            a,
            a_work,
            v_work,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=m >= n,
            BLOCK_R=block_r,
            BLOCK_K=block_k,
            num_warps=1 if block_r <= 64 else 4,
        )
        if serial_medium:
            _serial_cyclic_jacobi_kernel[(batch,)](
                a_work,
                v_work,
                K=k,
                ROUND=round_size,
                ROWS=rows,
                SWEEPS=sweeps,
                TAIL_STEPS=tail_steps,
                BLOCK_R=block_r,
                BLOCK_K=block_k,
                num_warps=1,
            )
        else:
            for _ in range(sweeps):
                for step in range(round_size - 1):
                    _cyclic_jacobi_pair_kernel[(batch, round_size // 2)](
                        a_work,
                        v_work,
                        step,
                        K=k,
                        ROUND=round_size,
                        ROWS=rows,
                        BLOCK_R=block_r,
                        BLOCK_K=block_k,
                        num_warps=1 if block_r <= 64 else 4,
                    )
        _cyclic_jacobi_norm_kernel[(batch, k)](
            a_work,
            s_work,
            K=k,
            ROWS=rows,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        _cyclic_jacobi_finalize_kernel[(batch, k)](
            a_work,
            v_work,
            s_work,
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
            num_warps=1 if block_r <= 64 else 4,
        )
    if k <= 17 and rows <= 64:
        with torch_device_fn.device(input.device):
            if m >= n:
                _thin_reorthogonalize_kernel[(batch,)](
                    v,
                    ROWS=n,
                    K=k,
                    BLOCK_R=triton.next_power_of_2(n),
                    num_warps=1,
                )
            else:
                _thin_reorthogonalize_kernel[(batch,)](
                    u,
                    ROWS=m,
                    K=k,
                    BLOCK_R=triton.next_power_of_2(m),
                    num_warps=1,
                )

        if m >= n:
            u = _triton_bmm(a, v, (batch, m, k))
            projected = u
            projected_rows = m
        else:
            a_t = a.transpose(1, 2).contiguous()
            v = _triton_bmm(a_t, u, (batch, n, k))
            projected = v
            projected_rows = n
        with torch_device_fn.device(input.device):
            _normalize_projection_kernel[(batch, k)](
                projected,
                s,
                ROWS=projected_rows,
                K=k,
                BLOCK_R=triton.next_power_of_2(projected_rows),
                num_warps=1,
            )
            _complete_zero_projection_kernel[(batch, k)](
                projected,
                s,
                ROWS=projected_rows,
                K=k,
                BLOCK_R=triton.next_power_of_2(projected_rows),
                num_warps=1,
            )
            if batch > 1 and k <= 16:
                _thin_reorthogonalize_kernel[(batch,)](
                    projected,
                    ROWS=projected_rows,
                    K=k,
                    BLOCK_R=triton.next_power_of_2(projected_rows),
                    num_warps=1,
                )
    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


def _projected_jacobi_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    s_work = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    ranks = torch.empty((batch, k), dtype=torch.int32, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)

    projected_rows = m if tall else n
    projected = torch.empty(
        (batch, projected_rows, k), dtype=input.dtype, device=input.device
    )
    block_r = triton.next_power_of_2(rows)
    sweeps = 10 if k >= 128 else 8
    round_size = k if k % 2 == 0 else k + 1
    half_round = round_size // 2
    rot_c = torch.empty((batch, half_round), dtype=torch.float32, device=input.device)
    rot_s = torch.empty((batch, half_round), dtype=torch.float32, device=input.device)

    with torch_device_fn.device(input.device):
        _cyclic_jacobi_init_a_kernel[(batch, k)](
            a,
            a_work,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=tall,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        for _ in range(sweeps):
            for step in range(round_size - 1):
                _blocked_jacobi_pair_a_kernel[(batch, half_round)](
                    a_work,
                    rot_c,
                    rot_s,
                    step,
                    K=k,
                    ROUND=round_size,
                    ROWS=rows,
                    BLOCK_R=block_r,
                    num_warps=1 if block_r <= 64 else 4,
                )
        _cyclic_jacobi_norm_kernel[(batch, k)](
            a_work,
            s_work,
            K=k,
            ROWS=rows,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        _blocked_jacobi_rank_kernel[(batch, k)](
            s_work,
            ranks,
            s,
            k,
            num_warps=1,
        )
        _blocked_jacobi_store_projected_kernel[
            (batch, k, triton.cdiv(projected_rows, block_r))
        ](
            a_work,
            s_work,
            ranks,
            projected,
            K=k,
            ROWS=rows,
            OUT_ROWS=projected_rows,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )

    if tall:
        u = projected
        v = _triton_bmm(a.transpose(1, 2).contiguous(), u, (batch, n, k))
        normalized = v
        normalized_rows = n
    else:
        v = projected
        u = _triton_bmm(a, v, (batch, m, k))
        normalized = u
        normalized_rows = m

    with torch_device_fn.device(input.device):
        _normalize_projection_kernel[(batch, k)](
            normalized,
            s,
            ROWS=normalized_rows,
            K=k,
            BLOCK_R=triton.next_power_of_2(normalized_rows),
            num_warps=1 if normalized_rows <= 64 else 4,
        )

    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


def _blocked_jacobi_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    v_work = torch.empty((batch, k, k), dtype=torch.float32, device=input.device)
    s_work = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    ranks = torch.empty((batch, k), dtype=torch.int32, device=input.device)
    u = torch.empty((batch, m, k), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, k), dtype=input.dtype, device=input.device)

    block_r = triton.next_power_of_2(rows)
    block_k = triton.next_power_of_2(k)
    block_v = 64
    sweeps = 14 if k > 256 else 10
    round_size = k if k % 2 == 0 else k + 1
    half_round = round_size // 2
    rot_c = torch.empty((batch, half_round), dtype=torch.float32, device=input.device)
    rot_s = torch.empty((batch, half_round), dtype=torch.float32, device=input.device)
    with torch_device_fn.device(input.device):
        _cyclic_jacobi_init_kernel[(batch, k)](
            a,
            a_work,
            v_work,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=m >= n,
            BLOCK_R=block_r,
            BLOCK_K=block_k,
            num_warps=1 if block_r <= 64 else 4,
        )
        for _ in range(sweeps):
            for step in range(round_size - 1):
                _blocked_jacobi_pair_a_kernel[(batch, half_round)](
                    a_work,
                    rot_c,
                    rot_s,
                    step,
                    K=k,
                    ROUND=round_size,
                    ROWS=rows,
                    BLOCK_R=block_r,
                    num_warps=1 if block_r <= 64 else 4,
                )
                _blocked_jacobi_apply_v_kernel[
                    (batch, half_round, triton.cdiv(k, block_v))
                ](
                    v_work,
                    rot_c,
                    rot_s,
                    step,
                    K=k,
                    ROUND=round_size,
                    BLOCK_V=block_v,
                    num_warps=1,
                )
        _cyclic_jacobi_norm_kernel[(batch, k)](
            a_work,
            s_work,
            K=k,
            ROWS=rows,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        _blocked_jacobi_rank_kernel[(batch, k)](
            s_work,
            ranks,
            s,
            k,
            num_warps=1,
        )
        if m >= n:
            _blocked_jacobi_store_projected_kernel[(batch, k, triton.cdiv(m, block_r))](
                a_work,
                s_work,
                ranks,
                u,
                K=k,
                ROWS=rows,
                OUT_ROWS=m,
                BLOCK_R=block_r,
                num_warps=1 if block_r <= 64 else 4,
            )
            _blocked_jacobi_store_basis_kernel[(batch, k, triton.cdiv(n, block_v))](
                v_work,
                ranks,
                v,
                K=k,
                BLOCK_V=block_v,
                num_warps=1,
            )
        else:
            _blocked_jacobi_store_basis_kernel[(batch, k, triton.cdiv(m, block_v))](
                v_work,
                ranks,
                u,
                K=k,
                BLOCK_V=block_v,
                num_warps=1,
            )
            _blocked_jacobi_store_projected_kernel[(batch, k, triton.cdiv(n, block_r))](
                a_work,
                s_work,
                ranks,
                v,
                K=k,
                ROWS=rows,
                OUT_ROWS=n,
                BLOCK_R=block_r,
                num_warps=1 if block_r <= 64 else 4,
            )

    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


def _blocked_jacobi_square_project_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    s_work = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    ranks = torch.empty((batch, k), dtype=torch.int32, device=input.device)
    u = torch.empty((batch, m, k), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)

    block_r = triton.next_power_of_2(rows)
    sweeps = 12 if k <= 256 else 16
    round_size = k if k % 2 == 0 else k + 1
    half_round = round_size // 2
    rot_c = torch.empty((batch, half_round), dtype=torch.float32, device=input.device)
    rot_s = torch.empty((batch, half_round), dtype=torch.float32, device=input.device)
    with torch_device_fn.device(input.device):
        _cyclic_jacobi_init_a_kernel[(batch, k)](
            a,
            a_work,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=True,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        for _ in range(sweeps):
            for step in range(round_size - 1):
                _blocked_jacobi_pair_a_kernel[(batch, half_round)](
                    a_work,
                    rot_c,
                    rot_s,
                    step,
                    K=k,
                    ROUND=round_size,
                    ROWS=rows,
                    BLOCK_R=block_r,
                    num_warps=1 if block_r <= 64 else 4,
                )
        _cyclic_jacobi_norm_kernel[(batch, k)](
            a_work,
            s_work,
            K=k,
            ROWS=rows,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        _blocked_jacobi_rank_kernel[(batch, k)](
            s_work,
            ranks,
            s,
            k,
            num_warps=1,
        )
        _blocked_jacobi_store_projected_kernel[(batch, k, triton.cdiv(m, block_r))](
            a_work,
            s_work,
            ranks,
            u,
            K=k,
            ROWS=rows,
            OUT_ROWS=m,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )

    a_t = a.transpose(1, 2).contiguous()
    v = _triton_bmm(a_t, u, (batch, n, k))
    with torch_device_fn.device(input.device):
        _renorm_projection_update_s_kernel[(batch, k)](
            v,
            s,
            ROWS=n,
            K=k,
            BLOCK_R=triton.next_power_of_2(n),
            num_warps=1 if n <= 64 else 4,
        )

    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


def _hier_block_jacobi_square_project_svd(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    s_work = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    ranks = torch.empty((batch, k), dtype=torch.int32, device=input.device)
    u = torch.empty((batch, m, k), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)

    tile_b = 4 if k == 512 else 2
    if m != n or k % tile_b != 0:
        return _unsupported_svd(
            input,
            True,
            True,
            "Hierarchical block Jacobi supports square matrices with "
            "k divisible by two.",
        )

    block_r = triton.next_power_of_2(rows)
    block_count = k // tile_b
    round_blocks = block_count if block_count % 2 == 0 else block_count + 1
    half_round_blocks = round_blocks // 2
    sweep_count = 10 if k <= 256 else 12
    tile_cols = tile_b * 2
    with torch_device_fn.device(input.device):
        _cyclic_jacobi_init_a_kernel[(batch, k)](
            a,
            a_work,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=True,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        for _ in range(sweep_count):
            for step in range(round_blocks - 1):
                _hier_block_jacobi_pair_a_kernel[(batch, half_round_blocks)](
                    a_work,
                    step,
                    K=k,
                    K_BLOCKS=block_count,
                    ROUND_BLOCKS=round_blocks,
                    ROWS=rows,
                    TILE_B=tile_b,
                    TILE_COLS=tile_cols,
                    BLOCK_R=block_r,
                    LOCAL_SWEEPS=1,
                    num_warps=4,
                )
        _cyclic_jacobi_norm_kernel[(batch, k)](
            a_work,
            s_work,
            K=k,
            ROWS=rows,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        _blocked_jacobi_rank_kernel[(batch, k)](
            s_work,
            ranks,
            s,
            k,
            num_warps=1,
        )
        _blocked_jacobi_store_projected_kernel[(batch, k, triton.cdiv(m, block_r))](
            a_work,
            s_work,
            ranks,
            u,
            K=k,
            ROWS=rows,
            OUT_ROWS=m,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )

    a_t = a.transpose(1, 2).contiguous()
    v = _triton_bmm(a_t, u, (batch, n, k))
    with torch_device_fn.device(input.device):
        _renorm_projection_update_s_kernel[(batch, k)](
            v,
            s,
            ROWS=n,
            K=k,
            BLOCK_R=triton.next_power_of_2(n),
            num_warps=1 if n <= 64 else 4,
        )

    return (
        u.reshape(*input.shape[:-2], m, k),
        s.reshape(*input.shape[:-2], k),
        v.reshape(*input.shape[:-2], n, k),
    )


def _blocked_jacobi_singular_values(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    rows = max(m, n)
    a = input.contiguous().reshape(batch, m, n)
    a_work = torch.empty((batch, k, rows), dtype=torch.float32, device=input.device)
    s_work = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    ranks = torch.empty((batch, k), dtype=torch.int32, device=input.device)
    s = torch.empty((batch, k), dtype=input.dtype, device=input.device)

    block_r = triton.next_power_of_2(rows)
    sweeps = 14 if k > 256 else 10
    round_size = k if k % 2 == 0 else k + 1
    half_round = round_size // 2
    with torch_device_fn.device(input.device):
        _cyclic_jacobi_init_a_kernel[(batch, k)](
            a,
            a_work,
            M=m,
            N=n,
            K=k,
            ROWS=rows,
            TALL=m >= n,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        for _ in range(sweeps):
            for step in range(round_size - 1):
                _blocked_jacobi_pair_svals_kernel[(batch, half_round)](
                    a_work,
                    step,
                    K=k,
                    ROUND=round_size,
                    ROWS=rows,
                    BLOCK_R=block_r,
                    num_warps=1 if block_r <= 64 else 4,
                )
        _cyclic_jacobi_norm_kernel[(batch, k)](
            a_work,
            s_work,
            K=k,
            ROWS=rows,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
        _blocked_jacobi_rank_kernel[(batch, k)](
            s_work,
            ranks,
            s,
            k,
            num_warps=1,
        )

    return s.reshape(*input.shape[:-2], k)


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


@libentry()
@triton.jit
def _rank1_svd_kernel(
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
    offsets = tl.arange(0, BLOCK_R)
    eps = 1.1920928955078125e-7
    a_base = A + pid * M * N
    norm_sq = tl.full((), 0.0, dtype=tl.float32)

    if TALL:
        for base in range(0, M, BLOCK_R):
            rows = base + offsets
            mask = rows < M
            vals = tl.load(a_base + rows * N, mask=mask, other=0.0).to(tl.float32)
            norm_sq += tl.sum(vals * vals)

        norm = tl.sqrt(norm_sq)
        denom = tl.maximum(norm, eps)
        tl.store(S + pid, norm)
        tl.store(V + pid, 1.0)

        u_base = U + pid * M
        for base in range(0, M, BLOCK_R):
            rows = base + offsets
            mask = rows < M
            vals = tl.load(a_base + rows * N, mask=mask, other=0.0).to(tl.float32)
            tl.store(u_base + rows, vals / denom, mask=mask)
    else:
        for base in range(0, N, BLOCK_R):
            cols = base + offsets
            mask = cols < N
            vals = tl.load(a_base + cols, mask=mask, other=0.0).to(tl.float32)
            norm_sq += tl.sum(vals * vals)

        norm = tl.sqrt(norm_sq)
        denom = tl.maximum(norm, eps)
        tl.store(S + pid, norm)
        tl.store(U + pid, 1.0)

        v_base = V + pid * N
        for base in range(0, N, BLOCK_R):
            cols = base + offsets
            mask = cols < N
            vals = tl.load(a_base + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(v_base + cols, vals / denom, mask=mask)


def _rank1_svd(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    u = torch.empty((batch, m, 1), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, 1), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, 1), dtype=input.dtype, device=input.device)
    if batch != 0:
        rows = max(m, n)
        block_r = _RANK1_BLOCK_R_MAX
        if rows <= _RANK1_BLOCK_R_MAX:
            block_r = triton.next_power_of_2(rows)
        with torch_device_fn.device(input.device):
            _rank1_svd_kernel[(batch,)](
                a,
                u,
                s,
                v,
                m,
                n,
                TALL=n == 1,
                BLOCK_R=block_r,
                num_warps=1 if block_r <= 64 else 4,
            )
    return (
        u.reshape(*input.shape[:-2], m, 1),
        s.reshape(*input.shape[:-2], 1),
        v.reshape(*input.shape[:-2], n, 1),
    )


@libentry()
@triton.jit
def _complex_to_real_embedding_kernel(
    A_RI,
    R,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    total = 4 * M * N
    mask = offsets < total
    row = offsets // (2 * N)
    col = offsets - row * (2 * N)
    src_row = tl.where(row < M, row, row - M)
    src_col = tl.where(col < N, col, col - N)
    comp = tl.where((row < M) & (col >= N), 1, 0)
    comp = tl.where((row >= M) & (col < N), 1, comp)
    vals = tl.load(
        A_RI + batch * M * N * 2 + (src_row * N + src_col) * 2 + comp,
        mask=mask,
        other=0.0,
    )
    sign = tl.where((row < M) & (col >= N), -1.0, 1.0)
    tl.store(R + batch * 4 * M * N + offsets, vals * sign, mask=mask)


@libentry()
@triton.jit
def _complex_svd_pick_factor_kernel(
    REAL_FACTOR,
    OUT_RI,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    REAL_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < ROWS * K
    row = offsets // K
    col = offsets % K
    src_col = col * 2
    real = tl.load(
        REAL_FACTOR + batch * (2 * ROWS) * REAL_K + row * REAL_K + src_col,
        mask=mask,
        other=0.0,
    )
    imag = tl.load(
        REAL_FACTOR + batch * (2 * ROWS) * REAL_K + (ROWS + row) * REAL_K + src_col,
        mask=mask,
        other=0.0,
    )
    out_base = OUT_RI + batch * ROWS * K * 2 + offsets * 2
    tl.store(out_base, real, mask=mask)
    tl.store(out_base + 1, imag, mask=mask)


@libentry()
@triton.jit
def _complex_svd_pick_s_kernel(
    S_REAL,
    S,
    K: tl.constexpr,
    REAL_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    src = cols * 2
    vals_a = tl.load(S_REAL + batch * REAL_K + src, mask=mask, other=0.0)
    vals_b = tl.load(S_REAL + batch * REAL_K + src + 1, mask=mask, other=0.0)
    tl.store(S + batch * K + cols, 0.5 * (vals_a + vals_b), mask=mask)


@libentry()
@triton.jit
def _complex_svd_pick_orthonormal_v_kernel(
    V_REAL,
    V_RI,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    REAL_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    col_mask = cols < K
    src_cols = cols * 2
    base = V_REAL + batch * (2 * ROWS) * REAL_K
    vr = tl.load(
        base + rows[:, None] * REAL_K + src_cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    vi = tl.load(
        base + (ROWS + rows[:, None]) * REAL_K + src_cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )

    for c in tl.static_range(0, 16):
        cur_mask = c < K
        cur_r = tl.sum(tl.where(cols[None, :] == c, vr, 0.0), axis=1)
        cur_i = tl.sum(tl.where(cols[None, :] == c, vi, 0.0), axis=1)
        for p in tl.static_range(0, c):
            prev_r = tl.sum(tl.where(cols[None, :] == p, vr, 0.0), axis=1)
            prev_i = tl.sum(tl.where(cols[None, :] == p, vi, 0.0), axis=1)
            coeff_r = tl.sum(
                tl.where(row_mask, prev_r * cur_r + prev_i * cur_i, 0.0), axis=0
            )
            coeff_i = tl.sum(
                tl.where(row_mask, prev_r * cur_i - prev_i * cur_r, 0.0), axis=0
            )
            cur_r -= prev_r * coeff_r - prev_i * coeff_i
            cur_i -= prev_r * coeff_i + prev_i * coeff_r
        norm_sq = tl.sum(tl.where(row_mask, cur_r * cur_r + cur_i * cur_i, 0.0), axis=0)
        inv_norm = tl.rsqrt(tl.maximum(norm_sq, 1.0e-20))
        cur_r *= inv_norm
        cur_i *= inv_norm
        vr = tl.where((cols[None, :] == c) & cur_mask, cur_r[:, None], vr)
        vi = tl.where((cols[None, :] == c) & cur_mask, cur_i[:, None], vi)

    out_base = V_RI + batch * ROWS * K * 2
    offsets = rows[:, None] * K + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_base + offsets * 2, vr, mask=mask)
    tl.store(out_base + offsets * 2 + 1, vi, mask=mask)


@libentry()
@triton.jit
def _complex_svd_project_u_kernel(
    A_RI,
    V_RI,
    S,
    U_RI,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * K
    row = offsets // K
    col = offsets % K

    acc_r = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for j in tl.static_range(0, N):
        a_base = A_RI + batch * M * N * 2 + (row * N + j) * 2
        v_base = V_RI + batch * N * K * 2 + (j * K + col) * 2
        ar = tl.load(a_base, mask=mask, other=0.0)
        ai = tl.load(a_base + 1, mask=mask, other=0.0)
        vr = tl.load(v_base, mask=mask, other=0.0)
        vi = tl.load(v_base + 1, mask=mask, other=0.0)
        acc_r += ar * vr - ai * vi
        acc_i += ar * vi + ai * vr

    s = tl.load(S + batch * K + col, mask=mask, other=1.0)
    inv_s = tl.where(s > 1.0e-20, 1.0 / s, 0.0)
    out_base = U_RI + batch * M * K * 2 + offsets * 2
    tl.store(out_base, acc_r * inv_s, mask=mask)
    tl.store(out_base + 1, acc_i * inv_s, mask=mask)


def _complex_svd_via_real_embedding(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    a_ri = torch.view_as_real(input.contiguous()).reshape(batch, m, n, 2)
    real_matrix = torch.empty(
        (batch, 2 * m, 2 * n), dtype=torch.float32, device=input.device
    )
    block_size = triton.next_power_of_2(4 * m * n)
    with torch_device_fn.device(input.device):
        _complex_to_real_embedding_kernel[(batch,)](
            a_ri,
            real_matrix,
            M=m,
            N=n,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
    _, s_real, v_real = svd(real_matrix, some=True, compute_uv=True)
    s = torch.empty((batch, k), dtype=torch.float32, device=input.device)
    u = torch.empty((*input.shape[:-2], m, k), dtype=input.dtype, device=input.device)
    v = torch.empty((*input.shape[:-2], n, k), dtype=input.dtype, device=input.device)
    u_ri = torch.view_as_real(u).reshape(batch, m, k, 2)
    v_ri = torch.view_as_real(v).reshape(batch, n, k, 2)
    with torch_device_fn.device(input.device):
        _complex_svd_pick_s_kernel[(batch,)](
            s_real,
            s,
            K=k,
            REAL_K=2 * k,
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=1,
        )
        _complex_svd_pick_orthonormal_v_kernel[(batch,)](
            v_real,
            v_ri,
            ROWS=n,
            K=k,
            REAL_K=2 * k,
            BLOCK_ROWS=triton.next_power_of_2(n),
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=1,
        )
        _complex_svd_project_u_kernel[(batch,)](
            a_ri,
            v_ri,
            s,
            u_ri,
            M=m,
            N=n,
            K=k,
            BLOCK_SIZE=triton.next_power_of_2(m * k),
            num_warps=1,
        )
    return (
        u,
        s.reshape(*input.shape[:-2], k),
        v,
    )


def _gram_svd(input):
    return _unsupported_svd(input, True, True)


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
    return _unsupported_svd(input, True, True)


def _large_native_svd(input):
    if _can_use_hier_block_square_project_kernel(input, True, True):
        return _hier_block_jacobi_square_project_svd(input)
    if _can_use_blocked_square_project_kernel(input, True, True):
        return _blocked_jacobi_square_project_svd(input)
    if _can_use_blocked_jacobi_kernel(input, True, True):
        return _blocked_jacobi_svd(input)
    return _bidiagonal_qr_dqds_svd(input)


def _bidiagonal_qr_dqds_svd(input):
    return _unsupported_svd(
        input,
        True,
        True,
        "The k > 512 blocked-bidiagonalization plus QR/DQDS path is reserved "
        "for the next native large-matrix solver stage.",
    )


def _empty_svd_result(input, some=True, compute_uv=True):
    _, m, n = _svd_shape(input)
    k = min(m, n)
    u_cols = k if compute_uv and some else m
    v_cols = k if compute_uv and some else n
    u = torch.empty(
        (*input.shape[:-2], m, u_cols), dtype=input.dtype, device=input.device
    )
    s = torch.empty((*input.shape[:-2], k), dtype=input.dtype, device=input.device)
    v = torch.empty(
        (*input.shape[:-2], n, v_cols), dtype=input.dtype, device=input.device
    )
    return u, s, v


@libentry()
@triton.jit
def _complete_svd_factor_kernel(
    THIN,
    FULL,
    ROWS: tl.constexpr,
    THIN_COLS: tl.constexpr,
    FULL_COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    row_mask = rows < ROWS
    col_mask = cols < FULL_COLS
    vals = tl.load(
        THIN + batch * ROWS * THIN_COLS + rows[:, None] * THIN_COLS + cols[None, :],
        mask=row_mask[:, None] & (cols[None, :] < THIN_COLS),
        other=0.0,
    )
    identity = tl.where(rows[:, None] == cols[None, :], 1.0, 0.0)
    vals = tl.where(cols[None, :] < THIN_COLS, vals, identity)

    for c in tl.static_range(0, 64):
        cur_mask = c < FULL_COLS
        cur = tl.sum(tl.where(cols[None, :] == c, vals, 0.0), axis=1)
        for p in tl.static_range(0, c):
            prev = tl.sum(tl.where(cols[None, :] == p, vals, 0.0), axis=1)
            coeff = tl.sum(tl.where(row_mask, prev * cur, 0.0), axis=0)
            cur -= prev * coeff
        norm_sq = tl.sum(tl.where(row_mask, cur * cur, 0.0), axis=0)
        inv_norm = tl.rsqrt(tl.maximum(norm_sq, 1.0e-20))
        cur *= inv_norm
        vals = tl.where((cols[None, :] == c) & cur_mask, cur[:, None], vals)

    out_base = FULL + batch * ROWS * FULL_COLS
    offsets = rows[:, None] * FULL_COLS + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_base + offsets, vals, mask=mask)


def _low_precision_svd_via_float32(input, some=True, compute_uv=True):
    u, s, v = svd(input.to(torch.float32), some=some, compute_uv=compute_uv)
    return u.to(input.dtype), s.to(input.dtype), v.to(input.dtype)


def _some_false_svd_via_thin(input):
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    thin_u, s, thin_v = svd(input, some=True, compute_uv=True)
    u = torch.empty((*input.shape[:-2], m, m), dtype=input.dtype, device=input.device)
    v = torch.empty((*input.shape[:-2], n, n), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device):
        _complete_svd_factor_kernel[(batch,)](
            thin_u,
            u,
            ROWS=m,
            THIN_COLS=k,
            FULL_COLS=m,
            BLOCK_ROWS=triton.next_power_of_2(m),
            BLOCK_COLS=triton.next_power_of_2(m),
            num_warps=4,
        )
        _complete_svd_factor_kernel[(batch,)](
            thin_v,
            v,
            ROWS=n,
            THIN_COLS=k,
            FULL_COLS=n,
            BLOCK_ROWS=triton.next_power_of_2(n),
            BLOCK_COLS=triton.next_power_of_2(n),
            num_warps=4,
        )
    return u, s, v


def _compute_uv_false_result(input, s):
    _, m, n = _svd_shape(input)
    u = torch.empty((*input.shape[:-2], m, m), dtype=input.dtype, device=input.device)
    v = torch.empty((*input.shape[:-2], n, n), dtype=input.dtype, device=input.device)
    return u, s, v


def _singular_values_only(input):
    _, m, n = _svd_shape(input)
    k = min(m, n)
    largest = max(m, n)
    if k == 2 and largest <= _RANK2_BLOCK_R_MAX:
        return _rank2_singular_values(input)
    if k <= 16 and largest <= 1024:
        return _small_jacobi_singular_values(input)
    if 16 < k <= 512 and largest <= 1024:
        return _blocked_jacobi_singular_values(input)
    return _unsupported_svd(input, True, False)


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
    if (
        input.is_cuda
        and input.dtype == torch.complex64
        and some
        and compute_uv
        and input.dim() >= 2
        and 0 not in input.shape[-2:]
        and max(input.shape[-2:]) <= 16
    ):
        return SVDResult(*_complex_svd_via_real_embedding(input))
    if _is_low_precision_cuda_matrix(input):
        return SVDResult(*_low_precision_svd_via_float32(input, some, compute_uv))
    if _is_float32_cuda_matrix(input) and 0 in input.shape[-2:]:
        return SVDResult(*_empty_svd_result(input, some, compute_uv))
    if _can_use_singular_values_only(input, some, compute_uv):
        return SVDResult(*_compute_uv_false_result(input, _singular_values_only(input)))
    if (
        _is_float32_cuda_matrix(input)
        and not some
        and compute_uv
        and max(input.shape[-2:]) <= 64
    ):
        return SVDResult(*_some_false_svd_via_thin(input))
    if not _is_float32_cuda_matrix(input) or not some:
        return SVDResult(*_unsupported_svd(input, some, compute_uv))
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    try:
        if k == 1:
            return SVDResult(*_rank1_svd(input))
        if k == 2 and max(m, n) <= _RANK2_BLOCK_R_MAX:
            return SVDResult(*_rank2_svd(input))
        if k == 4 and m == 4 and n == 4 and batch >= 16:
            return SVDResult(*_small4_square_svd(input))
        if _can_use_tall_wide_gram_jacobi_kernel(input, some, compute_uv):
            return SVDResult(*_gram_jacobi_svd(input))
        use_batched_cyclic16 = k == 16 and batch >= 8 and max(m, n) <= 64
        if (
            _can_use_small_jacobi_kernel(input, some, compute_uv)
            and not use_batched_cyclic16
        ):
            return SVDResult(*_small_jacobi_svd(input))
        if _can_use_tsqr_cholesky_kernel(input, some, compute_uv):
            return SVDResult(*_tsqr_cholesky_svd(input))
        if _can_use_projected_jacobi_kernel(input, some, compute_uv):
            return SVDResult(*_projected_jacobi_svd(input))
        if _can_use_cyclic_jacobi_kernel(input, some, compute_uv):
            return SVDResult(*_cyclic_jacobi_svd(input))
        if _can_use_hier_block_square_project_kernel(input, some, compute_uv):
            return SVDResult(*_hier_block_jacobi_square_project_svd(input))
        if _can_use_blocked_square_project_kernel(input, some, compute_uv):
            return SVDResult(*_blocked_jacobi_square_project_svd(input))
        if _can_use_blocked_jacobi_kernel(input, some, compute_uv):
            return SVDResult(*_blocked_jacobi_svd(input))
        if _should_use_gram16(batch, m, n):
            return SVDResult(*_gram16_svd(input))
        if _should_use_gram(batch, m, n):
            return SVDResult(*_gram_svd(input))
        return SVDResult(*_large_native_svd(input))
    except RuntimeError:
        return SVDResult(*_unsupported_svd(input, some, compute_uv))
