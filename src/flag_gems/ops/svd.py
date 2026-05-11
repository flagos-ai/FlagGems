import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)

# FP32 only. For fp16/bf16, upcast to fp32 at entry and cast back.
_SUPPORTED_SVD_DTYPES = (torch.float32,)

_STEP_TENSOR_CACHE = {}


# =============================================================================
# Helpers
# =============================================================================

def _is_supported_input(x):
    return x.is_cuda and x.dtype in _SUPPORTED_SVD_DTYPES and x.ndim >= 2


def _svd_dims(x):
    m, n = x.shape[-2], x.shape[-1]
    b = 1
    for d in x.shape[:-2]:
        b *= d
    return b, m, n


def _next_power_of_2(x: int):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _brent_luk_pairs(K):
    if K <= 1:
        return []

    steps = []
    n_eff = K if K % 2 == 0 else K + 1

    for s in range(n_eff - 1):
        i_l, j_l = [], []
        for k in range(n_eff // 2):
            i = (s + k) % (n_eff - 1)
            j = n_eff - 1 if k == 0 else (s + n_eff - 1 - k) % (n_eff - 1)
            if i < K and j < K:
                i_l.append(i)
                j_l.append(j)

        if i_l:
            steps.append((i_l, j_l))

    return steps


def _cache_key_for_steps(device, K):
    dev_index = device.index if device.index is not None else torch.cuda.current_device()
    return dev_index, K


def _get_step_tensors(K, device):
    key = _cache_key_for_steps(device, K)
    cached = _STEP_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    steps = _brent_luk_pairs(K)
    step_tensors = [
        (
            torch.tensor(i, device=device, dtype=torch.int32),
            torch.tensor(j, device=device, dtype=torch.int32),
            len(i),
        )
        for i, j in steps
    ]
    _STEP_TENSOR_CACHE[key] = step_tensors
    return step_tensors


# =============================================================================
# zero fill
# =============================================================================

@libentry()
@triton.jit
def _zero_fill_kernel(X, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    tl.store(X + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


def _empty_zero_tensor(shape, device, dtype):
    out = torch.empty(shape, device=device, dtype=dtype)
    total = out.numel()
    if total > 0:
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        _zero_fill_kernel[grid](out, TOTAL=total, BLOCK=BLOCK, num_warps=4)
    return out


# =============================================================================
# Rank-1 kernels
# =============================================================================

@libentry()
@triton.jit
def svd_mx1_kernel(
    x,
    u,
    s,
    v,
    batch,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    mask = rows < M

    vals = tl.load(x + pid * M + rows, mask=mask, other=0.0).to(tl.float32)
    norm2 = tl.sum(vals * vals, axis=0)
    norm = tl.sqrt(tl.maximum(norm2, 0.0))
    inv = 1.0 / tl.where(norm > 1.0e-20, norm, 1.0)
    u_vals = tl.where(norm > 1.0e-20, vals * inv, rows == 0)

    tl.store(s + pid, norm)
    tl.store(u + pid * M + rows, u_vals, mask=mask)
    tl.store(v + pid, 1.0)


@libentry()
@triton.jit
def svd_1xn_kernel(
    x,
    u,
    s,
    v,
    batch,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    vals = tl.load(x + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    norm2 = tl.sum(vals * vals, axis=0)
    norm = tl.sqrt(tl.maximum(norm2, 0.0))
    inv = 1.0 / tl.where(norm > 1.0e-20, norm, 1.0)
    v_vals = tl.where(norm > 1.0e-20, vals * inv, cols == 0)

    tl.store(s + pid, norm)
    tl.store(u + pid, 1.0)
    tl.store(v + pid * N + cols, v_vals, mask=mask)


def _svd_rank1(A):
    b, m, n = _svd_dims(A)
    device = A.device
    dtype = A.dtype

    U = torch.empty((b, m, 1), device=device, dtype=dtype)
    S = torch.empty((b, 1), device=device, dtype=dtype)
    V = torch.empty((b, n, 1), device=device, dtype=dtype)

    if n == 1:
        block_m = _next_power_of_2(m)
        block_m = min(max(block_m, 16), 4096)
        svd_mx1_kernel[(b,)](
            A,
            U,
            S,
            V,
            b,
            M=m,
            BLOCK_M=block_m,
            num_warps=4 if block_m <= 256 else 8,
        )
    else:
        block_n = _next_power_of_2(n)
        block_n = min(max(block_n, 16), 4096)
        svd_1xn_kernel[(b,)](
            A,
            U,
            S,
            V,
            b,
            N=n,
            BLOCK_N=block_n,
            num_warps=4 if block_n <= 256 else 8,
        )

    return U, S, V


# =============================================================================
# 2x2 closed-form kernel
# =============================================================================

@libentry()
@triton.jit
def svd_2x2_kernel(
    x,
    u,
    s,
    v,
    batch,
    compute_uv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch

    base = offsets * 4

    a = tl.load(x + base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x + base + 1, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(x + base + 2, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(x + base + 3, mask=mask, other=0.0).to(tl.float32)

    ata00 = a * a + c * c
    ata01 = a * b + c * d
    ata11 = b * b + d * d

    half_diff = 0.5 * (ata00 - ata11)
    half_trace = 0.5 * (ata00 + ata11)
    radius = tl.sqrt(half_diff * half_diff + ata01 * ata01)

    lam0 = tl.maximum(half_trace + radius, 0.0)
    lam1 = tl.maximum(half_trace - radius, 0.0)

    s0 = tl.sqrt(lam0)
    s1 = tl.sqrt(lam1)

    s_base = offsets * 2
    tl.store(s + s_base, s0, mask=mask)
    tl.store(s + s_base + 1, s1, mask=mask)

    if compute_uv:
        use_first = ata00 >= ata11
        raw_v00 = tl.where(use_first, lam0 - ata11, ata01)
        raw_v10 = tl.where(use_first, ata01, lam0 - ata00)

        raw_norm = tl.sqrt(raw_v00 * raw_v00 + raw_v10 * raw_v10)
        inv_raw = 1.0 / tl.where(raw_norm > 0.0, raw_norm, 1.0)

        v00 = tl.where(raw_norm > 0.0, raw_v00 * inv_raw, 1.0)
        v10 = tl.where(raw_norm > 0.0, raw_v10 * inv_raw, 0.0)
        v01 = -v10
        v11 = v00

        eps = 1.0e-20
        inv_s0 = 1.0 / tl.where(s0 > eps, s0, 1.0)

        av0_0 = a * v00 + b * v10
        av0_1 = c * v00 + d * v10

        u00 = tl.where(s0 > eps, av0_0 * inv_s0, 1.0)
        u10 = tl.where(s0 > eps, av0_1 * inv_s0, 0.0)

        av1_0 = a * v01 + b * v11
        av1_1 = c * v01 + d * v11

        perp_u01 = -u10
        perp_u11 = u00

        sign = tl.where(perp_u01 * av1_0 + perp_u11 * av1_1 >= 0.0, 1.0, -1.0)
        use_direct = s1 > s0 * 2.0e-1
        inv_s1 = 1.0 / tl.where(use_direct, s1, 1.0)

        u01 = tl.where(use_direct, av1_0 * inv_s1, sign * perp_u01)
        u11 = tl.where(use_direct, av1_1 * inv_s1, sign * perp_u11)
    else:
        u00 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        u01 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        u10 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        u11 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        v00 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        v01 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        v10 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        v11 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    tl.store(u + base, u00, mask=mask)
    tl.store(u + base + 1, u01, mask=mask)
    tl.store(u + base + 2, u10, mask=mask)
    tl.store(u + base + 3, u11, mask=mask)

    tl.store(v + base, v00, mask=mask)
    tl.store(v + base + 1, v01, mask=mask)
    tl.store(v + base + 2, v10, mask=mask)
    tl.store(v + base + 3, v11, mask=mask)


def _svd_2x2(A, compute_uv=True):
    b, m, n = _svd_dims(A)
    device = A.device
    dtype = A.dtype

    U = torch.empty((b, 2, 2), device=device, dtype=dtype)
    S = torch.empty((b, 2), device=device, dtype=dtype)
    V = torch.empty((b, 2, 2), device=device, dtype=dtype)

    block = 256
    grid = (triton.cdiv(b, block),)

    svd_2x2_kernel[grid](
        A,
        U,
        S,
        V,
        b,
        compute_uv,
        BLOCK_SIZE=block,
        num_warps=4,
    )

    return U, S, V


# =============================================================================
# Rank-2 closed-form kernels
# =============================================================================

@libentry()
@triton.jit
def svd_mx2_kernel(
    x,
    u,
    s,
    v,
    batch,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    mask = rows < M

    base = pid * M * 2

    x0 = tl.load(x + base + rows * 2, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x + base + rows * 2 + 1, mask=mask, other=0.0).to(tl.float32)

    ata00 = tl.sum(x0 * x0, axis=0)
    ata01 = tl.sum(x0 * x1, axis=0)
    ata11 = tl.sum(x1 * x1, axis=0)

    half_diff = 0.5 * (ata00 - ata11)
    half_trace = 0.5 * (ata00 + ata11)
    radius = tl.sqrt(half_diff * half_diff + ata01 * ata01)

    lam0 = tl.maximum(half_trace + radius, 0.0)
    lam1 = tl.maximum(half_trace - radius, 0.0)

    s0 = tl.sqrt(lam0)
    s1 = tl.sqrt(lam1)

    use_first = ata00 >= ata11
    raw_v00 = tl.where(use_first, lam0 - ata11, ata01)
    raw_v10 = tl.where(use_first, ata01, lam0 - ata00)

    raw_norm = tl.sqrt(raw_v00 * raw_v00 + raw_v10 * raw_v10)
    inv_raw = 1.0 / tl.where(raw_norm > 0.0, raw_norm, 1.0)

    v00 = tl.where(raw_norm > 0.0, raw_v00 * inv_raw, 1.0)
    v10 = tl.where(raw_norm > 0.0, raw_v10 * inv_raw, 0.0)
    v01 = -v10
    v11 = v00

    eps = 1.0e-20
    inv_s0 = 1.0 / tl.where(s0 > eps, s0, 1.0)
    inv_s1 = 1.0 / tl.where(s1 > eps, s1, 1.0)

    u0 = (x0 * v00 + x1 * v10) * inv_s0
    u1 = (x0 * v01 + x1 * v11) * inv_s1

    s_base = pid * 2
    tl.store(s + s_base, s0)
    tl.store(s + s_base + 1, s1)

    u_base = pid * M * 2
    tl.store(u + u_base + rows * 2, u0, mask=mask)
    tl.store(u + u_base + rows * 2 + 1, u1, mask=mask)

    v_base = pid * 4
    tl.store(v + v_base, v00)
    tl.store(v + v_base + 1, v01)
    tl.store(v + v_base + 2, v10)
    tl.store(v + v_base + 3, v11)


@libentry()
@triton.jit
def svd_2xn_kernel(
    x,
    u,
    s,
    v,
    batch,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    base = pid * 2 * N

    x0 = tl.load(x + base + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x + base + N + cols, mask=mask, other=0.0).to(tl.float32)

    aat00 = tl.sum(x0 * x0, axis=0)
    aat01 = tl.sum(x0 * x1, axis=0)
    aat11 = tl.sum(x1 * x1, axis=0)

    half_diff = 0.5 * (aat00 - aat11)
    half_trace = 0.5 * (aat00 + aat11)
    radius = tl.sqrt(half_diff * half_diff + aat01 * aat01)

    lam0 = tl.maximum(half_trace + radius, 0.0)
    lam1 = tl.maximum(half_trace - radius, 0.0)

    s0 = tl.sqrt(lam0)
    s1 = tl.sqrt(lam1)

    use_first = aat00 >= aat11
    raw_u00 = tl.where(use_first, lam0 - aat11, aat01)
    raw_u10 = tl.where(use_first, aat01, lam0 - aat00)

    raw_norm = tl.sqrt(raw_u00 * raw_u00 + raw_u10 * raw_u10)
    inv_raw = 1.0 / tl.where(raw_norm > 0.0, raw_norm, 1.0)

    u00 = tl.where(raw_norm > 0.0, raw_u00 * inv_raw, 1.0)
    u10 = tl.where(raw_norm > 0.0, raw_u10 * inv_raw, 0.0)
    u01 = -u10
    u11 = u00

    eps = 1.0e-20
    inv_s0 = 1.0 / tl.where(s0 > eps, s0, 1.0)
    inv_s1 = 1.0 / tl.where(s1 > eps, s1, 1.0)

    v0 = (x0 * u00 + x1 * u10) * inv_s0
    v1 = (x0 * u01 + x1 * u11) * inv_s1

    s_base = pid * 2
    tl.store(s + s_base, s0)
    tl.store(s + s_base + 1, s1)

    u_base = pid * 4
    tl.store(u + u_base, u00)
    tl.store(u + u_base + 1, u01)
    tl.store(u + u_base + 2, u10)
    tl.store(u + u_base + 3, u11)

    v_base = pid * N * 2
    tl.store(v + v_base + cols * 2, v0, mask=mask)
    tl.store(v + v_base + cols * 2 + 1, v1, mask=mask)


def _svd_rank2(A):
    b, m, n = _svd_dims(A)
    device = A.device
    dtype = A.dtype

    U = torch.empty((b, m, 2), device=device, dtype=dtype)
    S = torch.empty((b, 2), device=device, dtype=dtype)
    V = torch.empty((b, n, 2), device=device, dtype=dtype)

    if n == 2:
        block_m = _next_power_of_2(m)
        block_m = min(max(block_m, 16), 4096)
        svd_mx2_kernel[(b,)](
            A,
            U,
            S,
            V,
            b,
            M=m,
            BLOCK_M=block_m,
            num_warps=4 if block_m <= 256 else 8,
        )
    else:
        block_n = _next_power_of_2(n)
        block_n = min(max(block_n, 16), 4096)
        svd_2xn_kernel[(b,)](
            A,
            U,
            S,
            V,
            b,
            N=n,
            BLOCK_N=block_n,
            num_warps=4 if block_n <= 256 else 8,
        )

    return U, S, V


# =============================================================================
# In-register one-sided Jacobi for small K
# =============================================================================

@libentry()
@triton.jit
def svd_small_jacobi_kernel(
    x,
    u,
    s,
    v,
    batch,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SWEEPS: tl.constexpr,
):
    pid = tle.program_id(0)

    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N

    base = pid * M * N

    a = tl.load(
        x + base + rows[:, None] * N + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    v_rows = tl.arange(0, BLOCK_N)
    v_cols = tl.arange(0, BLOCK_N)

    v_work = tl.where(
        v_rows[:, None] == v_cols[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, dtype=tl.float32),
        tl.full((BLOCK_N, BLOCK_N), 0.0, dtype=tl.float32),
    )

    for _ in range(NUM_SWEEPS):
        for p in range(N):
            for q in range(p + 1, N):
                a_p = tl.sum(
                    tl.where(
                        cols[None, :] == p,
                        a,
                        tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
                    ),
                    axis=1,
                )

                a_q = tl.sum(
                    tl.where(
                        cols[None, :] == q,
                        a,
                        tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
                    ),
                    axis=1,
                )

                alpha = tl.sum(tl.where(row_mask, a_p * a_p, 0.0), axis=0)
                beta = tl.sum(tl.where(row_mask, a_q * a_q, 0.0), axis=0)
                gamma = tl.sum(tl.where(row_mask, a_p * a_q, 0.0), axis=0)

                threshold = 1.0e-7 * tl.sqrt(alpha * beta + 1.0e-30)
                should_rotate = tl.abs(gamma) >= threshold

                safe_gamma = tl.where(should_rotate, gamma, 1.0)
                zeta = (beta - alpha) / (2.0 * safe_gamma)

                sign_zeta = tl.where(zeta >= 0.0, 1.0, -1.0)
                t = sign_zeta / (tl.abs(zeta) + tl.sqrt(1.0 + zeta * zeta))

                c = 1.0 / tl.sqrt(1.0 + t * t)
                sn = t * c

                c = tl.where(should_rotate, c, 1.0)
                sn = tl.where(should_rotate, sn, 0.0)

                new_a_p = c * a_p - sn * a_q
                new_a_q = sn * a_p + c * a_q

                a = tl.where(cols[None, :] == p, new_a_p[:, None], a)
                a = tl.where(cols[None, :] == q, new_a_q[:, None], a)

                v_p = tl.sum(
                    tl.where(
                        v_cols[None, :] == p,
                        v_work,
                        tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32),
                    ),
                    axis=1,
                )

                v_q = tl.sum(
                    tl.where(
                        v_cols[None, :] == q,
                        v_work,
                        tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32),
                    ),
                    axis=1,
                )

                new_v_p = c * v_p - sn * v_q
                new_v_q = sn * v_p + c * v_q

                v_work = tl.where(v_cols[None, :] == p, new_v_p[:, None], v_work)
                v_work = tl.where(v_cols[None, :] == q, new_v_q[:, None], v_work)

    s_vals = tl.sqrt(tl.sum(a * a, axis=0))
    s_vals = tl.where(col_mask, s_vals, 0.0)

    ranks = tl.sum(
        (
            (s_vals[:, None] > s_vals[None, :])
            | ((s_vals[:, None] == s_vals[None, :]) & (cols[:, None] < cols[None, :]))
        ).to(tl.int32),
        axis=0,
    )

    tl.store(s + pid * N + ranks, s_vals, mask=col_mask)

    for j in range(N):
        rank_j = tl.sum(tl.where(cols == j, ranks, tl.zeros((BLOCK_N,), tl.int32)))
        s_j = tl.sum(tl.where(cols == j, s_vals, tl.zeros((BLOCK_N,), tl.float32)))

        a_j = tl.sum(
            tl.where(
                cols[None, :] == j,
                a,
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
            ),
            axis=1,
        )

        u_j = a_j / tl.where(s_j > 1.0e-20, s_j, 1.0)

        tl.store(
            u + pid * M * N + rows * N + rank_j,
            u_j,
            mask=row_mask,
        )

        v_j = tl.sum(
            tl.where(
                v_cols[None, :] == j,
                v_work,
                tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32),
            ),
            axis=1,
        )

        tl.store(
            v + pid * N * N + v_rows * N + rank_j,
            v_j,
            mask=v_rows < N,
        )


def _can_use_small_jacobi(A):
    b, m, n = _svd_dims(A)
    k = min(m, n)
    max_dim = max(m, n)

    return (
        (3 <= k <= 8 and max_dim <= 1024)
        or (k == 16 and max_dim <= 512)
        or (k == 20 and max_dim <= 128)
        or (k == 32 and max_dim <= 256)
    )


def _svd_small_jacobi(A):
    b, m0, n0 = _svd_dims(A)

    transpose = m0 < n0
    inp = A.transpose(-2, -1).contiguous() if transpose else A

    b, m, n = _svd_dims(inp)
    device = inp.device
    dtype = inp.dtype

    U_tmp = torch.empty((b, m, n), device=device, dtype=dtype)
    S = torch.empty((b, n), device=device, dtype=dtype)
    V_tmp = torch.empty((b, n, n), device=device, dtype=dtype)

    block_m = _next_power_of_2(m)
    block_n = _next_power_of_2(n)

    block_m = min(max(block_m, 16), 1024)
    block_n = min(max(block_n, 16), 32)

    if n <= 4:
        num_sweeps = 40
    elif n == 20:
        num_sweeps = 6
    elif n == 32:
        num_sweeps = 6
    else:
        num_sweeps = 10

    svd_small_jacobi_kernel[(b,)](
        inp,
        U_tmp,
        S,
        V_tmp,
        b,
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_SWEEPS=num_sweeps,
        num_warps=4 if block_m <= 256 else 8,
    )

    if transpose:
        return V_tmp, S, U_tmp

    return U_tmp, S, V_tmp


# =============================================================================
# Streaming Jacobi for batched K=64/128
# =============================================================================

@libentry()
@triton.jit
def svd_streaming_jacobi_kernel(
    x,
    a_work,
    v_work,
    u,
    s,
    v,
    batch,
    aw_batch_stride,
    aw_col_stride,
    vw_batch_stride,
    vw_col_stride,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SWEEPS: tl.constexpr,
):
    pid = tle.program_id(0)

    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N

    aw_base = a_work + pid * aw_batch_stride
    vw_base = v_work + pid * vw_batch_stride

    for j in range(N):
        x_col = tl.load(
            x + pid * M * N + rows * N + j,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(aw_base + j * aw_col_stride + rows, x_col, mask=row_mask)

        v_col = tl.where(cols == j, 1.0, 0.0)
        tl.store(vw_base + j * vw_col_stride + cols, v_col, mask=col_mask)

    for _ in range(NUM_SWEEPS):
        for p in range(N):
            for q in range(p + 1, N):
                a_p = tl.load(
                    aw_base + p * aw_col_stride + rows,
                    mask=row_mask,
                    other=0.0,
                )

                a_q = tl.load(
                    aw_base + q * aw_col_stride + rows,
                    mask=row_mask,
                    other=0.0,
                )

                alpha = tl.sum(a_p * a_p, axis=0)
                beta = tl.sum(a_q * a_q, axis=0)
                gamma = tl.sum(a_p * a_q, axis=0)

                threshold = 1.0e-7 * tl.sqrt(alpha * beta + 1.0e-30)
                should_rotate = tl.abs(gamma) >= threshold

                safe_gamma = tl.where(should_rotate, gamma, 1.0)
                tau = (beta - alpha) / (2.0 * safe_gamma)

                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))

                c = 1.0 / tl.sqrt(1.0 + t * t)
                sn = t * c

                c = tl.where(should_rotate, c, 1.0)
                sn = tl.where(should_rotate, sn, 0.0)

                tl.store(
                    aw_base + p * aw_col_stride + rows,
                    c * a_p - sn * a_q,
                    mask=row_mask,
                )

                tl.store(
                    aw_base + q * aw_col_stride + rows,
                    sn * a_p + c * a_q,
                    mask=row_mask,
                )

                v_p = tl.load(
                    vw_base + p * vw_col_stride + cols,
                    mask=col_mask,
                    other=0.0,
                )

                v_q = tl.load(
                    vw_base + q * vw_col_stride + cols,
                    mask=col_mask,
                    other=0.0,
                )

                tl.store(
                    vw_base + p * vw_col_stride + cols,
                    c * v_p - sn * v_q,
                    mask=col_mask,
                )

                tl.store(
                    vw_base + q * vw_col_stride + cols,
                    sn * v_p + c * v_q,
                    mask=col_mask,
                )

    s_vals = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for j in range(N):
        a_j = tl.load(
            aw_base + j * aw_col_stride + rows,
            mask=row_mask,
            other=0.0,
        )

        norm_j = tl.sqrt(tl.sum(a_j * a_j, axis=0))
        s_vals = tl.where(cols == j, norm_j, s_vals)

    ranks = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for j in range(N):
        s_j = tl.sum(tl.where(cols == j, s_vals, tl.zeros((BLOCK_N,), tl.float32)))
        j_vec = tl.full((BLOCK_N,), j, dtype=tl.int32)

        ranks += (
            ((s_j > s_vals) | ((s_j == s_vals) & (j_vec < cols))) & col_mask
        ).to(tl.int32)

    tl.store(s + pid * N + ranks, s_vals, mask=col_mask)

    for j in range(N):
        rank_j = tl.sum(tl.where(cols == j, ranks, tl.zeros((BLOCK_N,), tl.int32)))
        s_j = tl.sum(tl.where(cols == j, s_vals, tl.zeros((BLOCK_N,), tl.float32)))

        a_j = tl.load(
            aw_base + j * aw_col_stride + rows,
            mask=row_mask,
            other=0.0,
        )

        u_j = a_j / tl.where(s_j > 1.0e-20, s_j, 1.0)

        tl.store(
            u + pid * M * N + rows * N + rank_j,
            u_j,
            mask=row_mask,
        )

        v_j = tl.load(
            vw_base + j * vw_col_stride + cols,
            mask=col_mask,
            other=0.0,
        )

        tl.store(
            v + pid * N * N + cols * N + rank_j,
            v_j,
            mask=col_mask,
        )


def _can_use_streaming_jacobi(A):
    b, m, n = _svd_dims(A)
    k = min(m, n)
    max_dim = max(m, n)

    return (
        (k == 64 and max_dim <= 1024 and b >= 16)
        or (k == 128 and max_dim <= 128 and b >= 16)
    )


def _svd_streaming_jacobi(A):
    b0, m0, n0 = _svd_dims(A)

    transpose = m0 < n0
    inp = A.transpose(-2, -1).contiguous() if transpose else A

    b, m, n = _svd_dims(inp)
    device = inp.device
    dtype = inp.dtype

    U_tmp = torch.empty((b, m, n), device=device, dtype=dtype)
    S = torch.empty((b, n), device=device, dtype=dtype)
    V_tmp = torch.empty((b, n, n), device=device, dtype=dtype)

    a_work = torch.empty((b, n, m), device=device, dtype=torch.float32)
    v_work = torch.empty((b, n, n), device=device, dtype=torch.float32)

    block_m = _next_power_of_2(m)
    block_n = _next_power_of_2(n)

    block_m = min(max(block_m, 16), 1024)
    block_n = min(max(block_n, 16), 128)

    num_sweeps = 10 if n == 128 else 8

    svd_streaming_jacobi_kernel[(b,)](
        inp,
        a_work,
        v_work,
        U_tmp,
        S,
        V_tmp,
        b,
        a_work.stride(0),
        a_work.stride(1),
        v_work.stride(0),
        v_work.stride(1),
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_SWEEPS=num_sweeps,
        num_warps=8,
    )

    if transpose:
        return V_tmp, S, U_tmp

    return U_tmp, S, V_tmp


# =============================================================================
# Pure Triton Gram + Jacobi eig fallback
# =============================================================================

@libentry()
@triton.jit
def _gram_sym_kernel(
    A,
    G,
    ORIG_M: tl.constexpr,
    ORIG_N: tl.constexpr,
    M_BIG: tl.constexpr,
    K: tl.constexpr,
    TALL: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
):
    pb = tle.program_id(0)
    pi = tle.program_id(1)
    pj = tle.program_id(2)

    if pj < pi:
        return

    oi = pi * BN + tl.arange(0, BN)
    oj = pj * BN + tl.arange(0, BN)
    om = tl.arange(0, BM)

    mi = oi < K
    mj = oj < K

    acc = tl.zeros((BN, BN), dtype=tl.float32)

    base_a = A + pb * ORIG_M * ORIG_N
    base_g = G + pb * K * K

    for m0 in range(0, M_BIG, BM):
        rows = m0 + om
        mr = rows < M_BIG

        if TALL:
            xi = tl.load(
                base_a + rows[:, None] * ORIG_N + oi[None, :],
                mask=mr[:, None] & mi[None, :],
                other=0.0,
            ).to(tl.float32)

            xj = tl.load(
                base_a + rows[:, None] * ORIG_N + oj[None, :],
                mask=mr[:, None] & mj[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            xi = tl.load(
                base_a + oi[None, :] * ORIG_N + rows[:, None],
                mask=mi[None, :] & mr[:, None],
                other=0.0,
            ).to(tl.float32)

            xj = tl.load(
                base_a + oj[None, :] * ORIG_N + rows[:, None],
                mask=mj[None, :] & mr[:, None],
                other=0.0,
            ).to(tl.float32)

        acc += tl.dot(tl.trans(xi), xj, input_precision="ieee")

    tl.store(
        base_g + oi[:, None] * K + oj[None, :],
        acc,
        mask=mi[:, None] & mj[None, :],
    )

    if pj != pi:
        tl.store(
            base_g + oj[:, None] * K + oi[None, :],
            tl.trans(acc),
            mask=mj[:, None] & mi[None, :],
        )


def _compute_gram(A, b, m, n):
    device = A.device
    K = min(m, n)
    M_big = max(m, n)
    tall = m >= n

    G = torch.empty((b, K, K), device=device, dtype=torch.float32)

    BN = 32
    BM = 64

    if b <= 4 and K >= 128 and M_big >= 512:
        BM = 128
    else:
        BM = 64
        
    grid = (b, triton.cdiv(K, BN), triton.cdiv(K, BN))

    _gram_sym_kernel[grid](
        A,
        G,
        ORIG_M=m,
        ORIG_N=n,
        M_BIG=M_big,
        K=K,
        TALL=tall,
        BN=BN,
        BM=BM,
        num_warps=4,
        num_stages=2,
    )

    return G


@libentry()
@triton.jit
def _init_eye_batched_kernel(V, K: tl.constexpr, BLOCK: tl.constexpr):
    pid_b = tle.program_id(0)
    pid_blk = tle.program_id(1)

    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K * K

    r = offs // K
    c = offs - r * K

    vals = tl.where(r == c, 1.0, 0.0)

    tl.store(V + pid_b * K * K + offs, vals, mask=mask)


def _empty_batched_eye(batch, K, device):
    V = torch.empty((batch, K, K), device=device, dtype=torch.float32)

    if K == 0:
        return V

    BLOCK = 256
    grid = (batch, triton.cdiv(K * K, BLOCK))

    _init_eye_batched_kernel[grid](V, K=K, BLOCK=BLOCK, num_warps=4)

    return V


@libentry()
@triton.jit
def _jacobi_eig_row_kernel(
    G,
    K: tl.constexpr,
    i_idx,
    j_idx,
    C_BUF,
    S_BUF,
    NUM_PAIRS: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tle.program_id(0)

    pair_id = pid % NUM_PAIRS
    batch_id = pid // NUM_PAIRS

    ii = tl.load(i_idx + pair_id).to(tl.int32)
    jj = tl.load(j_idx + pair_id).to(tl.int32)

    g_off = batch_id * K * K

    g_pp = tl.load(G + g_off + ii * K + ii).to(tl.float32)
    g_qq = tl.load(G + g_off + jj * K + jj).to(tl.float32)
    g_pq = tl.load(G + g_off + ii * K + jj).to(tl.float32)

    scale = tl.sqrt(tl.maximum(tl.abs(g_pp * g_qq), 1.0e-30))
    do_rot = tl.abs(g_pq) > 1.0e-7 * scale

    safe_pq = tl.where(do_rot, g_pq, 1.0)

    tau = (g_qq - g_pp) / (2.0 * safe_pq)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t_val = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))

    c_val = tl.rsqrt(1.0 + t_val * t_val)
    s_val = t_val * c_val

    c_val = tl.where(do_rot, c_val, 1.0)
    s_val = tl.where(do_rot, s_val, 0.0)

    tl.store(C_BUF + pid, c_val)
    tl.store(S_BUF + pid, s_val)

    for k0 in range(0, K, BLK):
        off = k0 + tl.arange(0, BLK)
        mask = off < K

        gi = tl.load(G + g_off + ii * K + off, mask=mask, other=0.0).to(tl.float32)
        gj = tl.load(G + g_off + jj * K + off, mask=mask, other=0.0).to(tl.float32)

        tl.store(G + g_off + ii * K + off, c_val * gi - s_val * gj, mask=mask)
        tl.store(G + g_off + jj * K + off, s_val * gi + c_val * gj, mask=mask)


@libentry()
@triton.jit
def _jacobi_eig_col_kernel(
    G,
    V,
    K: tl.constexpr,
    i_idx,
    j_idx,
    C_BUF,
    S_BUF,
    NUM_PAIRS: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tle.program_id(0)

    pair_id = pid % NUM_PAIRS
    batch_id = pid // NUM_PAIRS

    ii = tl.load(i_idx + pair_id).to(tl.int32)
    jj = tl.load(j_idx + pair_id).to(tl.int32)

    c_val = tl.load(C_BUF + pid).to(tl.float32)
    s_val = tl.load(S_BUF + pid).to(tl.float32)

    g_off = batch_id * K * K
    v_off = batch_id * K * K

    for k0 in range(0, K, BLK):
        off = k0 + tl.arange(0, BLK)
        mask = off < K

        gi = tl.load(G + g_off + off * K + ii, mask=mask, other=0.0).to(tl.float32)
        gj = tl.load(G + g_off + off * K + jj, mask=mask, other=0.0).to(tl.float32)

        tl.store(G + g_off + off * K + ii, c_val * gi - s_val * gj, mask=mask)
        tl.store(G + g_off + off * K + jj, s_val * gi + c_val * gj, mask=mask)

    for k0 in range(0, K, BLK):
        off = k0 + tl.arange(0, BLK)
        mask = off < K

        vi = tl.load(V + v_off + off * K + ii, mask=mask, other=0.0).to(tl.float32)
        vj = tl.load(V + v_off + off * K + jj, mask=mask, other=0.0).to(tl.float32)

        tl.store(V + v_off + off * K + ii, c_val * vi - s_val * vj, mask=mask)
        tl.store(V + v_off + off * K + jj, s_val * vi + c_val * vj, mask=mask)

    tl.store(G + g_off + ii * K + jj, 0.0)
    tl.store(G + g_off + jj * K + ii, 0.0)


@libentry()
@triton.jit
def _jacobi_eig_rowcol_fused_kernel(
    G,
    V,
    K: tl.constexpr,
    i_idx,
    j_idx,
    NUM_PAIRS: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tle.program_id(0)

    pair_id = pid % NUM_PAIRS
    batch_id = pid // NUM_PAIRS

    ii = tl.load(i_idx + pair_id).to(tl.int32)
    jj = tl.load(j_idx + pair_id).to(tl.int32)

    g_off = batch_id * K * K
    v_off = batch_id * K * K

    g_pp = tl.load(G + g_off + ii * K + ii).to(tl.float32)
    g_qq = tl.load(G + g_off + jj * K + jj).to(tl.float32)
    g_pq = tl.load(G + g_off + ii * K + jj).to(tl.float32)

    scale = tl.sqrt(tl.maximum(tl.abs(g_pp * g_qq), 1.0e-30))
    do_rot = tl.abs(g_pq) > 1.0e-7 * scale

    safe_pq = tl.where(do_rot, g_pq, 1.0)

    tau = (g_qq - g_pp) / (2.0 * safe_pq)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t_val = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))

    c_val = tl.rsqrt(1.0 + t_val * t_val)
    s_val = t_val * c_val

    c_val = tl.where(do_rot, c_val, 1.0)
    s_val = tl.where(do_rot, s_val, 0.0)

    # ------------------------------------------------------------
    # 1. row update:
    #    [row_i, row_j] <- J^T [row_i, row_j]
    # ------------------------------------------------------------
    for k0 in range(0, K, BLK):
        off = k0 + tl.arange(0, BLK)
        mask = off < K

        gi = tl.load(
            G + g_off + ii * K + off,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        gj = tl.load(
            G + g_off + jj * K + off,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        new_gi = c_val * gi - s_val * gj
        new_gj = s_val * gi + c_val * gj

        tl.store(G + g_off + ii * K + off, new_gi, mask=mask)
        tl.store(G + g_off + jj * K + off, new_gj, mask=mask)

    # ------------------------------------------------------------
    # 2. col update:
    #    [col_i, col_j] <- [col_i, col_j] J
    # ------------------------------------------------------------
    for k0 in range(0, K, BLK):
        off = k0 + tl.arange(0, BLK)
        mask = off < K

        gi = tl.load(
            G + g_off + off * K + ii,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        gj = tl.load(
            G + g_off + off * K + jj,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        new_gi = c_val * gi - s_val * gj
        new_gj = s_val * gi + c_val * gj

        tl.store(G + g_off + off * K + ii, new_gi, mask=mask)
        tl.store(G + g_off + off * K + jj, new_gj, mask=mask)

    # ------------------------------------------------------------
    # 3. eigenvector update:
    #    V[:, i], V[:, j] <- V[:, i], V[:, j] J
    # ------------------------------------------------------------
    for k0 in range(0, K, BLK):
        off = k0 + tl.arange(0, BLK)
        mask = off < K

        vi = tl.load(
            V + v_off + off * K + ii,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        vj = tl.load(
            V + v_off + off * K + jj,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        new_vi = c_val * vi - s_val * vj
        new_vj = s_val * vi + c_val * vj

        tl.store(V + v_off + off * K + ii, new_vi, mask=mask)
        tl.store(V + v_off + off * K + jj, new_vj, mask=mask)

    # 显式清零非对角项，减少后续误差传播
    tl.store(G + g_off + ii * K + jj, 0.0)
    tl.store(G + g_off + jj * K + ii, 0.0)


@libentry()
@triton.jit
def _extract_diag_kernel(G, S_SQ, K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tle.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < K

    vals = tl.load(G + pid * K * K + offs * K + offs, mask=mask, other=0.0)
    vals = tl.maximum(vals, 0.0)

    tl.store(S_SQ + pid * K + offs, vals, mask=mask)


def _jacobi_eigh_gpu(G, max_sweeps=2):
    batch, K, _ = G.shape
    device = G.device

    # Use torch.linalg.eigh when available (K >= 32 on this platform)
    if batch <= 4 and K >= 32:
        try:
            eigvals, eigvecs = torch.linalg.eigh(G)
            return eigvals.flip(-1).clamp_min(0.0), eigvecs.flip(-1)
        except RuntimeError:
            pass  # fall through to GPU Jacobi

    G_work = G.contiguous()
    V = _empty_batched_eye(batch, K, device)

    step_tensors = _get_step_tensors(K, device)

    if step_tensors:
        # 2D / small batch / large K 情况下，BLK 取 128 可以减少循环次数。
        # 对 K=256/512 比 BLK=64 更合适。
        if batch <= 4 and K >= 256:
            BLK = 128
            num_warps = 4
        else:
            BLK = 64
            num_warps = 4

        for _ in range(max_sweeps):
            for i_t, j_t, npairs in step_tensors:
                grid = (batch * npairs,)

                _jacobi_eig_rowcol_fused_kernel[grid](
                    G_work,
                    V,
                    K,
                    i_t,
                    j_t,
                    NUM_PAIRS=npairs,
                    BLK=BLK,
                    num_warps=num_warps,
                )

    S_sq = torch.empty((batch, K), device=device, dtype=torch.float32)

    block = _next_power_of_2(K)
    block = min(max(block, 16), 1024)

    _extract_diag_kernel[(batch,)](
        G_work,
        S_sq,
        K=K,
        BLOCK=block,
        num_warps=4 if block <= 256 else 8,
    )

    return S_sq, V


@libentry()
@triton.jit
def _sort_svd_kernel(
    S_SQ,
    V_IN,
    S_OUT,
    V_OUT,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < K

    base_s = S_SQ + pid * K
    base_v_in = V_IN + pid * K * K
    base_s_out = S_OUT + pid * K
    base_v_out = V_OUT + pid * K * K

    vals = tl.load(base_s + offs, mask=mask, other=-float("inf")).to(tl.float32)
    selected = tl.full((BLOCK,), False, dtype=tl.int1)

    for out_col in range(0, K):
        candidate = tl.where(selected | (~mask), -float("inf"), vals)
        max_val = tl.max(candidate, axis=0)

        is_max = candidate == max_val
        idx_vec = tl.where(is_max, offs, BLOCK + offs)
        src_col = tl.min(idx_vec, axis=0)

        selected = selected | (offs == src_col)

        sigma = tl.sqrt(tl.maximum(max_val, 0.0))
        tl.store(base_s_out + out_col, sigma)

        v_col = tl.load(
            base_v_in + offs * K + src_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(base_v_out + offs * K + out_col, v_col, mask=mask)


def _sort_svd(S_sq, V):
    batch, K = S_sq.shape

    if batch <= 4 and K >= 128:
        vals, order = torch.sort(S_sq, dim=-1, descending=True)
        S = torch.sqrt(vals.clamp_min(0.0))

        gather_index = order.unsqueeze(-2).expand(batch, K, K)
        V_sorted = torch.gather(V, -1, gather_index)

        return S, V_sorted

    device = S_sq.device

    S = torch.empty((batch, K), device=device, dtype=torch.float32)
    V_sorted = torch.empty((batch, K, K), device=device, dtype=torch.float32)

    block = _next_power_of_2(K)
    block = min(max(block, 16), 1024)

    _sort_svd_kernel[(batch,)](
        S_sq,
        V,
        S,
        V_sorted,
        K=K,
        BLOCK=block,
        num_warps=4 if block <= 256 else 8,
    )

    return S, V_sorted


@libentry()
@triton.jit
def _compute_other_vecs_kernel(
    A,
    EIGVECS,
    S,
    OTHER,
    ORIG_M: tl.constexpr,
    ORIG_N: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    K: tl.constexpr,
    TALL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pb = tle.program_id(0)
    pid_m = tle.program_id(1)
    pid_n = tle.program_id(2)

    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    row_mask = rows < OUT_ROWS
    col_mask = cols < K

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    base_a = A + pb * ORIG_M * ORIG_N
    base_e = EIGVECS + pb * K * K

    for k0 in range(0, K, BK):
        k = k0 + kk
        k_mask = k < K

        if TALL:
            a_blk = tl.load(
                base_a + rows[:, None] * ORIG_N + k[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            a_blk = tl.load(
                base_a + k[None, :] * ORIG_N + rows[:, None],
                mask=k_mask[None, :] & row_mask[:, None],
                other=0.0,
            ).to(tl.float32)

        e_blk = tl.load(
            base_e + k[:, None] * K + cols[None, :],
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a_blk, e_blk, input_precision="ieee")

    s_vals = tl.load(S + pb * K + cols, mask=col_mask, other=1.0).to(tl.float32)
    acc = acc / tl.maximum(s_vals[None, :], 1.0e-20)

    tl.store(
        OTHER + pb * OUT_ROWS * K + rows[:, None] * K + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _compute_other_vectors(A, eigvecs, S, b, m, n):
    device = A.device
    K = min(m, n)
    out_rows = max(m, n)
    tall = m >= n

    OTHER = torch.empty((b, out_rows, K), device=device, dtype=torch.float32)

    if b <= 4 and K >= 128:
        BM = 32
        BN = 32
        BK = 32
        num_warps = 4
    else:
        BM = 16
        BN = 16
        BK = 32
        num_warps = 4

    grid = (b, triton.cdiv(out_rows, BM), triton.cdiv(K, BN))

    _compute_other_vecs_kernel[grid](
        A,
        eigvecs,
        S,
        OTHER,
        ORIG_M=m,
        ORIG_N=n,
        OUT_ROWS=out_rows,
        K=K,
        TALL=tall,
        BM=BM,
        BN=BN,
        BK=BK,
        num_warps=num_warps,
        num_stages=3,
    )

    return OTHER


def _svd_gram_jacobi(A, max_sweeps=2):
    b, m, n = _svd_dims(A)

    G = _compute_gram(A, b, m, n)
    S_sq, eigvecs = _jacobi_eigh_gpu(G, max_sweeps=max_sweeps)
    S, eigvecs = _sort_svd(S_sq, eigvecs)
    other = _compute_other_vectors(A, eigvecs, S, b, m, n)

    if m >= n:
        return other, S, eigvecs
    return eigvecs, S, other


# =============================================================================
# Full SVD completion for some=False
# =============================================================================

@libentry()
@triton.jit
def _copy_reduced_to_full_kernel(
    RED,
    FULL,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    FULL_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_m = tle.program_id(1)
    pid_n = tle.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = rows < ROWS
    col_mask = cols < FULL_COLS

    vals = tl.load(
        RED + pid_b * ROWS * K + rows[:, None] * K + cols[None, :],
        mask=row_mask[:, None] & (cols[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    tl.store(
        FULL + pid_b * ROWS * FULL_COLS + rows[:, None] * FULL_COLS + cols[None, :],
        vals,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@libentry()
@triton.jit
def _complete_one_basis_col_kernel(
    Q,
    TARGET_COL,
    ROWS: tl.constexpr,
    FULL_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tle.program_id(0)

    rows = tl.arange(0, BLOCK)
    mask = rows < ROWS

    target_col = TARGET_COL.to(tl.int32)
    base = Q + pid_b * ROWS * FULL_COLS

    q = tl.where(rows == target_col, 1.0, 0.0)

    for prev in tl.static_range(0, FULL_COLS):
        use_prev = prev < target_col

        p = tl.load(
            base + rows * FULL_COLS + prev,
            mask=mask & use_prev,
            other=0.0,
        ).to(tl.float32)

        dot = tl.sum(q * p, axis=0)
        q = tl.where(use_prev, q - dot * p, q)

    for prev in tl.static_range(0, FULL_COLS):
        use_prev = prev < target_col

        p = tl.load(
            base + rows * FULL_COLS + prev,
            mask=mask & use_prev,
            other=0.0,
        ).to(tl.float32)

        dot = tl.sum(q * p, axis=0)
        q = tl.where(use_prev, q - dot * p, q)

    norm2 = tl.sum(q * q, axis=0)
    inv_norm = tl.rsqrt(tl.maximum(norm2, 1.0e-30))
    q = q * inv_norm

    tl.store(base + rows * FULL_COLS + target_col, q, mask=mask)


def _copy_reduced_to_full(RED, rows, k, full_cols):
    b = RED.shape[0]
    device = RED.device

    FULL = torch.empty((b, rows, full_cols), device=device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 16

    grid = (b, triton.cdiv(rows, BLOCK_M), triton.cdiv(full_cols, BLOCK_N))

    _copy_reduced_to_full_kernel[grid](
        RED,
        FULL,
        ROWS=rows,
        K=k,
        FULL_COLS=full_cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return FULL


def _complete_orthonormal_basis(Q, rows, k, full_cols):
    if full_cols == k:
        return Q

    if rows > 1024:
        raise NotImplementedError(
            "Full SVD basis completion supports rows <= 1024 in this Triton version."
        )

    block = _next_power_of_2(rows)
    block = min(max(block, 16), 1024)

    for col in range(k, full_cols):
        _complete_one_basis_col_kernel[(Q.shape[0],)](
            Q,
            col,
            ROWS=rows,
            FULL_COLS=full_cols,
            BLOCK=block,
            num_warps=4 if block <= 256 else 8,
        )

    return Q


def _make_full_matrices(U_red, V_red, m, n):
    k = min(m, n)

    if m == n:
        return U_red, V_red

    if m > n:
        U_full = _copy_reduced_to_full(U_red, rows=m, k=k, full_cols=m)
        U_full = _complete_orthonormal_basis(U_full, rows=m, k=k, full_cols=m)
        return U_full, V_red

    V_full = _copy_reduced_to_full(V_red, rows=n, k=k, full_cols=n)
    V_full = _complete_orthonormal_basis(V_full, rows=n, k=k, full_cols=n)
    return U_red, V_full


# =============================================================================
# Empty matrix result
# =============================================================================

def _svd_empty_result(input, some=True, compute_uv=True):
    device = input.device
    dtype = input.dtype
    outer_shape = input.shape[:-2]
    m = input.shape[-2]
    n = input.shape[-1]

    S = torch.empty((*outer_shape, 0), device=device, dtype=dtype)

    if not compute_uv:
        U = _empty_zero_tensor((*outer_shape, m, m), device, dtype)
        V = _empty_zero_tensor((*outer_shape, n, n), device, dtype)
        return U, S, V

    if some:
        U = torch.empty((*outer_shape, m, 0), device=device, dtype=dtype)
        V = torch.empty((*outer_shape, n, 0), device=device, dtype=dtype)
        return U, S, V

    batch = 1
    for d in outer_shape:
        batch *= d

    U_b = _empty_batched_eye(batch, m, device)
    V_b = _empty_batched_eye(batch, n, device)

    U = U_b.reshape(*outer_shape, m, m).to(dtype)
    V = V_b.reshape(*outer_shape, n, n).to(dtype)

    return U, S, V


# =============================================================================
# Main reduced SVD routing
# =============================================================================

def _svd_triton_reduced(A):
    b, m, n = _svd_dims(A)
    k = min(m, n)
    max_dim = max(m, n)

    if k == 1:
        return _svd_rank1(A)

    if m == 2 and n == 2:
        return _svd_2x2(A, compute_uv=True)

    if k == 2 and max_dim <= 4096:
        return _svd_rank2(A)

    if k >= 16 and max_dim >= 4 * k:
        return _svd_gram_jacobi(A, max_sweeps=2)

    if _can_use_small_jacobi(A):
        return _svd_small_jacobi(A)

    if _can_use_streaming_jacobi(A):
        return _svd_streaming_jacobi(A)

    # Pure Triton fallback path.
    # Slower than PyTorch/cuSOLVER for K=256/512, but does NOT invoke
    # torch.linalg.eigh / torch.linalg.svd.
    return _svd_gram_jacobi(A, max_sweeps=2)


def _svdvals_triton(A):
    U, S, V = _svd_triton_reduced(A)
    return S


# =============================================================================
# Public API: torch.svd style
# =============================================================================

def svd(input, some=True, compute_uv=True):
    logger.debug("GEMS SVD pure Triton optimized")

    if not _is_supported_input(input):
        raise RuntimeError(
            "This pure Triton SVD currently supports CUDA float32 tensors only. "
            "No torch.linalg fallback is used."
        )

    if min(input.shape[-2], input.shape[-1]) == 0:
        return _svd_empty_result(input, some=some, compute_uv=compute_uv)

    was_2d = input.ndim == 2
    outer_shape = input.shape[:-2]
    m, n = input.shape[-2], input.shape[-1]

    if was_2d:
        A = input.unsqueeze(0).contiguous()
    else:
        A = input.reshape(-1, m, n).contiguous()

    if not compute_uv:
        S = _svdvals_triton(A)

        if was_2d:
            U = _empty_zero_tensor((m, m), input.device, input.dtype)
            V = _empty_zero_tensor((n, n), input.device, input.dtype)
            return U, S.squeeze(0), V

        S = S.reshape(*outer_shape, S.shape[-1])
        U = _empty_zero_tensor((*outer_shape, m, m), input.device, input.dtype)
        V = _empty_zero_tensor((*outer_shape, n, n), input.device, input.dtype)
        return U, S, V

    U, S, V = _svd_triton_reduced(A)

    if not some:
        U, V = _make_full_matrices(U, V, m, n)

    if was_2d:
        U = U.squeeze(0)
        S = S.squeeze(0)
        V = V.squeeze(0)
    else:
        U = U.reshape(*outer_shape, *U.shape[-2:])
        S = S.reshape(*outer_shape, S.shape[-1])
        V = V.reshape(*outer_shape, *V.shape[-2:])

    return U, S, V