import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_SVD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit
def svd_gram_kernel(
    x,
    gram,
    batch,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_i = tle.program_id(1)
    pid_j = tle.program_id(2)

    offs_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_i = offs_i < N
    mask_j = offs_j < N

    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        m = m0 + offs_m
        mask_m = m < M
        a_i = tl.load(
            x + pid_b * M * N + m[:, None] * N + offs_i[None, :],
            mask=mask_m[:, None] & mask_i[None, :],
            other=0.0,
        ).to(tl.float32)
        a_j = tl.load(
            x + pid_b * M * N + m[:, None] * N + offs_j[None, :],
            mask=mask_m[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(tl.trans(a_i), a_j, input_precision="ieee")

    tl.store(
        gram + pid_b * N * N + offs_i[:, None] * N + offs_j[None, :],
        acc,
        mask=mask_i[:, None] & mask_j[None, :],
    )


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
    row_mask = rows < M
    x_vals = tl.load(x + pid * M + rows, mask=row_mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x_vals * x_vals, axis=0))
    inv_norm = 1.0 / tl.where(norm > 1.0e-20, norm, 1.0)
    u_vals = tl.where(norm > 1.0e-20, x_vals * inv_norm, rows == 0)

    tl.store(s + pid, norm)
    tl.store(u + pid * M + rows, u_vals, mask=row_mask)
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
    col_mask = cols < N
    x_vals = tl.load(x + pid * N + cols, mask=col_mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x_vals * x_vals, axis=0))
    inv_norm = 1.0 / tl.where(norm > 1.0e-20, norm, 1.0)
    v_vals = tl.where(norm > 1.0e-20, x_vals * inv_norm, cols == 0)

    tl.store(s + pid, norm)
    tl.store(u + pid, 1.0)
    tl.store(v + pid * N + cols, v_vals, mask=col_mask)


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
    half_diff = (ata00 - ata11) * 0.5
    half_trace = (ata00 + ata11) * 0.5
    radius = tl.sqrt(half_diff * half_diff + ata01 * ata01)
    lambda0 = tl.maximum(half_trace + radius, 0.0)
    lambda1 = tl.maximum(half_trace - radius, 0.0)
    s0 = tl.sqrt(lambda0)
    s1 = tl.sqrt(lambda1)

    s_base = offsets * 2
    tl.store(s + s_base, s0, mask=mask)
    tl.store(s + s_base + 1, s1, mask=mask)

    if compute_uv:
        use_first_eigenvector_form = ata00 >= ata11
        raw_v00 = tl.where(use_first_eigenvector_form, lambda0 - ata11, ata01)
        raw_v10 = tl.where(use_first_eigenvector_form, ata01, lambda0 - ata00)
        raw_v_norm = tl.sqrt(raw_v00 * raw_v00 + raw_v10 * raw_v10)
        inv_raw_v_norm = 1.0 / tl.where(raw_v_norm > 0.0, raw_v_norm, 1.0)
        v00 = tl.where(raw_v_norm > 0.0, raw_v00 * inv_raw_v_norm, 1.0)
        v10 = tl.where(raw_v_norm > 0.0, raw_v10 * inv_raw_v_norm, 0.0)
        v01 = -v10
        v11 = v00

        eps = 1.0e-20
        inv_s0 = 1.0 / tl.where(s0 > eps, s0, 1.0)

        av0_row0 = a * v00 + b * v10
        av0_row1 = c * v00 + d * v10
        u00 = tl.where(s0 > eps, av0_row0 * inv_s0, 1.0)
        u10 = tl.where(s0 > eps, av0_row1 * inv_s0, 0.0)

        av1_row0 = a * v01 + b * v11
        av1_row1 = c * v01 + d * v11
        perp_u01 = -u10
        perp_u11 = u00
        sign = tl.where(
            perp_u01 * av1_row0 + perp_u11 * av1_row1 >= 0.0,
            1.0,
            -1.0,
        )
        use_direct_u1 = s1 > s0 * 2.0e-1
        inv_s1 = 1.0 / tl.where(use_direct_u1, s1, 1.0)
        u01 = tl.where(use_direct_u1, av1_row0 * inv_s1, sign * perp_u01)
        u11 = tl.where(use_direct_u1, av1_row1 * inv_s1, sign * perp_u11)
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


@libentry()
@triton.jit
def svd_4x4_gram_kernel(
    x,
    u,
    s,
    v,
    batch,
    BLOCK_B: tl.constexpr,
    NUM_SWEEPS: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    batch_mask = pid < batch
    idx = tl.arange(0, 4)
    rows = tl.arange(0, 4)
    cols = tl.arange(0, 4)

    x_base = pid[:, None, None] * 16
    a = tl.load(
        x + x_base + rows[None, :, None] * 4 + cols[None, None, :],
        mask=batch_mask[:, None, None],
        other=0.0,
    ).to(tl.float32)

    a0 = tl.sum(
        tl.where(
            cols[None, None, :] == 0,
            a,
            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
        ),
        axis=2,
    )
    a1 = tl.sum(
        tl.where(
            cols[None, None, :] == 1,
            a,
            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
        ),
        axis=2,
    )
    a2 = tl.sum(
        tl.where(
            cols[None, None, :] == 2,
            a,
            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
        ),
        axis=2,
    )
    a3 = tl.sum(
        tl.where(
            cols[None, None, :] == 3,
            a,
            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
        ),
        axis=2,
    )
    g00 = tl.sum(a0 * a0, axis=1)
    g01 = tl.sum(a0 * a1, axis=1)
    g02 = tl.sum(a0 * a2, axis=1)
    g03 = tl.sum(a0 * a3, axis=1)
    g11 = tl.sum(a1 * a1, axis=1)
    g12 = tl.sum(a1 * a2, axis=1)
    g13 = tl.sum(a1 * a3, axis=1)
    g22 = tl.sum(a2 * a2, axis=1)
    g23 = tl.sum(a2 * a3, axis=1)
    g33 = tl.sum(a3 * a3, axis=1)

    g = tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32)
    g = tl.where(
        (rows[None, :, None] == 0) & (cols[None, None, :] == 0),
        g00[:, None, None],
        g,
    )
    g = tl.where(
        ((rows[None, :, None] == 0) & (cols[None, None, :] == 1))
        | ((rows[None, :, None] == 1) & (cols[None, None, :] == 0)),
        g01[:, None, None],
        g,
    )
    g = tl.where(
        ((rows[None, :, None] == 0) & (cols[None, None, :] == 2))
        | ((rows[None, :, None] == 2) & (cols[None, None, :] == 0)),
        g02[:, None, None],
        g,
    )
    g = tl.where(
        ((rows[None, :, None] == 0) & (cols[None, None, :] == 3))
        | ((rows[None, :, None] == 3) & (cols[None, None, :] == 0)),
        g03[:, None, None],
        g,
    )
    g = tl.where(
        (rows[None, :, None] == 1) & (cols[None, None, :] == 1),
        g11[:, None, None],
        g,
    )
    g = tl.where(
        ((rows[None, :, None] == 1) & (cols[None, None, :] == 2))
        | ((rows[None, :, None] == 2) & (cols[None, None, :] == 1)),
        g12[:, None, None],
        g,
    )
    g = tl.where(
        ((rows[None, :, None] == 1) & (cols[None, None, :] == 3))
        | ((rows[None, :, None] == 3) & (cols[None, None, :] == 1)),
        g13[:, None, None],
        g,
    )
    g = tl.where(
        (rows[None, :, None] == 2) & (cols[None, None, :] == 2),
        g22[:, None, None],
        g,
    )
    g = tl.where(
        ((rows[None, :, None] == 2) & (cols[None, None, :] == 3))
        | ((rows[None, :, None] == 3) & (cols[None, None, :] == 2)),
        g23[:, None, None],
        g,
    )
    g = tl.where(
        (rows[None, :, None] == 3) & (cols[None, None, :] == 3),
        g33[:, None, None],
        g,
    )
    eye = tl.where(
        rows[None, :, None] == cols[None, None, :],
        tl.full((BLOCK_B, 4, 4), 1.0, dtype=tl.float32),
        tl.full((BLOCK_B, 4, 4), 0.0, dtype=tl.float32),
    )
    v_work = eye

    for _ in range(NUM_SWEEPS):
        for p in range(4):
            for q in range(p + 1, 4):
                app = tl.sum(
                    tl.sum(
                        tl.where(
                            (rows[None, :, None] == p) & (cols[None, None, :] == p),
                            g,
                            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                        ),
                        axis=2,
                    ),
                    axis=1,
                )
                aqq = tl.sum(
                    tl.sum(
                        tl.where(
                            (rows[None, :, None] == q) & (cols[None, None, :] == q),
                            g,
                            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                        ),
                        axis=2,
                    ),
                    axis=1,
                )
                apq = tl.sum(
                    tl.sum(
                        tl.where(
                            (rows[None, :, None] == p) & (cols[None, None, :] == q),
                            g,
                            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                        ),
                        axis=2,
                    ),
                    axis=1,
                )
                should_rotate = tl.abs(apq) > 1.0e-7 * tl.sqrt(
                    tl.abs(app * aqq) + 1.0e-30
                )
                tau = (aqq - app) / (2.0 * tl.where(should_rotate, apq, 1.0))
                sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                sn = t * c
                c = tl.where(should_rotate, c, 1.0)
                sn = tl.where(should_rotate, sn, 0.0)

                g_p_col = tl.sum(
                    tl.where(
                        cols[None, None, :] == p,
                        g,
                        tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                    ),
                    axis=2,
                )
                g_q_col = tl.sum(
                    tl.where(
                        cols[None, None, :] == q,
                        g,
                        tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                    ),
                    axis=2,
                )
                new_p_col = c[:, None] * g_p_col - sn[:, None] * g_q_col
                new_q_col = sn[:, None] * g_p_col + c[:, None] * g_q_col
                g = tl.where(cols[None, None, :] == p, new_p_col[:, :, None], g)
                g = tl.where(cols[None, None, :] == q, new_q_col[:, :, None], g)

                g_p_row = tl.sum(
                    tl.where(
                        rows[None, :, None] == p,
                        g,
                        tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                    ),
                    axis=1,
                )
                g_q_row = tl.sum(
                    tl.where(
                        rows[None, :, None] == q,
                        g,
                        tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                    ),
                    axis=1,
                )
                new_p_row = c[:, None] * g_p_row - sn[:, None] * g_q_row
                new_q_row = sn[:, None] * g_p_row + c[:, None] * g_q_row
                g = tl.where(rows[None, :, None] == p, new_p_row[:, None, :], g)
                g = tl.where(rows[None, :, None] == q, new_q_row[:, None, :], g)

                v_p_col = tl.sum(
                    tl.where(
                        cols[None, None, :] == p,
                        v_work,
                        tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                    ),
                    axis=2,
                )
                v_q_col = tl.sum(
                    tl.where(
                        cols[None, None, :] == q,
                        v_work,
                        tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
                    ),
                    axis=2,
                )
                new_v_p = c[:, None] * v_p_col - sn[:, None] * v_q_col
                new_v_q = sn[:, None] * v_p_col + c[:, None] * v_q_col
                v_work = tl.where(
                    cols[None, None, :] == p,
                    new_v_p[:, :, None],
                    v_work,
                )
                v_work = tl.where(
                    cols[None, None, :] == q,
                    new_v_q[:, :, None],
                    v_work,
                )

    diag_vals = tl.sum(
        tl.where(
            rows[None, :, None] == cols[None, None, :],
            g,
            tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
        ),
        axis=1,
    )
    s_vals = tl.sqrt(tl.maximum(diag_vals, 0.0))
    ranks = tl.sum(
        (
            (s_vals[:, :, None] > s_vals[:, None, :])
            | (
                (s_vals[:, :, None] == s_vals[:, None, :])
                & (idx[None, :, None] < idx[None, None, :])
            )
        ).to(tl.int32),
        axis=1,
    )
    tl.store(s + pid[:, None] * 4 + ranks, s_vals, mask=batch_mask[:, None])

    for j in range(4):
        rank_j = tl.sum(
            tl.where(idx[None, :] == j, ranks, tl.zeros((BLOCK_B, 4), tl.int32)),
            axis=1,
        )
        s_j = tl.sum(
            tl.where(idx[None, :] == j, s_vals, tl.zeros((BLOCK_B, 4), tl.float32)),
            axis=1,
        )
        v_j = tl.sum(
            tl.where(
                cols[None, None, :] == j,
                v_work,
                tl.zeros((BLOCK_B, 4, 4), dtype=tl.float32),
            ),
            axis=2,
        )
        tl.store(
            v + pid[:, None] * 16 + rows[None, :] * 4 + rank_j[:, None],
            v_j,
            mask=batch_mask[:, None],
        )

        av_j = tl.sum(a * v_j[:, None, :], axis=2)
        u_j = av_j / tl.where(s_j[:, None] > 1.0e-20, s_j[:, None], 1.0)
        tl.store(
            u + pid[:, None] * 16 + rows[None, :] * 4 + rank_j[:, None],
            u_j,
            mask=batch_mask[:, None],
        )


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
    row_mask = rows < M
    x_base = pid * M * 2
    x0 = tl.load(x + x_base + rows * 2, mask=row_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x + x_base + rows * 2 + 1, mask=row_mask, other=0.0).to(tl.float32)

    ata00 = tl.sum(x0 * x0, axis=0)
    ata01 = tl.sum(x0 * x1, axis=0)
    ata11 = tl.sum(x1 * x1, axis=0)
    half_diff = (ata00 - ata11) * 0.5
    half_trace = (ata00 + ata11) * 0.5
    radius = tl.sqrt(half_diff * half_diff + ata01 * ata01)
    lambda0 = tl.maximum(half_trace + radius, 0.0)
    lambda1 = tl.maximum(half_trace - radius, 0.0)
    s0 = tl.sqrt(lambda0)
    s1 = tl.sqrt(lambda1)

    use_first_eigenvector_form = ata00 >= ata11
    raw_v00 = tl.where(use_first_eigenvector_form, lambda0 - ata11, ata01)
    raw_v10 = tl.where(use_first_eigenvector_form, ata01, lambda0 - ata00)
    raw_v_norm = tl.sqrt(raw_v00 * raw_v00 + raw_v10 * raw_v10)
    inv_raw_v_norm = 1.0 / tl.where(raw_v_norm > 0.0, raw_v_norm, 1.0)
    v00 = tl.where(raw_v_norm > 0.0, raw_v00 * inv_raw_v_norm, 1.0)
    v10 = tl.where(raw_v_norm > 0.0, raw_v10 * inv_raw_v_norm, 0.0)
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
    tl.store(u + u_base + rows * 2, u0, mask=row_mask)
    tl.store(u + u_base + rows * 2 + 1, u1, mask=row_mask)

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
    col_mask = cols < N
    x_base = pid * 2 * N
    x0 = tl.load(x + x_base + cols, mask=col_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x + x_base + N + cols, mask=col_mask, other=0.0).to(tl.float32)

    aat00 = tl.sum(x0 * x0, axis=0)
    aat01 = tl.sum(x0 * x1, axis=0)
    aat11 = tl.sum(x1 * x1, axis=0)
    half_diff = (aat00 - aat11) * 0.5
    half_trace = (aat00 + aat11) * 0.5
    radius = tl.sqrt(half_diff * half_diff + aat01 * aat01)
    lambda0 = tl.maximum(half_trace + radius, 0.0)
    lambda1 = tl.maximum(half_trace - radius, 0.0)
    s0 = tl.sqrt(lambda0)
    s1 = tl.sqrt(lambda1)

    use_first_eigenvector_form = aat00 >= aat11
    raw_u00 = tl.where(use_first_eigenvector_form, lambda0 - aat11, aat01)
    raw_u10 = tl.where(use_first_eigenvector_form, aat01, lambda0 - aat00)
    raw_u_norm = tl.sqrt(raw_u00 * raw_u00 + raw_u10 * raw_u10)
    inv_raw_u_norm = 1.0 / tl.where(raw_u_norm > 0.0, raw_u_norm, 1.0)
    u00 = tl.where(raw_u_norm > 0.0, raw_u00 * inv_raw_u_norm, 1.0)
    u10 = tl.where(raw_u_norm > 0.0, raw_u10 * inv_raw_u_norm, 0.0)
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
    tl.store(v + v_base + cols * 2, v0, mask=col_mask)
    tl.store(v + v_base + cols * 2 + 1, v1, mask=col_mask)


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

    x_base = pid * M * N
    a = tl.load(
        x + x_base + rows[:, None] * N + cols[None, :],
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

                abs_gamma = tl.abs(gamma)
                threshold = 1.0e-7 * tl.sqrt(alpha * beta + 1.0e-30)
                should_rotate = abs_gamma >= threshold
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
            ((s_vals[:, None] > s_vals[None, :]))
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

                should_rotate = tl.abs(gamma) >= 1.0e-7 * tl.sqrt(
                    alpha * beta + 1.0e-30
                )
                safe_gamma = tl.where(should_rotate, gamma, 1.0)
                tau = (beta - alpha) / (2.0 * safe_gamma)
                tau_sign = tl.where(tau >= 0.0, 1.0, -1.0)
                t = tau_sign / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
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
        ranks += (((s_j > s_vals) | ((s_j == s_vals) & (j_vec < cols))) & col_mask).to(
            tl.int32
        )

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


def _can_use_2x2_kernel(self):
    return (
        self.dtype in _SUPPORTED_SVD_DTYPES
        and self.is_contiguous()
        and self.shape[-2:] == (2, 2)
    )


def _can_use_rank1_kernel(self, some, compute_uv):
    return (
        some
        and compute_uv
        and self.dtype in _SUPPORTED_SVD_DTYPES
        and self.is_contiguous()
        and len(self.shape) >= 2
        and min(self.shape[-2:]) == 1
        and max(self.shape[-2:]) <= 4096
    )


def _can_use_rank2_kernel(self, some, compute_uv):
    return (
        some
        and compute_uv
        and self.dtype in _SUPPORTED_SVD_DTYPES
        and self.is_contiguous()
        and len(self.shape) >= 2
        and self.shape[-2:] != (2, 2)
        and (self.shape[-1] == 2 or self.shape[-2] == 2)
        and max(self.shape[-2:]) <= 4096
    )


def _can_use_4x4_kernel(self, some, compute_uv):
    return (
        some
        and compute_uv
        and self.dtype in _SUPPORTED_SVD_DTYPES
        and self.is_contiguous()
        and len(self.shape) >= 2
        and self.shape[-2:] == (4, 4)
    )


def _can_use_small_jacobi_kernel(self, some, compute_uv):
    if not (
        some
        and compute_uv
        and self.dtype in _SUPPORTED_SVD_DTYPES
        and self.is_contiguous()
        and len(self.shape) >= 2
    ):
        return False
    m, n = self.shape[-2:]
    k = min(m, n)
    max_dim = max(m, n)
    return (
        (k == 3 and max_dim <= 1024)
        or (k == 4 and max_dim <= 1024)
        or (5 <= k <= 8 and max_dim <= 512)
        or (k == 16 and 64 <= max_dim <= 256)
        or (k == 32 and 64 <= max_dim <= 1024)
    )


def _can_use_streaming_jacobi_kernel(self, some, compute_uv):
    if not (
        some
        and compute_uv
        and self.dtype in _SUPPORTED_SVD_DTYPES
        and self.is_contiguous()
        and len(self.shape) >= 2
    ):
        return False
    m, n = self.shape[-2:]
    if m == 0 or n == 0:
        return False
    k = min(m, n)
    max_dim = max(m, n)
    batch_size = self.numel() // (m * n)
    return (k == 64 and max_dim <= 1024 and batch_size >= 16) or (
        k == 128 and max_dim <= 128 and batch_size >= 16
    )


def _can_use_gram_eigh_kernel(self, some, compute_uv):
    if not (
        some
        and compute_uv
        and self.dtype in _SUPPORTED_SVD_DTYPES
        and self.is_contiguous()
        and len(self.shape) >= 2
    ):
        return False
    m, n = self.shape[-2:]
    if m == 0 or n == 0:
        return False
    k = min(m, n)
    return 1 <= k <= 1024


def _svd_2x2(self, compute_uv):
    batch_shape = self.shape[:-2]
    batch = self.numel() // 4
    u = torch.empty(batch_shape + (2, 2), dtype=self.dtype, device=self.device)
    s = torch.empty(batch_shape + (2,), dtype=self.dtype, device=self.device)
    v = torch.empty(batch_shape + (2, 2), dtype=self.dtype, device=self.device)
    block_size = 256
    grid = (triton.cdiv(batch, block_size),)
    svd_2x2_kernel[grid](self, u, s, v, batch, compute_uv, BLOCK_SIZE=block_size)
    return u, s, v


def _svd_rank1(self):
    batch_shape = self.shape[:-2]
    m = self.shape[-2]
    n = self.shape[-1]
    batch = self.numel() // (m * n)
    u = torch.empty(batch_shape + (m, 1), dtype=self.dtype, device=self.device)
    s = torch.empty(batch_shape + (1,), dtype=self.dtype, device=self.device)
    v = torch.empty(batch_shape + (n, 1), dtype=self.dtype, device=self.device)

    if n == 1:
        block_m = triton.next_power_of_2(m)
        svd_mx1_kernel[(batch,)](self, u, s, v, batch, m, BLOCK_M=block_m)
    else:
        block_n = triton.next_power_of_2(n)
        svd_1xn_kernel[(batch,)](self, u, s, v, batch, n, BLOCK_N=block_n)
    return u, s, v


def _svd_rank2(self):
    batch_shape = self.shape[:-2]
    m = self.shape[-2]
    n = self.shape[-1]
    batch = self.numel() // (m * n)
    u = torch.empty(batch_shape + (m, 2), dtype=self.dtype, device=self.device)
    s = torch.empty(batch_shape + (2,), dtype=self.dtype, device=self.device)
    v = torch.empty(batch_shape + (n, 2), dtype=self.dtype, device=self.device)

    if n == 2:
        block_m = triton.next_power_of_2(m)
        svd_mx2_kernel[(batch,)](self, u, s, v, batch, m, BLOCK_M=block_m)
    else:
        block_n = triton.next_power_of_2(n)
        svd_2xn_kernel[(batch,)](self, u, s, v, batch, n, BLOCK_N=block_n)
    return u, s, v


def _svd_4x4(self):
    batch_shape = self.shape[:-2]
    batch = self.numel() // 16
    u = torch.empty(batch_shape + (4, 4), dtype=self.dtype, device=self.device)
    s = torch.empty(batch_shape + (4,), dtype=self.dtype, device=self.device)
    v = torch.empty(batch_shape + (4, 4), dtype=self.dtype, device=self.device)
    block_b = 16
    grid = (triton.cdiv(batch, block_b),)
    svd_4x4_gram_kernel[grid](
        self,
        u,
        s,
        v,
        batch,
        BLOCK_B=block_b,
        NUM_SWEEPS=8,
    )
    return u, s, v


def _svd_small_jacobi(self):
    transpose = self.shape[-2] < self.shape[-1]
    inp = self.mT.contiguous() if transpose else self
    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2:]
    batch = inp.numel() // (m * n)

    u = torch.empty(batch_shape + (m, n), dtype=inp.dtype, device=inp.device)
    s = torch.empty(batch_shape + (n,), dtype=inp.dtype, device=inp.device)
    v = torch.empty(batch_shape + (n, n), dtype=inp.dtype, device=inp.device)
    block_m = triton.next_power_of_2(m)
    block_n = triton.next_power_of_2(n)
    num_sweeps = 40 if min(m, n) <= 4 else 6 if min(m, n) == 32 else 10
    svd_small_jacobi_kernel[(batch,)](
        inp,
        u,
        s,
        v,
        batch,
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_SWEEPS=num_sweeps,
    )

    if transpose:
        return v, s, u
    return u, s, v


def _svd_streaming_jacobi(self):
    transpose = self.shape[-2] < self.shape[-1]
    inp = self.mT.contiguous() if transpose else self
    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2:]
    batch = inp.numel() // (m * n)

    u = torch.empty(batch_shape + (m, n), dtype=inp.dtype, device=inp.device)
    s = torch.empty(batch_shape + (n,), dtype=inp.dtype, device=inp.device)
    v = torch.empty(batch_shape + (n, n), dtype=inp.dtype, device=inp.device)
    a_work = torch.empty(batch, n, m, dtype=torch.float32, device=inp.device)
    v_work = torch.empty(batch, n, n, dtype=torch.float32, device=inp.device)
    block_m = triton.next_power_of_2(m)
    block_n = triton.next_power_of_2(n)
    num_sweeps = 10 if n == 128 else 8
    svd_streaming_jacobi_kernel[(batch,)](
        inp,
        a_work,
        v_work,
        u,
        s,
        v,
        batch,
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
        return v, s, u
    return u, s, v


def _svd_gram(self):
    transpose = self.shape[-2] < self.shape[-1]
    inp = self.mT.contiguous() if transpose else self
    batch_shape = inp.shape[:-2]
    m, n = inp.shape[-2:]
    batch = inp.numel() // (m * n)

    gram = torch.empty(batch_shape + (n, n), dtype=torch.float32, device=inp.device)
    block_n = 32
    block_m = 32
    grid = (batch, triton.cdiv(n, block_n), triton.cdiv(n, block_n))
    svd_gram_kernel[grid](
        inp,
        gram,
        batch,
        M=m,
        N=n,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return gram


def _svd_gram_eigh(self):
    transpose = self.shape[-2] < self.shape[-1]
    inp = self.mT.contiguous() if transpose else self

    gram = _svd_gram(inp)
    evals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, dim=-1, descending=True)
    evals = torch.gather(evals, -1, order).clamp_min(0.0)
    gather_index = order.unsqueeze(-2).expand(
        *order.shape[:-1],
        eigvecs.shape[-2],
        order.shape[-1],
    )
    v = torch.gather(eigvecs, -1, gather_index)
    s = torch.sqrt(evals)
    s_max = s.amax(dim=-1, keepdim=True)
    rel_cutoff = 1.0e-4 if self.dtype in (torch.float16, torch.bfloat16) else 1.0e-7
    s_cutoff = (s_max * rel_cutoff).clamp_min(1.0e-12)
    valid_s = s > s_cutoff
    safe_s = torch.where(valid_s, s, torch.ones_like(s))
    u = inp.float() @ v / safe_s.unsqueeze(-2)
    u = torch.where(valid_s.unsqueeze(-2), u, torch.zeros_like(u))

    if transpose:
        return v.to(self.dtype), s.to(self.dtype), u.to(self.dtype)
    return u.to(self.dtype), s.to(self.dtype), v.to(self.dtype)


def _svd_compute_uv_false(self):
    if self.dtype in (torch.float16, torch.bfloat16):
        s = torch.linalg.svdvals(self.float()).to(self.dtype)
    else:
        s = torch.linalg.svdvals(self)
    m = self.shape[-2]
    n = self.shape[-1]
    batch_shape = self.shape[:-2]
    u = torch.zeros(batch_shape + (m, m), dtype=self.dtype, device=self.device)
    v = torch.zeros(batch_shape + (n, n), dtype=self.dtype, device=self.device)
    return u, s, v


def _svd_torch_fallback(self, some):
    if self.dtype in (torch.float16, torch.bfloat16):
        u, s, vh = torch.linalg.svd(self.float(), full_matrices=not some)
        return u.to(self.dtype), s.to(self.dtype), vh.mH.to(self.dtype)

    u, s, vh = torch.linalg.svd(self, full_matrices=not some)
    return u, s, vh.mH


def svd(self, some=True, compute_uv=True):
    logger.debug("GEMS SVD")

    if _can_use_2x2_kernel(self):
        return _svd_2x2(self, compute_uv)

    if _can_use_rank1_kernel(self, some, compute_uv):
        return _svd_rank1(self)

    if _can_use_rank2_kernel(self, some, compute_uv):
        return _svd_rank2(self)

    if _can_use_4x4_kernel(self, some, compute_uv):
        return _svd_4x4(self)

    if _can_use_small_jacobi_kernel(self, some, compute_uv):
        return _svd_small_jacobi(self)

    if _can_use_streaming_jacobi_kernel(self, some, compute_uv):
        return _svd_streaming_jacobi(self)

    if _can_use_gram_eigh_kernel(self, some, compute_uv):
        return _svd_gram_eigh(self)

    if not compute_uv:
        return _svd_compute_uv_false(self)

    return _svd_torch_fallback(self, some)
