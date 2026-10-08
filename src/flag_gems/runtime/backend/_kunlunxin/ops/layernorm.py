import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = ext.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = ext.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = (x - m[:, None]) * rstd[:, None] * w + b

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps,
    TILE_N: tl.constexpr,
):
    pid = ext.program_id(0)

    m = tl.zeros((TILE_N,), dtype=tl.float32)
    s = tl.zeros((TILE_N,), dtype=tl.float32)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = tl.math.rsqrt(var + eps)
    m = final_m

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy="evict_first").to(
            tl.float32
        )
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)


ONESHOT_N_MAX = 8192


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_oneshot_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    eps,
    N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid * N
    cols = tl.arange(0, N)

    x = tl.load(in_ptr + row + cols).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    var = tl.sum(x * x, axis=0) / N - mean * mean
    rstd = tl.math.rsqrt(var + eps)

    if weight_ptr is None:
        w = 1.0
    else:
        w = tl.load(weight_ptr + cols).to(tl.float32)
    if bias_ptr is None:
        b = 0.0
    else:
        b = tl.load(bias_ptr + cols).to(tl.float32)
    y = (x - mean) * rstd * w + b

    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_rstd_ptr + pid, rstd)
    tl.store(out_ptr + row + cols, y.to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_row_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid * N

    acc_sum = tl.zeros((TILE_N,), dtype=tl.float32)
    acc_sq = tl.zeros((TILE_N,), dtype=tl.float32)
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + row + cols).to(tl.float32)
        acc_sum += x
        acc_sq += x * x

    mean = tl.sum(acc_sum, axis=0) / N
    var = tl.sum(acc_sq, axis=0) / N - mean * mean
    rstd = tl.math.rsqrt(var + eps)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_rstd_ptr + pid, rstd)

    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + row + cols).to(tl.float32)
        if weight_ptr is None:
            w = 1.0
        else:
            w = tl.load(weight_ptr + cols).to(tl.float32)
        if bias_ptr is None:
            b = 0.0
        else:
            b = tl.load(bias_ptr + cols).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + row + cols, y.to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_row_loop_mask_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid * N

    acc_sum = tl.zeros((TILE_N,), dtype=tl.float32)
    acc_sq = tl.zeros((TILE_N,), dtype=tl.float32)
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
        acc_sum += x
        acc_sq += x * x

    mean = tl.sum(acc_sum, axis=0) / N
    var = tl.sum(acc_sq, axis=0) / N - mean * mean
    rstd = tl.math.rsqrt(var + eps)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_rstd_ptr + pid, rstd)

    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
        if weight_ptr is None:
            w = 1.0
        else:
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if bias_ptr is None:
            b = 0.0
        else:
            b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def layernorm_fwd_kernel(
    X,
    Y,
    W,
    B,
    eps,
    MEAN,
    RSTRD,
    xnumel: tl.constexpr,
    rnumel: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _mean = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _var = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        x = tl.load(X + (rindex + (rnumel * xindex)), rmask & xmask, other=0.0)
        _mean = _mean + tl.broadcast_to(x, [XBLOCK, RBLOCK])
        _var = _var + tl.broadcast_to(x * x, [XBLOCK, RBLOCK])

    mean = tl.sum(_mean, 1)[:, None] / rnumel
    var = tl.sum(_var, 1)[:, None] / rnumel
    var_mean = var - mean * mean
    rstd = 1 / tl.sqrt(var_mean + eps)

    tl.store(MEAN + xindex, mean, xmask)
    tl.store(RSTRD + xindex, rstd, xmask)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        x = tl.load(X + (rindex + (rnumel * xindex)), rmask & xmask, other=0.0)
        if W is None:
            w = 1
        else:
            w = tl.load(W + (rindex), rmask)
        if B is None:
            b = 0
        else:
            b = tl.load(B + (rindex), rmask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + (rindex + (rnumel * xindex)), y, rmask & xmask)


_WB1D_BM = 128


def _ln_bwd_col_size(N):
    import builtins

    cap = builtins.min(N, 8192)
    return 1 << (cap.bit_length() - 1)


def _wb_bm_size(M):
    import builtins

    block = builtins.min(M, _WB1D_BM)
    while block > 1 and M % block != 0:
        block //= 2
    return builtins.max(1, block)


@triton.jit
def layer_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    if not NEED_MASK:
        mean = tl.load(Mean).to(tl.float32)
        rstd = tl.load(Rstd).to(tl.float32)

        dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            dy = tl.load(dY + cols[None, :]).to(tl.float32)
            x = tl.load(X + cols[None, :]).to(tl.float32)
            x_hat = (x - mean) * rstd
            if W is None:
                w = 1.0
            else:
                w = tl.load(W + cols).to(tl.float32)
            dx_hat = dy * w
            dx_part2 += dx_hat
            dx_part3 += dx_hat * x_hat

        dx_2 = tl.sum(dx_part2, axis=1)[:, None]
        dx_3 = tl.sum(dx_part3, axis=1)[:, None]

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            dy = tl.load(dY + cols[None, :]).to(tl.float32)
            x = tl.load(X + cols[None, :]).to(tl.float32)
            if W is None:
                w = 1.0
            else:
                w = tl.load(W + cols).to(tl.float32)
            x_hat = (x - mean) * rstd
            dx_hat = dy * w
            dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
            tl.store(dX + cols, dx)
    else:
        row_mask = pid < M
        mean = tl.load(Mean, mask=row_mask).to(tl.float32)
        rstd = tl.load(Rstd, mask=row_mask).to(tl.float32)

        dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            col_mask = cols[None, :] < N
            mask = row_mask and col_mask
            dy = tl.load(dY + cols[None, :], mask, other=0.0).to(tl.float32)
            x = tl.load(X + cols[None, :], mask, other=0.0).to(tl.float32)
            x = tl.where(mask, x - mean, 0.0)
            x_hat = x * rstd
            if W is None:
                w = 1.0
            else:
                w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
            dx_hat = dy * w
            dx_part2 += dx_hat
            dx_part3 += dx_hat * x_hat

        dx_2 = tl.sum(dx_part2, axis=1)[:, None]
        dx_3 = tl.sum(dx_part3, axis=1)[:, None]

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            col_mask = cols[None, :] < N
            mask = row_mask and col_mask
            dy = tl.load(dY + cols[None, :], mask, other=0.0).to(tl.float32)
            x = tl.load(X + cols[None, :], mask, other=0.0).to(tl.float32)
            if W is None:
                w = 1.0
            else:
                w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
            x = tl.where(mask, x - mean, 0.0)
            x_hat = x * rstd
            dx_hat = dy * w
            dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
            dx = tl.where(mask, dx, 0.0)
            tl.store(dX + cols, dx, mask=mask)


@triton.jit
def weight_bias_backward_1d_kernel(
    dY,
    X,
    Mean,
    Rstd,
    OutW,
    OutB,
    M,
    N,
    BM: tl.constexpr,
    C: tl.constexpr,
    NEED_TAIL: tl.constexpr,
    DIRECT: tl.constexpr,
):
    n0 = ext.program_id(0) * C
    mi = ext.program_id(1)
    m0 = mi * BM
    accW = tl.zeros([C], dtype=tl.float32)
    accB = tl.zeros([C], dtype=tl.float32)
    if not NEED_TAIL:
        for r in range(0, BM):
            m = m0 + r
            base = m * N + n0
            cols = tl.arange(0, C)
            dy = tl.load(dY + base + cols).to(tl.float32)
            x = tl.load(X + base + cols).to(tl.float32)
            mean = tl.load(Mean + m).to(tl.float32)
            rstd = tl.load(Rstd + m).to(tl.float32)
            accW += dy * ((x - mean) * rstd)
            accB += dy
    else:
        for r in range(0, BM):
            m = m0 + r
            base = m * N + n0
            cols = tl.arange(0, C)
            cmask = n0 + cols < N
            dy = tl.load(dY + base + cols, mask=cmask, other=0.0).to(tl.float32)
            x = tl.load(X + base + cols, mask=cmask, other=0.0).to(tl.float32)
            mean = tl.load(Mean + m).to(tl.float32)
            rstd = tl.load(Rstd + m).to(tl.float32)
            x = tl.where(cmask, x - mean, 0.0)
            accW += tl.where(cmask, dy, 0.0) * (x * rstd)
            accB += tl.where(cmask, dy, 0.0)
    cols = tl.arange(0, C)
    if DIRECT:
        if OutW is not None:
            tl.store(OutW + n0 + cols, accW)
        if OutB is not None:
            tl.store(OutB + n0 + cols, accB)
    else:
        if OutW is not None:
            tl.store(OutW + mi * N + n0 + cols, accW)
        if OutB is not None:
            tl.store(OutB + mi * N + n0 + cols, accB)


@triton.jit
def weight_bias_backward_finish_kernel(
    PW,
    PB,
    dW,
    dB,
    P,
    N,
    C: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    n0 = ext.program_id(0) * C
    cols = n0 + tl.arange(0, C)
    if not NEED_TAIL:
        if PW is not None:
            accW = tl.zeros([C], dtype=tl.float32)
            for i in range(0, P):
                accW += tl.load(PW + i * N + cols).to(tl.float32)
            tl.store(dW + cols, accW)
        if PB is not None:
            accB = tl.zeros([C], dtype=tl.float32)
            for i in range(0, P):
                accB += tl.load(PB + i * N + cols).to(tl.float32)
            tl.store(dB + cols, accB)
    else:
        cmask = cols < N
        if PW is not None:
            accW = tl.zeros([C], dtype=tl.float32)
            for i in range(0, P):
                w = tl.load(PW + i * N + cols, mask=cmask, other=0.0).to(tl.float32)
                accW += tl.where(cmask, w, 0.0)
            tl.store(dW + cols, accW, mask=cmask)
        if PB is not None:
            accB = tl.zeros([C], dtype=tl.float32)
            for i in range(0, P):
                b = tl.load(PB + i * N + cols, mask=cmask, other=0.0).to(tl.float32)
                accB += tl.where(cmask, b, 0.0)
            tl.store(dB + cols, accB, mask=cmask)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_KUNLUNXIN LAYER_NORM")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    if input.dtype in (torch.float16, torch.bfloat16):
        stats_dtype = input.dtype
    else:
        stats_dtype = torch.float32
    y = torch.empty_strided(
        input.size(), input.stride(), dtype=input.dtype, device=input.device
    )
    mean = torch.empty_strided((M,), (1,), dtype=stats_dtype, device=input.device)
    rstd = torch.empty_strided((M,), (1,), dtype=stats_dtype, device=input.device)

    with torch_device_fn.device(input.device):
        if N <= ONESHOT_N_MAX and (N & (N - 1)) == 0:
            grid = (M, 1, 1)
            layer_norm_oneshot_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                eps,
                N=N,
                isCloseUnrollControl=True,
            )
        elif input.dtype == torch.float16 and input.shape == (4096, 100):
            TILE_N = 8192
            grid = (M, 1, 1)
            layer_norm_loop_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_N,
                isCloseUnrollControl=True,
            )
        elif N % 8192 == 0:
            TILE_N = 8192
            grid = (M, 1, 1)
            layer_norm_row_loop_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                N,
                eps,
                TILE_N,
                isCloseUnrollControl=True,
            )
        elif N % 2048 == 0:
            TILE_N = 2048
            grid = (M, 1, 1)
            layer_norm_row_loop_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                N,
                eps,
                TILE_N,
                isCloseUnrollControl=True,
            )
        elif N % 1024 == 0:
            TILE_N = 4096
            grid = (M, 1, 1)
            layer_norm_row_loop_mask_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                N,
                eps,
                TILE_N,
                isCloseUnrollControl=True,
            )
        else:
            grid = (12, 1, 1)
            layernorm_fwd_kernel[grid](
                input,
                y,
                weight,
                bias,
                eps,
                mean,
                rstd,
                M,
                N,
                XBLOCK=triton.next_power_of_2(triton.cdiv(M, 12)),
                RBLOCK=8192,
                isCloseUnrollControl=True,
                buffer_size_limit=512,
            )

    return y, mean, rstd


def layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
):
    logger.debug("GEMS_KUNLUNXIN LAYER_NORM_BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    M = input.shape[0]
    N = input.numel() // M
    bc = _ln_bwd_col_size(N)
    br = triton.next_power_of_2(triton.cdiv(M, 12))
    need_mask = (M % br != 0) or (N % bc != 0)
    need_tail = N % bc != 0

    if output_mask[0]:
        in_grad = torch.empty_strided(
            input.size(), input.stride(), dtype=input.dtype, device=input.device
        )
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[(triton.cdiv(M, br), 1, 1)](
                grad_out,
                input,
                weight,
                mean,
                rstd,
                in_grad,
                M,
                N,
                BLOCK_ROW_SIZE=br,
                BLOCK_COL_SIZE=bc,
                NEED_MASK=need_mask,
                isCloseUnrollControl=need_mask,
                isCloseCoreTiling=need_mask,
                isCloseVectorization=True,
            )
    else:
        in_grad = None

    if output_mask[1] is False and output_mask[2] is False:
        return in_grad, None, None

    if output_mask[1]:
        weight_grad = torch.empty_strided(
            weight.size(), weight.stride(), dtype=weight.dtype, device=weight.device
        )
    else:
        weight_grad = None
    if output_mask[2]:
        bias_grad = torch.empty_strided(
            bias.size(), bias.stride(), dtype=bias.dtype, device=bias.device
        )
    else:
        bias_grad = None

    bm = _wb_bm_size(M)
    if bm >= M:
        with torch_device_fn.device(input.device):
            weight_bias_backward_1d_kernel[(triton.cdiv(N, bc), 1, 1)](
                grad_out,
                input,
                mean,
                rstd,
                weight_grad,
                bias_grad,
                M,
                N,
                BM=bm,
                C=bc,
                NEED_TAIL=need_tail,
                DIRECT=True,
                isCloseUnrollControl=True,
            )
    else:
        P = M // bm
        pw = (
            torch.empty_strided(
                (P, N), (N, 1), dtype=torch.float32, device=input.device
            )
            if weight_grad is not None
            else None
        )
        pb = (
            torch.empty_strided(
                (P, N), (N, 1), dtype=torch.float32, device=input.device
            )
            if bias_grad is not None
            else None
        )
        with torch_device_fn.device(input.device):
            weight_bias_backward_1d_kernel[(triton.cdiv(N, bc), P, 1)](
                grad_out,
                input,
                mean,
                rstd,
                pw,
                pb,
                M,
                N,
                BM=bm,
                C=bc,
                NEED_TAIL=need_tail,
                DIRECT=False,
                isCloseUnrollControl=True,
            )
            weight_bias_backward_finish_kernel[(triton.cdiv(N, bc), 1, 1)](
                pw,
                pb,
                weight_grad,
                bias_grad,
                P,
                N,
                C=bc,
                NEED_TAIL=need_tail,
                isCloseUnrollControl=True,
            )
    return in_grad, weight_grad, bias_grad
