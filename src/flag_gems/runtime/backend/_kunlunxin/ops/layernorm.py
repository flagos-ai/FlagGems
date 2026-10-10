import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---- flat_launcher: bind a raw kernel once, replay flat (skip ~20us of the
# per-call JITFunction.run python). These backward kernels are host/launch-bound
# on the tiny benchmark shapes, so this is the biggest single win. See
# mySkill/kernelOpt/flatLaunchSkill.md.
_FLAT_MISS = object()
_FLAT = _FLAT_MISS


def _flat_launchers():
    global _FLAT
    if _FLAT is _FLAT_MISS:
        try:
            from triton.runtime import driver

            _FLAT = getattr(driver.active, "flat_launchers", None)
        except Exception:
            _FLAT = None
    return _FLAT


def _flat_call(kernel, grid, key, compile_fn, operands):
    """Bind `kernel` once (via `compile_fn`, which must launch it and return the
    CompiledKernel) then replay flat with `operands` (data_ptr ints / scalars in
    signature order, constexprs excluded)."""
    launchers = _flat_launchers()
    if launchers is None:
        compile_fn()
        return
    launch, stream = launchers.acquire(kernel, key)
    if launch is None:
        compiled = compile_fn()
        launchers.bind(kernel, key, compiled, grid)
        return
    launch(stream, *operands)


# ---- SIMD dx (grad_input) via tle.raw (xpu3) --------------------------------
# Hand-written float32x16 SIMD payload for the small-M / mid-large-N dx path,
# where the triton dx_row kernel is host+convert bound. One launch, grid == M
# rows; each row's N-reduction is split across the 64 cores with SIMD vectors
# (vload2_lm converts fp16/bf16 -> f32 in the load). f32/f16/bf16 all supported
# (fp16 needs 64-byte-aligned LM buffers), gated to fp32 stats.
# Any import failure leaves _HAS_LN_DX_RAW False and the triton kernels run.
_HAS_LN_DX_RAW = False
_LN_DX_RAW_KERNELS = {}
try:
    import os as _os

    import triton.experimental.tle as _tle

    # Precompiled device object: stub names must equal the entry symbols in the .o
    # and the signatures must match the C++ ABI.
    _LN_RAW_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _LN_DX_OBJ = _os.path.join(
        _os.path.dirname(_LN_RAW_DIR), "payload", "obj", "layernorm_backward_dx_raw.o"
    )

    @_tle.raw.dialect("xpu3", object=_LN_DX_OBJ, arch=3)
    def ln_dx_f32(dy, x, w, mean, rstd, dx, M, N, has_weight): ...

    @_tle.raw.dialect("xpu3", object=_LN_DX_OBJ, arch=3)
    def ln_dx_f16(dy, x, w, mean, rstd, dx, M, N, has_weight): ...

    @_tle.raw.dialect("xpu3", object=_LN_DX_OBJ, arch=3)
    def ln_dx_bf16(dy, x, w, mean, rstd, dx, M, N, has_weight): ...

    _LN_DX_DNS = ["M", "N", "has_weight"]
    _LN_DX_DNA = ["Dy", "X", "W", "Mean", "Rstd", "Dx"]

    @triton.jit(do_not_specialize=_LN_DX_DNS, do_not_specialize_on_alignment=_LN_DX_DNA)
    def _ln_dx_raw_kernel_f32(Dy, X, W, Mean, Rstd, Dx, M, N, has_weight):
        _tle.raw.call(ln_dx_f32, (Dy, X, W, Mean, Rstd, Dx, M, N, has_weight))

    @triton.jit(do_not_specialize=_LN_DX_DNS, do_not_specialize_on_alignment=_LN_DX_DNA)
    def _ln_dx_raw_kernel_f16(Dy, X, W, Mean, Rstd, Dx, M, N, has_weight):
        _tle.raw.call(ln_dx_f16, (Dy, X, W, Mean, Rstd, Dx, M, N, has_weight))

    @triton.jit(do_not_specialize=_LN_DX_DNS, do_not_specialize_on_alignment=_LN_DX_DNA)
    def _ln_dx_raw_kernel_bf16(Dy, X, W, Mean, Rstd, Dx, M, N, has_weight):
        _tle.raw.call(ln_dx_bf16, (Dy, X, W, Mean, Rstd, Dx, M, N, has_weight))

    _LN_DX_RAW_KERNELS = {
        torch.float32: _ln_dx_raw_kernel_f32,
        torch.float16: _ln_dx_raw_kernel_f16,
        torch.bfloat16: _ln_dx_raw_kernel_bf16,
    }

    # combined dx + dW/dB in one launch (heterogeneous clusters). grid ==
    # M + wb_blocks; wb_cols == 64 cores * 32 = 2048 columns per WB cluster.
    _LN_DXWB_WB_COLS = 2048

    @_tle.raw.dialect("xpu3", object=_LN_DX_OBJ, arch=3)
    def ln_dxwb_f32(dy, x, w, mean, rstd, dx, dw, db, M, N, wb_cols): ...

    @_tle.raw.dialect("xpu3", object=_LN_DX_OBJ, arch=3)
    def ln_dxwb_f16(dy, x, w, mean, rstd, dx, dw, db, M, N, wb_cols): ...

    @_tle.raw.dialect("xpu3", object=_LN_DX_OBJ, arch=3)
    def ln_dxwb_bf16(dy, x, w, mean, rstd, dx, dw, db, M, N, wb_cols): ...

    _LN_DXWB_DNS = ["M", "N", "wb_cols"]
    _LN_DXWB_DNA = ["Dy", "X", "W", "Mean", "Rstd", "Dx", "Dw", "Db"]

    @triton.jit(do_not_specialize=_LN_DXWB_DNS, do_not_specialize_on_alignment=_LN_DXWB_DNA)
    def _ln_dxwb_raw_kernel_f32(Dy, X, W, Mean, Rstd, Dx, Dw, Db, M, N, wb_cols):
        _tle.raw.call(ln_dxwb_f32, (Dy, X, W, Mean, Rstd, Dx, Dw, Db, M, N, wb_cols))

    @triton.jit(do_not_specialize=_LN_DXWB_DNS, do_not_specialize_on_alignment=_LN_DXWB_DNA)
    def _ln_dxwb_raw_kernel_f16(Dy, X, W, Mean, Rstd, Dx, Dw, Db, M, N, wb_cols):
        _tle.raw.call(ln_dxwb_f16, (Dy, X, W, Mean, Rstd, Dx, Dw, Db, M, N, wb_cols))

    @triton.jit(do_not_specialize=_LN_DXWB_DNS, do_not_specialize_on_alignment=_LN_DXWB_DNA)
    def _ln_dxwb_raw_kernel_bf16(Dy, X, W, Mean, Rstd, Dx, Dw, Db, M, N, wb_cols):
        _tle.raw.call(ln_dxwb_bf16, (Dy, X, W, Mean, Rstd, Dx, Dw, Db, M, N, wb_cols))

    _LN_DXWB_RAW_KERNELS = {
        torch.float32: _ln_dxwb_raw_kernel_f32,
        torch.float16: _ln_dxwb_raw_kernel_f16,
        torch.bfloat16: _ln_dxwb_raw_kernel_bf16,
    }
    _HAS_LN_DX_RAW = True
except Exception:  # pragma: no cover - environment without tle.raw
    _HAS_LN_DX_RAW = False
    _LN_DX_RAW_KERNELS = {}
    _LN_DXWB_RAW_KERNELS = {}


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
            mask = row_mask & col_mask
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
            mask = row_mask & col_mask
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


@triton.jit(
    do_not_specialize=["M", "N"],
    do_not_specialize_on_alignment=["dY", "X", "Mean", "Rstd", "OutW", "OutB"],
)
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
        # NEED_TAIL => C does not divide N, so n0+cols runs past N on the last
        # column program. The accumulator loads were already masked; the store
        # must be too, otherwise it writes C-wide past dW/dB (out-of-bounds heap
        # write -- only invisible because the valid region still checks out).
        if NEED_TAIL:
            cmask = n0 + cols < N
            if OutW is not None:
                tl.store(OutW + n0 + cols, accW, mask=cmask)
            if OutB is not None:
                tl.store(OutB + n0 + cols, accB, mask=cmask)
        else:
            if OutW is not None:
                tl.store(OutW + n0 + cols, accW)
            if OutB is not None:
                tl.store(OutB + n0 + cols, accB)
    else:
        # DIRECT=False writes per-partial-row buffers pw/pb[P,N]. With NEED_TAIL
        # (C does not divide N), n0+cols runs past N on the last column program,
        # so an unmasked store spills C-wide past the row into the next partial
        # row / past the buffer -- an OOB write that non-deterministically
        # corrupts adjacent allocations (e.g. in_grad). Mask it like the loads.
        if NEED_TAIL:
            cmask = n0 + cols < N
            if OutW is not None:
                tl.store(OutW + mi * N + n0 + cols, accW, mask=cmask)
            if OutB is not None:
                tl.store(OutB + mi * N + n0 + cols, accB, mask=cmask)
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


@triton.jit(
    do_not_specialize_on_alignment=["dY", "X", "W", "Mean", "Rstd", "dX"],
)
def layer_norm_backward_dx_row_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    HAS_W: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # One triton program == one row. All parallelism is on the M(row) axis, so
    # this path is only for small M (grid == M). The N-reduction (dx_2, dx_3) is
    # done per row across the 64 cores of the cluster, then dx is recomputed in a
    # second pass. Beats the 2D-tile kernel for small-M/large-N (avoids the
    # masked 2D CoreTiling scatter path).
    pid = ext.program_id(0)
    dY += pid * N
    X += pid * N
    dX += pid * N
    mean = tl.load(Mean + pid).to(tl.float32)
    rstd = tl.load(Rstd + pid).to(tl.float32)

    dx_2 = tl.zeros([TILE_N], dtype=tl.float32)
    dx_3 = tl.zeros([TILE_N], dtype=tl.float32)
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        if NEED_MASK:
            m = cols < N
            dy = tl.load(dY + cols, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X + cols, mask=m, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32) if HAS_W else 1.0
            x_hat = tl.where(m, (x - mean) * rstd, 0.0)
        else:
            dy = tl.load(dY + cols).to(tl.float32)
            x = tl.load(X + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32) if HAS_W else 1.0
            x_hat = (x - mean) * rstd
        dx_hat = dy * w
        dx_2 += dx_hat
        dx_3 += dx_hat * x_hat
    s2 = tl.sum(dx_2, axis=0)
    s3 = tl.sum(dx_3, axis=0)

    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        if NEED_MASK:
            m = cols < N
            dy = tl.load(dY + cols, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X + cols, mask=m, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32) if HAS_W else 1.0
            x_hat = (x - mean) * rstd
            dx_hat = dy * w
            dx = rstd * (dx_hat - (s2 + x_hat * s3) / N)
            tl.store(dX + cols, dx.to(dX.dtype.element_ty), mask=m)
        else:
            dy = tl.load(dY + cols).to(tl.float32)
            x = tl.load(X + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32) if HAS_W else 1.0
            x_hat = (x - mean) * rstd
            dx_hat = dy * w
            dx = rstd * (dx_hat - (s2 + x_hat * s3) / N)
            tl.store(dX + cols, dx.to(dX.dtype.element_ty))


# small-M dx routing threshold: grid == M, keep it to a few waves.
_DX_ROW_M_MAX = 64
# only apply the small-M dx_row / wide-C wb tuning for moderate N; large N keeps
# the original (proven-stable) paths.
_LN_BWD_N_MAX = 16384

# SIMD dx pays off only past a per-dtype N crossover (below it, the triton dx_row
# is already fast and the SIMD launch + cross-core sync overhead loses). Measured
# on KL3: bf16 crosses ~4096, f32 only past ~8192 (its triton convert tax is
# smaller). N must also stay <= _LN_BWD_N_MAX so ceil(N/64) fits the LM tile.
_LN_DX_SIMD_MIN_N = {torch.float32: 8192, torch.float16: 4096, torch.bfloat16: 4096}
# Combined dx+WB single launch: try from 2048 up (the launch-saving matters most
# at small N where each kernel is otherwise launch-bound).
_LN_DXWB_MIN_N = 2048


def _wb_col_size(N):
    import builtins

    # largest pow2 with grid(cdiv(N,C)) >= 4, floor 512, cap 2048. For small M
    # the wb reduction is over M(rows), so N is the only parallel axis; wide C
    # (good per-core vectorization) beats many tiny column programs on KL3.
    c = triton.next_power_of_2(N)
    while c > 512 and triton.cdiv(N, c) < 4:
        c //= 2
    return builtins.min(c, 2048)


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


def _ln_dx_triton(grad_out, input, weight, mean, rstd, in_grad, M, N, br, bc, need_mask):
    """Original triton dx path: dx_row (grid==M, mask-free pow2 tile) for small M,
    else the 2D-tile kernel. Fallback for shapes the SIMD payload doesn't cover."""
    import builtins

    # dx_row (one program per row) is only used when M is small AND we can pick a
    # power-of-2 tile that DIVIDES N (mask-free). Masked 1D stores in this single-
    # program path proved unreliable on KL3 (device-state contamination), so odd /
    # non-pow2-friendly N fall back to the original 2D-tile kernel.
    dtype_cap = 4096 if input.dtype == torch.float32 else 1024
    pow2_div = N & (-N)  # largest power of 2 dividing N
    tile_n = builtins.min(dtype_cap, pow2_div)
    use_dx_row = (M <= _DX_ROW_M_MAX) and (tile_n >= 256) and (N <= _LN_BWD_N_MAX)
    if use_dx_row:
        has_w = weight is not None
        grid = (M, 1, 1)

        def _dx_compile():
            return layer_norm_backward_dx_row_kernel[grid](
                grad_out,
                input,
                weight,
                mean,
                rstd,
                in_grad,
                N,
                TILE_N=tile_n,
                HAS_W=has_w,
                NEED_MASK=False,
                isCloseUnrollControl=True,
            )
        with torch_device_fn.device(input.device):
            if has_w:
                key = (input.dtype, N, tile_n, grid[0])
                operands = (
                    grad_out.data_ptr(),
                    input.data_ptr(),
                    weight.data_ptr(),
                    mean.data_ptr(),
                    rstd.data_ptr(),
                    in_grad.data_ptr(),
                )
                _flat_call(
                    layer_norm_backward_dx_row_kernel, grid, key, _dx_compile, operands
                )
            else:
                _dx_compile()
    else:
        # The masked 2D kernel is non-deterministic on KL3 (rare dx spikes from
        # unreliable masked load/store on partially-OOB row/col blocks). When M
        # and N have large-enough power-of-2 divisors, pick block sizes that
        # DIVIDE them so NEED_MASK==False (mask-free -> deterministic). Falls
        # back to the masked path only when no decent divisor exists (e.g. odd N).
        import builtins

        def _pow2_div(n, cap):
            d = n & (-n)  # largest power of 2 dividing n
            return builtins.min(d, cap)

        br2 = _pow2_div(M, 32)
        bc2 = _pow2_div(N, bc)
        if bc2 >= 4 and (M % br2 == 0) and (N % bc2 == 0):
            eff_br, eff_bc, eff_mask = br2, bc2, False
        else:
            eff_br, eff_bc, eff_mask = br, bc, need_mask
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[(triton.cdiv(M, eff_br), 1, 1)](
                grad_out,
                input,
                weight,
                mean,
                rstd,
                in_grad,
                M,
                N,
                BLOCK_ROW_SIZE=eff_br,
                BLOCK_COL_SIZE=eff_bc,
                NEED_MASK=eff_mask,
                isCloseUnrollControl=eff_mask,
                isCloseCoreTiling=eff_mask,
                isCloseVectorization=True,
            )


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

    # These tiny shapes are host-bound: every redundant dispatched op on the
    # wrapper is pure overhead. Only call .contiguous() when actually needed.
    def _cont(t):
        return t if (t is None or t.is_contiguous()) else t.contiguous()

    grad_out = _cont(grad_out)
    input = _cont(input)
    mean = _cont(mean)
    rstd = _cont(rstd)
    weight = _cont(weight)
    bias = _cont(bias)

    # N is the product of normalized_shape (the reduced feature dims); M is
    # everything else. Using input.shape[0] is only correct when there is exactly
    # one leading (batch) dim -- for >1 leading dim it mis-splits M/N. Match the
    # forward's convention (N = prod(normalized_shape)).
    N = math.prod(normalized_shape)
    M = input.numel() // N
    bc = _ln_bwd_col_size(N)
    br = triton.next_power_of_2(triton.cdiv(M, 12))
    need_mask = (M % br != 0) or (N % bc != 0)
    need_tail = N % bc != 0

    # ---- combined dx + dW/dB in ONE launch (heterogeneous clusters) --------
    # When all three grads are needed, fold the dx launch and the wb launch into
    # a single raw launch (grid == M + wb_blocks). Wins when it fits one cluster
    # wave (M+wb_blocks <= 12) or N is large enough that the SIMD dx beats triton
    # anyway (>=4096). At small N + large M (2 waves, tiny per-item work) the
    # two-launch path is faster, so exclude that corner.
    _dxwb_blocks = (N + _LN_DXWB_WB_COLS - 1) // _LN_DXWB_WB_COLS
    if (
        output_mask[0] and output_mask[1] and output_mask[2]
        and _HAS_LN_DX_RAW
        and input.dtype in _LN_DXWB_RAW_KERNELS
        and weight is not None and bias is not None
        and mean.dtype == torch.float32 and rstd.dtype == torch.float32
        and M <= _DX_ROW_M_MAX
        and _LN_DXWB_MIN_N <= N <= _LN_BWD_N_MAX
        and (M + _dxwb_blocks <= 12 or N >= 4096)
    ):
        in_grad = torch.empty_strided(
            input.size(), input.stride(), dtype=input.dtype, device=input.device
        )
        weight_grad = torch.empty_strided(
            weight.size(), weight.stride(), dtype=weight.dtype, device=weight.device
        )
        bias_grad = torch.empty_strided(
            bias.size(), bias.stride(), dtype=bias.dtype, device=bias.device
        )
        raw_kernel = _LN_DXWB_RAW_KERNELS[input.dtype]
        wb_blocks = _dxwb_blocks
        grid = (M + wb_blocks, 1, 1)

        def _dxwb_compile():
            return raw_kernel[grid](
                grad_out, input, weight, mean, rstd, in_grad, weight_grad,
                bias_grad, M, N, _LN_DXWB_WB_COLS,
            )
        with torch_device_fn.device(input.device):
            key = (input.dtype, M, N)
            operands = (
                grad_out.data_ptr(), input.data_ptr(), weight.data_ptr(),
                mean.data_ptr(), rstd.data_ptr(), in_grad.data_ptr(),
                weight_grad.data_ptr(), bias_grad.data_ptr(),
                M, N, _LN_DXWB_WB_COLS,
            )
            _flat_call(raw_kernel, grid, key, _dxwb_compile, operands)
        return in_grad, weight_grad, bias_grad

    if output_mask[0]:
        in_grad = torch.empty_strided(
            input.size(), input.stride(), dtype=input.dtype, device=input.device
        )
        import builtins

        # SIMD dx (float32x16 tle.raw payload): one launch, grid == M. Only when
        # stats are fp32 (the kernel reads mean/rstd as float*) and dtype is
        # f32/bf16 (fp16 conversion path unresolved). N <= _LN_BWD_N_MAX keeps
        # per-core columns (ceil(N/64)) within the LM tile. Weight required (the
        # payload always takes a weight pointer). Beats the triton dx_row kernel
        # by folding the fp16/bf16 -> f32 convert into the SIMD load.
        use_simd = (
            _HAS_LN_DX_RAW
            and input.dtype in _LN_DX_RAW_KERNELS
            and weight is not None
            and mean.dtype == torch.float32
            and rstd.dtype == torch.float32
            and M <= _DX_ROW_M_MAX
            and N <= _LN_BWD_N_MAX
            and N >= _LN_DX_SIMD_MIN_N[input.dtype]
        )
        if use_simd:
            raw_kernel = _LN_DX_RAW_KERNELS[input.dtype]
            grid = (M, 1, 1)

            def _simd_compile():
                return raw_kernel[grid](
                    grad_out,
                    input,
                    weight,
                    mean,
                    rstd,
                    in_grad,
                    M,
                    N,
                    1,
                )
            with torch_device_fn.device(input.device):
                key = (input.dtype, M, N)
                operands = (
                    grad_out.data_ptr(),
                    input.data_ptr(),
                    weight.data_ptr(),
                    mean.data_ptr(),
                    rstd.data_ptr(),
                    in_grad.data_ptr(),
                    M,
                    N,
                    1,
                )
                _flat_call(raw_kernel, grid, key, _simd_compile, operands)
        else:
            _ln_dx_triton(
                grad_out, input, weight, mean, rstd, in_grad, M, N, br, bc,
                need_mask,
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
        # small/moderate M: reduce over M(rows), parallelize over N columns.
        # Use a wb-specific column block (wide C, grid>=4) for moderate N; large
        # N keeps the original bc (proven-stable).
        if N <= _LN_BWD_N_MAX:
            wb_c = _wb_col_size(N)
        else:
            wb_c = bc
        wb_need_tail = (N % wb_c) != 0
        wb_grid = (triton.cdiv(N, wb_c), 1, 1)

        def _wb_compile():
            return weight_bias_backward_1d_kernel[wb_grid](
                grad_out,
                input,
                mean,
                rstd,
                weight_grad,
                bias_grad,
                M,
                N,
                BM=bm,
                C=wb_c,
                NEED_TAIL=wb_need_tail,
                DIRECT=True,
                isCloseUnrollControl=True,
            )
        with torch_device_fn.device(input.device):
            if weight_grad is not None and bias_grad is not None:
                key = (input.dtype, wb_c, bm, wb_need_tail, wb_grid[0])
                operands = (
                    grad_out.data_ptr(),
                    input.data_ptr(),
                    mean.data_ptr(),
                    rstd.data_ptr(),
                    weight_grad.data_ptr(),
                    bias_grad.data_ptr(),
                    M,
                    N,
                )
                _flat_call(
                    weight_bias_backward_1d_kernel,
                    wb_grid,
                    key,
                    _wb_compile,
                    operands,
                )
            else:
                _wb_compile()
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
