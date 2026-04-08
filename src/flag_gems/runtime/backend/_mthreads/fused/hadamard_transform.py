"""Fast Hadamard Transform — MThreads (MUSA) vendor specialization.

Hybrid strategy:
- Small/medium batches (<65536 rows): bf16 butterfly stages (halves memory BW),
  only upcast to fp32 in final stage for scale precision
- Large batches (>=65536 rows): matmul via F.linear with cached Hadamard matrix

Known limitation: triton-mtgpu cannot launch triton kernels inside autograd
backward ("invalid device context"), so backward uses F.linear matmul fallback.

Based on optimize-hadamard-transform-musa v6.
"""

import logging
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from scipy.linalg import hadamard as scipy_hadamard

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

MAX_GRID = 65535


@triton.jit
def _butterfly_stage_native(
    IN_ptr,
    OUT_ptr,
    stride_row,
    N_ROWS,
    ROWS_PER_PROGRAM: tl.constexpr,
    STRIDE_S: tl.constexpr,
    DIM: tl.constexpr,
):
    """Butterfly stage keeping data in native dtype (no fp32 upcast)."""
    pid = tle.program_id(0)
    col = tl.arange(0, DIM)
    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            base = row_id * stride_row
            x = tl.load(IN_ptr + base + col)
            p = tl.load(IN_ptr + base + (col ^ STRIDE_S))
            r = tl.where((col & STRIDE_S) == 0, x + p, p - x)
            tl.store(OUT_ptr + base + col, r)


@triton.jit
def _butterfly_final_scale(
    IN_ptr,
    OUT_ptr,
    stride_in_row,
    stride_out_row,
    scale,
    N_ROWS,
    ROWS_PER_PROGRAM: tl.constexpr,
    STRIDE_S: tl.constexpr,
    DIM: tl.constexpr,
):
    """Final butterfly stage: upcast to fp32 for scale, then cast back."""
    pid = tle.program_id(0)
    col = tl.arange(0, DIM)
    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            base_in = row_id * stride_in_row
            x = tl.load(IN_ptr + base_in + col).to(tl.float32)
            p = tl.load(IN_ptr + base_in + (col ^ STRIDE_S)).to(tl.float32)
            r = tl.where((col & STRIDE_S) == 0, x + p, p - x)
            r = r * scale
            base_out = row_id * stride_out_row
            tl.store(OUT_ptr + base_out + col, r)


# ============================================================
# Hadamard matrix cache (for matmul path)
# ============================================================

_hadamard_matrix_cache = {}


def _get_hadamard_matrix(dim_padded, dtype, device):
    key = (dim_padded, str(dtype), str(device))
    cached = _hadamard_matrix_cache.get(key)
    if cached is not None:
        return cached
    H = torch.tensor(
        scipy_hadamard(dim_padded, dtype=float), dtype=dtype, device=device
    )
    _hadamard_matrix_cache[key] = H
    return H


# ============================================================
# Forward implementation
# ============================================================


def _hadamard_transform_fwd(x, scale):
    logger.debug("GEMS_MTHREADS HADAMARD_TRANSFORM")
    orig_shape = x.shape
    dim = x.shape[-1]
    input_dtype = x.dtype

    log_n = math.ceil(math.log2(dim))
    dim_padded = 1 << log_n

    x_flat = x.reshape(-1, dim)
    batch = x_flat.shape[0]

    if dim != dim_padded:
        x_flat = F.pad(x_flat, (0, dim_padded - dim))

    # Large batches: matmul with tensor cores
    if batch >= 65536:
        H = _get_hadamard_matrix(dim_padded, input_dtype, x.device)
        out = F.linear(x_flat.to(input_dtype), H) * scale
        if dim != dim_padded:
            out = out[:, :dim]
        return out.reshape(orig_shape)

    # Keep data in native dtype for butterfly stages (halves BW for bf16/fp16)
    # Only upcast to fp32 in the final stage for precise scaling
    if input_dtype in (torch.bfloat16, torch.float16):
        buf_a = x_flat.contiguous()
        buf_b = torch.empty_like(buf_a)
    else:
        # fp32 input: no savings, use fp32 throughout
        buf_a = x_flat.to(torch.float32).contiguous()
        buf_b = torch.empty_like(buf_a)

    stride_row = dim_padded

    rows_per_prog = max(1, (batch + MAX_GRID - 1) // MAX_GRID)
    grid_size = (batch + rows_per_prog - 1) // rows_per_prog
    nw = 4

    src, dst = buf_a, buf_b
    with torch_device_fn.device(x.device):
        for s in range(log_n - 1):
            _butterfly_stage_native[(grid_size,)](
                src,
                dst,
                stride_row,
                batch,
                ROWS_PER_PROGRAM=rows_per_prog,
                STRIDE_S=(1 << s),
                DIM=dim_padded,
                num_warps=nw,
            )
            src, dst = dst, src

        # Final stage: upcast to fp32 for scale, output in input_dtype
        out = torch.empty(batch, dim_padded, dtype=input_dtype, device=x.device)
        _butterfly_final_scale[(grid_size,)](
            src,
            out,
            stride_row,
            dim_padded,
            scale,
            batch,
            ROWS_PER_PROGRAM=rows_per_prog,
            STRIDE_S=(1 << (log_n - 1)),
            DIM=dim_padded,
            num_warps=nw,
        )

    if dim != dim_padded:
        out = out[:, :dim]
    return out.reshape(orig_shape)


# ============================================================
# Matmul fallback (used for backward due to triton-mtgpu limitation)
# ============================================================


def _hadamard_transform_matmul(x, scale):
    orig_shape = x.shape
    dim = x.shape[-1]
    input_dtype = x.dtype
    log_n = math.ceil(math.log2(dim))
    dim_padded = 1 << log_n
    x_flat = x.reshape(-1, dim)
    if dim != dim_padded:
        x_flat = F.pad(x_flat, (0, dim_padded - dim))
    H = _get_hadamard_matrix(dim_padded, x_flat.dtype, x_flat.device)
    out = F.linear(x_flat, H) * scale
    if dim != dim_padded:
        out = out[:, :dim]
    return out.reshape(orig_shape).to(input_dtype)


# ============================================================
# Autograd wrapper
# ============================================================


class HadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return _hadamard_transform_fwd(x, scale)

    @staticmethod
    def backward(ctx, grad_output):
        # triton-mtgpu cannot launch kernels in backward, use matmul fallback
        return _hadamard_transform_matmul(grad_output, ctx.scale), None


# ============================================================
# Public API
# ============================================================


def hadamard_transform(x, scale=1.0):
    """Fast Hadamard Transform (MThreads/MUSA specialization).

    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is
    the next power of 2.
    """
    return HadamardTransformFn.apply(x, scale)


# ============================================================
# XXN variants (non-power-of-2 dims)
# ============================================================


def hadamard_transform_12N(x, scale=1.0):
    """Hadamard transform for dim = 12 * 2^k."""
    return hadamard_transform(x, scale)


def hadamard_transform_20N(x, scale=1.0):
    """Hadamard transform for dim = 20 * 2^k."""
    return hadamard_transform(x, scale)


def hadamard_transform_28N(x, scale=1.0):
    """Hadamard transform for dim = 28 * 2^k."""
    return hadamard_transform(x, scale)


def hadamard_transform_40N(x, scale=1.0):
    """Hadamard transform for dim = 40 * 2^k."""
    return hadamard_transform(x, scale)
