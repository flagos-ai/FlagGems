"""Fast Hadamard Transform — Ascend NPU vendor specialization.

Ascend 910C has extremely powerful GEMM units (752 TFLOPS bf16), so using
torch.matmul with a cached Hadamard matrix is much faster than O(N log N)
butterfly decomposition for all practical shapes.

Strategy:
  - Power-of-2 dims: N-D torch.matmul(x, H_scaled) directly, no reshape overhead
  - Non-power-of-2 dims: pad + torch.mm fallback
  - Triton butterfly kernels retained as fallback for non-power-of-2 dims
    (currently unused since matmul path handles padding too)

Based on optimize-hadamard-transform-ascend v11.
"""

import logging
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from scipy.linalg import hadamard

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

MAX_GRID = 65535

# Hadamard matrix cache: (dim_padded, dtype, device, scale) -> scaled matrix
_hadamard_matrix_cache = {}


def _get_scaled_hadamard_matrix(dim_padded, dtype, device, scale):
    """Get or create cached Hadamard matrix with scale fused in."""
    key = (dim_padded, dtype, device, scale)
    cached = _hadamard_matrix_cache.get(key)
    if cached is not None:
        return cached
    H = torch.tensor(hadamard(dim_padded, dtype=float), dtype=dtype, device=device)
    if scale != 1.0:
        H = H * scale
    H = H.contiguous()
    _hadamard_matrix_cache[key] = H
    return H


# ============================================================
# Fused 7-stage butterfly kernel (dim=128) — BF16 compute
# 3-segment rotating scratch (retained as fallback)
# ============================================================

_scratch_cache = {}
_MAX_CACHE_ENTRIES = 16


def _get_scratch(batch, dim_padded, scratch_dtype, device):
    """Get or allocate scratch buffer, with caching."""
    key = (batch, dim_padded, scratch_dtype, device.type, device.index)
    cached = _scratch_cache.get(key)
    if cached is not None:
        return cached
    scratch = torch.empty(3, batch, dim_padded, dtype=scratch_dtype, device=device)
    if len(_scratch_cache) >= _MAX_CACHE_ENTRIES:
        oldest_key = next(iter(_scratch_cache))
        del _scratch_cache[oldest_key]
    _scratch_cache[key] = scratch
    return scratch


@triton.jit
def _fht_fused_7stage_bf16(
    IN_ptr,
    SCRATCH_ptr,
    OUT_ptr,
    stride_row,
    stride_out_row,
    seg_stride,
    scale,
    N_ROWS,
    ROWS_PER_PROGRAM: tl.constexpr,
    DIM: tl.constexpr,
):
    """Fused FHT for dim=128 in bf16. Input/scratch/output all bf16.
    Scale applied in fp32 at the final stage for precision.
    """
    pid = tle.program_id(0)
    offsets = tl.arange(0, DIM)

    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            in_base = row_id * stride_row
            row_off = row_id * DIM
            out_base = row_id * stride_out_row

            b0_off = row_off
            b1_off = seg_stride + row_off
            b2_off = 2 * seg_stride + row_off

            # Stage 0: IN -> B0 (stride=1)
            x = tl.load(IN_ptr + in_base + offsets)
            p = tl.load(IN_ptr + in_base + (offsets ^ 1))
            r = tl.where((offsets & 1) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b0_off + offsets, r)

            # Stage 1: B0 -> B1 (stride=2)
            x = tl.load(SCRATCH_ptr + b0_off + offsets)
            p = tl.load(SCRATCH_ptr + b0_off + (offsets ^ 2))
            r = tl.where((offsets & 2) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b1_off + offsets, r)

            # Stage 2: B1 -> B2 (stride=4)
            x = tl.load(SCRATCH_ptr + b1_off + offsets)
            p = tl.load(SCRATCH_ptr + b1_off + (offsets ^ 4))
            r = tl.where((offsets & 4) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b2_off + offsets, r)

            # Stage 3: B2 -> B0 (stride=8)
            x = tl.load(SCRATCH_ptr + b2_off + offsets)
            p = tl.load(SCRATCH_ptr + b2_off + (offsets ^ 8))
            r = tl.where((offsets & 8) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b0_off + offsets, r)

            # Stage 4: B0 -> B1 (stride=16)
            x = tl.load(SCRATCH_ptr + b0_off + offsets)
            p = tl.load(SCRATCH_ptr + b0_off + (offsets ^ 16))
            r = tl.where((offsets & 16) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b1_off + offsets, r)

            # Stage 5: B1 -> B2 (stride=32)
            x = tl.load(SCRATCH_ptr + b1_off + offsets)
            p = tl.load(SCRATCH_ptr + b1_off + (offsets ^ 32))
            r = tl.where((offsets & 32) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b2_off + offsets, r)

            # Stage 6: B2 -> OUT (stride=64) + scale in fp32 + cast
            x = tl.load(SCRATCH_ptr + b2_off + offsets)
            p = tl.load(SCRATCH_ptr + b2_off + (offsets ^ 64))
            r = tl.where((offsets & 64) == 0, x + p, p - x)

            r_f32 = r.to(tl.float32) * scale
            tl.store(OUT_ptr + out_base + offsets, r_f32.to(tl.bfloat16))


# ============================================================
# Fused 7-stage butterfly kernel (dim=128) — FP32 compute
# ============================================================


@triton.jit
def _fht_fused_7stage(
    IN_ptr,
    SCRATCH_ptr,
    OUT_ptr,
    stride_row,
    stride_out_row,
    seg_stride,
    scale,
    N_ROWS,
    ROWS_PER_PROGRAM: tl.constexpr,
    DIM: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    """Fused FHT for dim=128 (7 butterfly stages) + scale + cast.
    SCRATCH_ptr points to a contiguous (3, batch, DIM) fp32 buffer.
    """
    pid = tle.program_id(0)
    offsets = tl.arange(0, DIM)

    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            in_base = row_id * stride_row
            row_off = row_id * DIM
            out_base = row_id * stride_out_row

            b0_off = row_off
            b1_off = seg_stride + row_off
            b2_off = 2 * seg_stride + row_off

            # Stage 0: IN -> B0 (stride=1)
            x = tl.load(IN_ptr + in_base + offsets)
            p = tl.load(IN_ptr + in_base + (offsets ^ 1))
            r = tl.where((offsets & 1) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b0_off + offsets, r)

            # Stage 1: B0 -> B1 (stride=2)
            x = tl.load(SCRATCH_ptr + b0_off + offsets)
            p = tl.load(SCRATCH_ptr + b0_off + (offsets ^ 2))
            r = tl.where((offsets & 2) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b1_off + offsets, r)

            # Stage 2: B1 -> B2 (stride=4)
            x = tl.load(SCRATCH_ptr + b1_off + offsets)
            p = tl.load(SCRATCH_ptr + b1_off + (offsets ^ 4))
            r = tl.where((offsets & 4) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b2_off + offsets, r)

            # Stage 3: B2 -> B0 (stride=8)
            x = tl.load(SCRATCH_ptr + b2_off + offsets)
            p = tl.load(SCRATCH_ptr + b2_off + (offsets ^ 8))
            r = tl.where((offsets & 8) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b0_off + offsets, r)

            # Stage 4: B0 -> B1 (stride=16)
            x = tl.load(SCRATCH_ptr + b0_off + offsets)
            p = tl.load(SCRATCH_ptr + b0_off + (offsets ^ 16))
            r = tl.where((offsets & 16) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b1_off + offsets, r)

            # Stage 5: B1 -> B2 (stride=32)
            x = tl.load(SCRATCH_ptr + b1_off + offsets)
            p = tl.load(SCRATCH_ptr + b1_off + (offsets ^ 32))
            r = tl.where((offsets & 32) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b2_off + offsets, r)

            # Stage 6: B2 -> OUT (stride=64) + fused scale + cast
            x = tl.load(SCRATCH_ptr + b2_off + offsets)
            p = tl.load(SCRATCH_ptr + b2_off + (offsets ^ 64))
            r = tl.where((offsets & 64) == 0, x + p, p - x)

            r = r * scale
            if OUTPUT_BF16:
                tl.store(OUT_ptr + out_base + offsets, r.to(tl.bfloat16))
            elif OUTPUT_FP16:
                tl.store(OUT_ptr + out_base + offsets, r.to(tl.float16))
            else:
                tl.store(OUT_ptr + out_base + offsets, r)


# ============================================================
# Generic fused butterfly kernel (any power-of-2 dim)
# ============================================================


@triton.jit
def _fht_fused_generic(
    IN_ptr,
    SCRATCH_ptr,
    OUT_ptr,
    stride_row,
    stride_out_row,
    seg_stride,
    scale,
    N_ROWS,
    ROWS_PER_PROGRAM: tl.constexpr,
    DIM: tl.constexpr,
    LOG_N: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    """Generic fused FHT for any power-of-2 dim.
    Uses 3 rotating scratch buffers: stage s writes to buffer (s % 3).
    """
    pid = tle.program_id(0)
    offsets = tl.arange(0, DIM)

    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            in_base = row_id * stride_row
            row_off = row_id * DIM
            out_base = row_id * stride_out_row

            for s in tl.static_range(LOG_N):
                stride_s: tl.constexpr = 1 << s
                is_upper = (offsets & stride_s) == 0

                if s == 0:
                    x = tl.load(IN_ptr + in_base + offsets)
                    p = tl.load(IN_ptr + in_base + (offsets ^ stride_s))
                else:
                    src_buf: tl.constexpr = (s - 1) % 3
                    src_off = src_buf * seg_stride + row_off
                    x = tl.load(SCRATCH_ptr + src_off + offsets)
                    p = tl.load(SCRATCH_ptr + src_off + (offsets ^ stride_s))

                r = tl.where(is_upper, x + p, p - x)

                if s == LOG_N - 1:
                    r = r * scale
                    if OUTPUT_BF16:
                        tl.store(OUT_ptr + out_base + offsets, r.to(tl.bfloat16))
                    elif OUTPUT_FP16:
                        tl.store(OUT_ptr + out_base + offsets, r.to(tl.float16))
                    else:
                        tl.store(OUT_ptr + out_base + offsets, r)
                else:
                    dst_buf: tl.constexpr = s % 3
                    dst_off = dst_buf * seg_stride + row_off
                    tl.store(SCRATCH_ptr + dst_off + offsets, r)


# ============================================================
# Triton butterfly forward (fallback for non-power-of-2 dims)
# ============================================================


def _hadamard_transform_triton(
    x_flat, batch, dim_padded, log_n, scale, input_dtype, orig_shape, dim, device
):
    """Triton butterfly FHT — fallback path."""
    use_bf16_kernel = input_dtype == torch.bfloat16 and log_n == 7

    if use_bf16_kernel:
        inp = x_flat if x_flat.is_contiguous() else x_flat.contiguous()
        scratch_dtype = torch.bfloat16
    else:
        if x_flat.dtype == torch.float32 and x_flat.is_contiguous():
            inp = x_flat
        else:
            inp = x_flat.float().contiguous()
        scratch_dtype = torch.float32

    scratch = _get_scratch(batch, dim_padded, scratch_dtype, device)
    out = torch.empty(batch, dim_padded, dtype=input_dtype, device=device)
    seg_stride = batch * dim_padded

    if batch <= 64:
        min_rpp = 1
    elif batch <= 1024:
        min_rpp = 2
    elif batch <= 16384:
        min_rpp = 4
    else:
        min_rpp = 16
    rows_per_program = max((batch + MAX_GRID - 1) // MAX_GRID, min_rpp)
    grid_size = (batch + rows_per_program - 1) // rows_per_program

    stride_row = dim_padded
    output_bf16 = input_dtype == torch.bfloat16
    output_fp16 = input_dtype == torch.float16

    with torch_device_fn.device(device):
        if use_bf16_kernel:
            _fht_fused_7stage_bf16[(grid_size,)](
                inp,
                scratch,
                out,
                stride_row,
                dim_padded,
                seg_stride,
                scale,
                N_ROWS=batch,
                ROWS_PER_PROGRAM=rows_per_program,
                DIM=dim_padded,
            )
        elif log_n == 7:
            _fht_fused_7stage[(grid_size,)](
                inp,
                scratch,
                out,
                stride_row,
                dim_padded,
                seg_stride,
                scale,
                N_ROWS=batch,
                ROWS_PER_PROGRAM=rows_per_program,
                DIM=dim_padded,
                OUTPUT_BF16=output_bf16,
                OUTPUT_FP16=output_fp16,
            )
        else:
            _fht_fused_generic[(grid_size,)](
                inp,
                scratch,
                out,
                stride_row,
                dim_padded,
                seg_stride,
                scale,
                N_ROWS=batch,
                ROWS_PER_PROGRAM=rows_per_program,
                DIM=dim_padded,
                LOG_N=log_n,
                OUTPUT_BF16=output_bf16,
                OUTPUT_FP16=output_fp16,
            )

    if dim != dim_padded:
        out = out[:, :dim].contiguous()
    return out.view(orig_shape)


# ============================================================
# Core forward — N-D torch.matmul for all shapes
# ============================================================


def _hadamard_transform_fwd(x, scale):
    """Core forward: torch.matmul with cached scaled Hadamard matrix.

    For power-of-2 dims, uses torch.matmul(x, H_scaled) directly on the
    original N-D tensor — avoids reshape/view overhead (~8 us on Ascend NPU)
    that dominates for small decode shapes where GEMM itself is only ~10 us.

    Falls back to Triton butterfly only for non-power-of-2 dims.
    """
    logger.debug("GEMS_ASCEND HADAMARD_TRANSFORM")
    dim = x.shape[-1]

    # Fast power-of-2 check
    is_pow2 = dim >= 2 and (dim & (dim - 1)) == 0

    if is_pow2:
        H_scaled = _get_scaled_hadamard_matrix(dim, x.dtype, x.device, scale)
        return torch.matmul(x, H_scaled)
    else:
        # Non-power-of-2: pad and use matmul
        orig_shape = x.shape
        input_dtype = x.dtype
        log_n = math.ceil(math.log2(dim))
        dim_padded = 1 << log_n
        x_flat = x.reshape(-1, dim)
        x_flat = F.pad(x_flat, (0, dim_padded - dim))
        H_scaled = _get_scaled_hadamard_matrix(dim_padded, x.dtype, x.device, scale)
        out = torch.mm(x_flat, H_scaled)
        if out.dtype != input_dtype:
            out = out.to(input_dtype)
        out = out[:, :dim].contiguous()
        return out.view(orig_shape)


# ============================================================
# Autograd wrapper
# ============================================================


class HadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(torch.tensor(scale))
        return _hadamard_transform_fwd(x, scale)

    @staticmethod
    def backward(ctx, grad_output):
        (scale_t,) = ctx.saved_tensors
        scale = scale_t.item()
        return _hadamard_transform_fwd(grad_output, scale), None


# ============================================================
# Public API
# ============================================================


def hadamard_transform(x, scale=1.0):
    """Fast Hadamard Transform (Ascend NPU specialization).

    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Uses cached Hadamard matrix + torch.matmul for maximum throughput
    on Ascend 910C GEMM units.
    """
    # Bypass autograd for inference
    if not x.requires_grad:
        return _hadamard_transform_fwd(x, scale)
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
