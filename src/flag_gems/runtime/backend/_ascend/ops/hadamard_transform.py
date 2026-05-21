"""Fast Hadamard Transform in Triton (Ascend NPU).

v1: Single-kernel fused butterfly with chained buffers + fused scale/cast.
All 7 butterfly stages + scale + dtype cast in one kernel launch.
Uses unique buffer for each stage to avoid NPU stale-read issues.
Eliminates 7 kernel launch overheads and the separate scale/cast kernel from v0.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MAX_GRID = 65535


# ============================================================
# Fused 7-stage butterfly kernel (dim=128 specialized)
# Uses 6 scratch buffer segments (B0..B5) in a contiguous allocation.
# Chain: IN -> B0 -> B1 -> B2 -> B3 -> B4 -> B5 -> OUT
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

    SCRATCH_ptr points to a contiguous (6, batch, DIM) fp32 buffer.
    seg_stride = batch * DIM (distance between scratch segments).
    Chain: IN -> seg0 -> seg1 -> seg2 -> seg3 -> seg4 -> seg5 -> OUT
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, DIM)

    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            in_base = row_id * stride_row
            row_off = row_id * DIM  # offset within each scratch segment

            # Stage 0: IN -> B0 (stride=1)
            x = tl.load(IN_ptr + in_base + offsets)
            p = tl.load(IN_ptr + in_base + (offsets ^ 1))
            r = tl.where((offsets & 1) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + row_off + offsets, r)

            # Stage 1: B0 -> B1 (stride=2)
            b0_off = row_off
            b1_off = seg_stride + row_off
            x = tl.load(SCRATCH_ptr + b0_off + offsets)
            p = tl.load(SCRATCH_ptr + b0_off + (offsets ^ 2))
            r = tl.where((offsets & 2) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b1_off + offsets, r)

            # Stage 2: B1 -> B2 (stride=4)
            b2_off = 2 * seg_stride + row_off
            x = tl.load(SCRATCH_ptr + b1_off + offsets)
            p = tl.load(SCRATCH_ptr + b1_off + (offsets ^ 4))
            r = tl.where((offsets & 4) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b2_off + offsets, r)

            # Stage 3: B2 -> B3 (stride=8)
            b3_off = 3 * seg_stride + row_off
            x = tl.load(SCRATCH_ptr + b2_off + offsets)
            p = tl.load(SCRATCH_ptr + b2_off + (offsets ^ 8))
            r = tl.where((offsets & 8) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b3_off + offsets, r)

            # Stage 4: B3 -> B4 (stride=16)
            b4_off = 4 * seg_stride + row_off
            x = tl.load(SCRATCH_ptr + b3_off + offsets)
            p = tl.load(SCRATCH_ptr + b3_off + (offsets ^ 16))
            r = tl.where((offsets & 16) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b4_off + offsets, r)

            # Stage 5: B4 -> B5 (stride=32)
            b5_off = 5 * seg_stride + row_off
            x = tl.load(SCRATCH_ptr + b4_off + offsets)
            p = tl.load(SCRATCH_ptr + b4_off + (offsets ^ 32))
            r = tl.where((offsets & 32) == 0, x + p, p - x)
            tl.store(SCRATCH_ptr + b5_off + offsets, r)

            # Stage 6: B5 -> OUT (stride=64) + fused scale + cast
            x = tl.load(SCRATCH_ptr + b5_off + offsets)
            p = tl.load(SCRATCH_ptr + b5_off + (offsets ^ 64))
            r = tl.where((offsets & 64) == 0, x + p, p - x)

            r = r * scale
            out_base = row_id * stride_out_row
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

    Uses chained scratch buffer segments. Each stage reads from one
    segment and writes to the next, avoiding NPU stale-read issues.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, DIM)

    for row_idx in tl.static_range(ROWS_PER_PROGRAM):
        row_id = pid * ROWS_PER_PROGRAM + row_idx
        if row_id < N_ROWS:
            in_base = row_id * stride_row
            row_off = row_id * DIM

            for s in tl.static_range(LOG_N):
                stride_s: tl.constexpr = 1 << s
                is_upper = (offsets & stride_s) == 0

                if s == 0:
                    # Read from input
                    x = tl.load(IN_ptr + in_base + offsets)
                    p = tl.load(IN_ptr + in_base + (offsets ^ stride_s))
                else:
                    src_off = (s - 1) * seg_stride + row_off
                    x = tl.load(SCRATCH_ptr + src_off + offsets)
                    p = tl.load(SCRATCH_ptr + src_off + (offsets ^ stride_s))

                r = tl.where(is_upper, x + p, p - x)

                if s == LOG_N - 1:
                    r = r * scale
                    out_base = row_id * stride_out_row
                    if OUTPUT_BF16:
                        tl.store(OUT_ptr + out_base + offsets, r.to(tl.bfloat16))
                    elif OUTPUT_FP16:
                        tl.store(OUT_ptr + out_base + offsets, r.to(tl.float16))
                    else:
                        tl.store(OUT_ptr + out_base + offsets, r)
                else:
                    dst_off = s * seg_stride + row_off
                    tl.store(SCRATCH_ptr + dst_off + offsets, r)


# ============================================================
# Core forward
# ============================================================


def _hadamard_transform_fwd(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Core forward: handles reshape, padding, kernel launch."""
    assert x.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"Unsupported dtype {x.dtype}"

    orig_shape = x.shape
    dim = orig_shape[-1]
    input_dtype = x.dtype
    x_flat = x.reshape(-1, dim)
    batch = x_flat.shape[0]

    # Pad dim to next power of 2
    log_n = math.ceil(math.log2(max(dim, 2)))
    dim_padded = 1 << log_n
    if dim != dim_padded:
        x_flat = F.pad(x_flat, (0, dim_padded - dim))

    # Input buffer in fp32
    inp_fp32 = x_flat.float()

    # Scratch buffer: (log_n - 1) segments of (batch, dim_padded) in fp32
    # Stage s writes to segment s (0..log_n-2), last stage writes to output
    n_scratch = max(log_n - 1, 1)
    scratch = torch.empty(
        n_scratch, batch, dim_padded, dtype=torch.float32, device=x.device
    )
    seg_stride = batch * dim_padded

    # Grid calculation
    rows_per_program = max((batch + MAX_GRID - 1) // MAX_GRID, 1)
    grid_size = (batch + rows_per_program - 1) // rows_per_program

    stride_row = dim_padded  # contiguous

    # Output buffer
    out = torch.empty(batch, dim_padded, dtype=input_dtype, device=x.device)

    output_bf16 = input_dtype == torch.bfloat16
    output_fp16 = input_dtype == torch.float16

    # Use specialized 7-stage kernel for dim=128, generic for others
    if log_n == 7:
        _fht_fused_7stage[(grid_size,)](
            inp_fp32,
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
            inp_fp32,
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

    # Trim padding and restore shape
    if dim != dim_padded:
        out = out[:, :dim]
    return out.reshape(orig_shape)


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
    """Fast Hadamard Transform.

    x: (..., dim), device=npu, fp32/fp16/bf16
    out: (..., dim), same dtype

    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is
    the next power of 2.
    """
    return HadamardTransformFn.apply(x, scale)
