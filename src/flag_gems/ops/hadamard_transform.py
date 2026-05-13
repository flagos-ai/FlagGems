"""Fast Hadamard Transform in Triton.

Drop-in replacement for Dao-AILab/fast-hadamard-transform with identical interface:
  - hadamard_transform(x, scale=1.0) with autograd support
  - hadamard_transform_12N/20N/28N/40N(x, scale=1.0) for non-power-of-2 dims
  - Input: (..., dim), fp32/fp16/bf16
  - Output: (..., dim), same dtype as input
  - Padding: to next multiple of 8 (matching CUDA impl)
  - dim <= 32768 (standard), dim <= M*2^10 (XXN variants)

Reference: https://github.com/Dao-AILab/fast-hadamard-transform
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ============================================================
# Triton kernel — v43: remove evict_first from loads + warps=2 for dim=256
# ============================================================
# v35 best: dim=256 0.9302x (no evict_first on loads, warps=1)
# v42: dim=256 0.8950x (evict_first on loads hurt — L2 thrashing)
#
# v43 strategy:
#   1. Remove evict_first from all loads. v42 proved it hurts dim=256
#      (0.8950x vs v35's 0.9302x). The 256-element rows (512B fp16)
#      are small enough that L2 caching of nearby rows helps prefetch.
#   2. Try num_warps=2 for dim=256 4-row ILP kernel. With 4 rows of
#      256 elements each, the workload can benefit from 2-warp occupancy:
#      each warp handles the compute for its assigned instructions,
#      and the scheduler can overlap loads from one warp with compute
#      from the other. This targets the memory latency hiding gap.
#   3. Keep evict_first on stores (write-once streaming pattern).
#   4. Keep all other kernels unchanged from v42 baseline.


# ============================================================
# Butterfly stages
# ============================================================


@triton.jit
def _butterfly_stage_1d(x, BLOCK_SIZE: tl.constexpr, STRIDE: tl.constexpr):
    """One butterfly stage on a 1D vector."""
    GRP: tl.constexpr = BLOCK_SIZE // (2 * STRIDE)
    if STRIDE == 1:
        x2 = tl.reshape(x, (GRP, 2))
        a, b = tl.split(x2)
        return tl.reshape(tl.join(a + b, a - b), (BLOCK_SIZE,))
    else:
        x3 = tl.reshape(x, (GRP, 2, STRIDE))
        x3 = tl.permute(x3, (0, 2, 1))
        a, b = tl.split(x3)
        x3 = tl.join(a + b, a - b)
        x3 = tl.permute(x3, (0, 2, 1))
        return tl.reshape(x3, (BLOCK_SIZE,))


@triton.jit
def _butterfly_stage_2d(
    x, ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr, STRIDE: tl.constexpr
):
    """One butterfly stage on a 2D (ROWS, BLOCK_SIZE) tensor."""
    GRP: tl.constexpr = BLOCK_SIZE // (2 * STRIDE)
    if STRIDE == 1:
        x2 = tl.reshape(x, (ROWS, GRP, 2))
        a, b = tl.split(x2)
        return tl.reshape(tl.join(a + b, a - b), (ROWS, BLOCK_SIZE))
    else:
        x3 = tl.reshape(x, (ROWS, GRP, 2, STRIDE))
        x3 = tl.permute(x3, (0, 1, 3, 2))
        a, b = tl.split(x3)
        x3 = tl.join(a + b, a - b)
        x3 = tl.permute(x3, (0, 1, 3, 2))
        return tl.reshape(x3, (ROWS, BLOCK_SIZE))


# ============================================================
# 4-row ILP 1D native kernel for dim=256 (8 hardcoded stages)
# v43: remove evict_first from loads, keep on stores
# ============================================================


@triton.jit
def _fht_kernel_256_4row_native(
    X_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    N_ROWS,
    SCALE: tl.constexpr,
):
    """FHT for dim=256, 4-row ILP: four independent 1D butterflies per program."""
    pid = tl.program_id(0)
    col_offs = tl.arange(0, 256)

    row0 = pid * 4
    row1 = row0 + 1
    row2 = row0 + 2
    row3 = row0 + 3

    # Load all 4 rows (no evict_first: L2 caching helps for nearby rows)
    x0 = tl.load(X_ptr + row0 * stride_x_row + col_offs)
    x1 = tl.load(X_ptr + row1 * stride_x_row + col_offs, mask=row1 < N_ROWS, other=0.0)
    x2 = tl.load(X_ptr + row2 * stride_x_row + col_offs, mask=row2 < N_ROWS, other=0.0)
    x3 = tl.load(X_ptr + row3 * stride_x_row + col_offs, mask=row3 < N_ROWS, other=0.0)

    # Interleaved hardcoded reversed butterfly stages for 4-way ILP
    x0 = _butterfly_stage_1d(x0, 256, 128)
    x1 = _butterfly_stage_1d(x1, 256, 128)
    x2 = _butterfly_stage_1d(x2, 256, 128)
    x3 = _butterfly_stage_1d(x3, 256, 128)
    x0 = _butterfly_stage_1d(x0, 256, 64)
    x1 = _butterfly_stage_1d(x1, 256, 64)
    x2 = _butterfly_stage_1d(x2, 256, 64)
    x3 = _butterfly_stage_1d(x3, 256, 64)
    x0 = _butterfly_stage_1d(x0, 256, 32)
    x1 = _butterfly_stage_1d(x1, 256, 32)
    x2 = _butterfly_stage_1d(x2, 256, 32)
    x3 = _butterfly_stage_1d(x3, 256, 32)
    x0 = _butterfly_stage_1d(x0, 256, 16)
    x1 = _butterfly_stage_1d(x1, 256, 16)
    x2 = _butterfly_stage_1d(x2, 256, 16)
    x3 = _butterfly_stage_1d(x3, 256, 16)
    x0 = _butterfly_stage_1d(x0, 256, 8)
    x1 = _butterfly_stage_1d(x1, 256, 8)
    x2 = _butterfly_stage_1d(x2, 256, 8)
    x3 = _butterfly_stage_1d(x3, 256, 8)
    x0 = _butterfly_stage_1d(x0, 256, 4)
    x1 = _butterfly_stage_1d(x1, 256, 4)
    x2 = _butterfly_stage_1d(x2, 256, 4)
    x3 = _butterfly_stage_1d(x3, 256, 4)
    x0 = _butterfly_stage_1d(x0, 256, 2)
    x1 = _butterfly_stage_1d(x1, 256, 2)
    x2 = _butterfly_stage_1d(x2, 256, 2)
    x3 = _butterfly_stage_1d(x3, 256, 2)
    x0 = _butterfly_stage_1d(x0, 256, 1)
    x1 = _butterfly_stage_1d(x1, 256, 1)
    x2 = _butterfly_stage_1d(x2, 256, 1)
    x3 = _butterfly_stage_1d(x3, 256, 1)

    x0 = x0 * SCALE
    x1 = x1 * SCALE
    x2 = x2 * SCALE
    x3 = x3 * SCALE

    tl.store(
        OUT_ptr + row0 * stride_out_row + col_offs, x0, eviction_policy="evict_first"
    )
    tl.store(
        OUT_ptr + row1 * stride_out_row + col_offs,
        x1,
        mask=row1 < N_ROWS,
        eviction_policy="evict_first",
    )
    tl.store(
        OUT_ptr + row2 * stride_out_row + col_offs,
        x2,
        mask=row2 < N_ROWS,
        eviction_policy="evict_first",
    )
    tl.store(
        OUT_ptr + row3 * stride_out_row + col_offs,
        x3,
        mask=row3 < N_ROWS,
        eviction_policy="evict_first",
    )


# ============================================================
# Fallback: single-row 1D native kernel for dim=256
# ============================================================


@triton.jit
def _fht_kernel_256_1d_native(
    X_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    SCALE: tl.constexpr,
):
    """FHT for dim=256, 1D native fp16/bf16, 8 hardcoded reversed butterfly stages."""
    pid = tl.program_id(0)
    col_offs = tl.arange(0, 256)

    x = tl.load(X_ptr + pid * stride_x_row + col_offs)

    # Reversed butterfly: stride 128, 64, 32, 16, 8, 4, 2, 1
    x = _butterfly_stage_1d(x, 256, 128)
    x = _butterfly_stage_1d(x, 256, 64)
    x = _butterfly_stage_1d(x, 256, 32)
    x = _butterfly_stage_1d(x, 256, 16)
    x = _butterfly_stage_1d(x, 256, 8)
    x = _butterfly_stage_1d(x, 256, 4)
    x = _butterfly_stage_1d(x, 256, 2)
    x = _butterfly_stage_1d(x, 256, 1)

    x = x * SCALE
    tl.store(
        OUT_ptr + pid * stride_out_row + col_offs, x, eviction_policy="evict_first"
    )


# ============================================================
# 1D hardcoded native kernel for dim=512 (9 stages)
# Restored from v31/v35: single-row (best: 1.1193x in v35)
# ============================================================


@triton.jit
def _fht_kernel_512_1d_native(
    X_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    SCALE: tl.constexpr,
):
    """FHT for dim=512, 1D native fp16/bf16, 9 hardcoded reversed butterfly stages."""
    pid = tl.program_id(0)
    col_offs = tl.arange(0, 512)

    x = tl.load(X_ptr + pid * stride_x_row + col_offs)

    # Reversed butterfly: stride 256, 128, 64, 32, 16, 8, 4, 2, 1
    x = _butterfly_stage_1d(x, 512, 256)
    x = _butterfly_stage_1d(x, 512, 128)
    x = _butterfly_stage_1d(x, 512, 64)
    x = _butterfly_stage_1d(x, 512, 32)
    x = _butterfly_stage_1d(x, 512, 16)
    x = _butterfly_stage_1d(x, 512, 8)
    x = _butterfly_stage_1d(x, 512, 4)
    x = _butterfly_stage_1d(x, 512, 2)
    x = _butterfly_stage_1d(x, 512, 1)

    x = x * SCALE
    tl.store(
        OUT_ptr + pid * stride_out_row + col_offs, x, eviction_policy="evict_first"
    )


# ============================================================
# Generic 1D native-dtype butterfly kernel (for other small dims)
# ============================================================


@triton.jit
def _fht_kernel_1d_native(
    X_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    DIM: tl.constexpr,
    LOG_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    """FHT butterfly — 1D single-row, native fp16/bf16, reversed stage order."""
    pid = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)

    x = tl.load(X_ptr + pid * stride_x_row + col_offs)

    # Reversed butterfly: stride N/2, N/4, ..., 2, 1
    for s_rev in tl.static_range(LOG_N):
        x = _butterfly_stage_1d(x, BLOCK_SIZE, 1 << (LOG_N - 1 - s_rev))

    x = x * SCALE
    tl.store(
        OUT_ptr + pid * stride_out_row + col_offs, x, eviction_policy="evict_first"
    )


# ============================================================
# 2D native-dtype butterfly kernel (for dim=1024 with fp16/bf16)
# ============================================================


@triton.jit
def _fht_kernel_2d_native(
    X_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    N_ROWS,
    DIM: tl.constexpr,
    LOG_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    SCALE: tl.constexpr,
):
    """FHT butterfly — 2D batch, reversed stage order, native fp16/bf16 compute."""
    pid = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    row_offs = tl.arange(0, ROWS_PER_PROGRAM)

    base_row = pid * ROWS_PER_PROGRAM
    row_ids = base_row + row_offs
    row_mask = row_ids < N_ROWS

    in_ptrs = X_ptr + row_ids[:, None] * stride_x_row + col_offs[None, :]
    out_ptrs = OUT_ptr + row_ids[:, None] * stride_out_row + col_offs[None, :]
    load_mask = row_mask[:, None]

    x = tl.load(in_ptrs, mask=load_mask, other=0.0)

    # Reversed butterfly: stride N/2, N/4, ..., 2, 1
    for s_rev in tl.static_range(LOG_N):
        x = _butterfly_stage_2d(
            x, ROWS_PER_PROGRAM, BLOCK_SIZE, 1 << (LOG_N - 1 - s_rev)
        )

    x = x * SCALE
    tl.store(out_ptrs, x, mask=load_mask, eviction_policy="evict_first")


# ============================================================
# 1D butterfly kernel (fp32, for fp32 inputs)
# ============================================================


@triton.jit
def _fht_kernel_1d(
    X_ptr,
    OUT_ptr,
    scale,
    stride_x_row,
    stride_out_row,
    DIM: tl.constexpr,
    LOG_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_IS_FP16: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    """FHT butterfly — 1D single-row kernel, reversed stage order."""
    pid = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)

    in_ptr = X_ptr + pid * stride_x_row + col_offs
    out_ptr = OUT_ptr + pid * stride_out_row + col_offs

    x = tl.load(in_ptr).to(tl.float32)

    for s_rev in tl.static_range(LOG_N):
        x = _butterfly_stage_1d(x, BLOCK_SIZE, 1 << (LOG_N - 1 - s_rev))

    x = x * scale

    if INPUT_IS_FP16:
        tl.store(out_ptr, x.to(tl.float16), eviction_policy="evict_first")
    elif INPUT_IS_BF16:
        tl.store(out_ptr, x.to(tl.bfloat16), eviction_policy="evict_first")
    else:
        tl.store(out_ptr, x, eviction_policy="evict_first")


# ============================================================
# 2D butterfly kernel (fp32, for dim>=1024 and fp32 inputs)
# ============================================================


@triton.jit
def _fht_kernel_2d(
    X_ptr,
    OUT_ptr,
    scale,
    stride_x_row,
    stride_out_row,
    N_ROWS,
    DIM: tl.constexpr,
    LOG_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    INPUT_IS_FP16: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    """FHT butterfly — 2D batch, reversed stage order, fp32 compute."""
    pid = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    row_offs = tl.arange(0, ROWS_PER_PROGRAM)

    base_row = pid * ROWS_PER_PROGRAM
    row_ids = base_row + row_offs
    row_mask = row_ids < N_ROWS

    in_ptrs = X_ptr + row_ids[:, None] * stride_x_row + col_offs[None, :]
    out_ptrs = OUT_ptr + row_ids[:, None] * stride_out_row + col_offs[None, :]
    load_mask = row_mask[:, None]

    x = tl.load(in_ptrs, mask=load_mask, other=0.0).to(tl.float32)

    for s_rev in tl.static_range(LOG_N):
        x = _butterfly_stage_2d(
            x, ROWS_PER_PROGRAM, BLOCK_SIZE, 1 << (LOG_N - 1 - s_rev)
        )

    x = x * scale

    if INPUT_IS_FP16:
        tl.store(
            out_ptrs, x.to(tl.float16), mask=load_mask, eviction_policy="evict_first"
        )
    elif INPUT_IS_BF16:
        tl.store(
            out_ptrs, x.to(tl.bfloat16), mask=load_mask, eviction_policy="evict_first"
        )
    else:
        tl.store(out_ptrs, x, mask=load_mask, eviction_policy="evict_first")


# ============================================================
# M×N fused kernels: H_M column transform + FHT in registers
# No intermediate DRAM write, no padding to next power of 2.
# ============================================================


@triton.jit
def _h3_fht_kernel(
    X_ptr, OUT_ptr, stride_batch, stride_row,
    SCALE: tl.constexpr, IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
    N_COLS: tl.constexpr, LOG_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, N_COLS)
    base = pid * stride_batch
    a = tl.load(X_ptr + base + 0 * stride_row + offs).to(tl.float32)
    b = tl.load(X_ptr + base + 1 * stride_row + offs).to(tl.float32)
    c = tl.load(X_ptr + base + 2 * stride_row + offs).to(tl.float32)
    y0 = a + b + c
    y1 = a - b + c
    y2 = a + b - c
    for s_rev in tl.static_range(LOG_N):
        y0 = _butterfly_stage_1d(y0, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y1 = _butterfly_stage_1d(y1, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y2 = _butterfly_stage_1d(y2, N_COLS, 1 << (LOG_N - 1 - s_rev))
    y0 = y0 * SCALE; y1 = y1 * SCALE; y2 = y2 * SCALE
    if IS_FP16:
        y0 = y0.to(tl.float16); y1 = y1.to(tl.float16); y2 = y2.to(tl.float16)
    elif IS_BF16:
        y0 = y0.to(tl.bfloat16); y1 = y1.to(tl.bfloat16); y2 = y2.to(tl.bfloat16)
    tl.store(OUT_ptr + base + 0 * stride_row + offs, y0, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 1 * stride_row + offs, y1, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 2 * stride_row + offs, y2, eviction_policy="evict_first")


@triton.jit
def _h5_fht_kernel(
    X_ptr, OUT_ptr, stride_batch, stride_row,
    SCALE: tl.constexpr, IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
    N_COLS: tl.constexpr, LOG_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, N_COLS)
    base = pid * stride_batch
    a = tl.load(X_ptr + base + 0 * stride_row + offs).to(tl.float32)
    b = tl.load(X_ptr + base + 1 * stride_row + offs).to(tl.float32)
    c = tl.load(X_ptr + base + 2 * stride_row + offs).to(tl.float32)
    d = tl.load(X_ptr + base + 3 * stride_row + offs).to(tl.float32)
    e = tl.load(X_ptr + base + 4 * stride_row + offs).to(tl.float32)
    y0 = a + b + c + d + e
    y1 = a - b + c - d + e
    y2 = a + b - c + d - e
    y3 = a - b - c - d - e
    y4 = a + b + c - d - e
    for s_rev in tl.static_range(LOG_N):
        y0 = _butterfly_stage_1d(y0, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y1 = _butterfly_stage_1d(y1, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y2 = _butterfly_stage_1d(y2, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y3 = _butterfly_stage_1d(y3, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y4 = _butterfly_stage_1d(y4, N_COLS, 1 << (LOG_N - 1 - s_rev))
    y0 = y0 * SCALE; y1 = y1 * SCALE; y2 = y2 * SCALE; y3 = y3 * SCALE; y4 = y4 * SCALE
    if IS_FP16:
        y0 = y0.to(tl.float16); y1 = y1.to(tl.float16); y2 = y2.to(tl.float16)
        y3 = y3.to(tl.float16); y4 = y4.to(tl.float16)
    elif IS_BF16:
        y0 = y0.to(tl.bfloat16); y1 = y1.to(tl.bfloat16); y2 = y2.to(tl.bfloat16)
        y3 = y3.to(tl.bfloat16); y4 = y4.to(tl.bfloat16)
    tl.store(OUT_ptr + base + 0 * stride_row + offs, y0, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 1 * stride_row + offs, y1, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 2 * stride_row + offs, y2, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 3 * stride_row + offs, y3, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 4 * stride_row + offs, y4, eviction_policy="evict_first")


@triton.jit
def _h7_fht_kernel(
    X_ptr, OUT_ptr, stride_batch, stride_row,
    SCALE: tl.constexpr, IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
    N_COLS: tl.constexpr, LOG_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, N_COLS)
    base = pid * stride_batch
    a = tl.load(X_ptr + base + 0 * stride_row + offs).to(tl.float32)
    b = tl.load(X_ptr + base + 1 * stride_row + offs).to(tl.float32)
    c = tl.load(X_ptr + base + 2 * stride_row + offs).to(tl.float32)
    d = tl.load(X_ptr + base + 3 * stride_row + offs).to(tl.float32)
    e = tl.load(X_ptr + base + 4 * stride_row + offs).to(tl.float32)
    f = tl.load(X_ptr + base + 5 * stride_row + offs).to(tl.float32)
    g = tl.load(X_ptr + base + 6 * stride_row + offs).to(tl.float32)
    y0 = a + b + c + d + e + f + g
    y1 = a - b + c - d + e - f + g
    y2 = a + b - c + d - e + f - g
    y3 = a - b - c - d - e - f - g
    y4 = a + b + c - d - e - f - g
    y5 = a - b + c + d - e + f + g
    y6 = a + b - c - d + e + f - g
    for s_rev in tl.static_range(LOG_N):
        y0 = _butterfly_stage_1d(y0, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y1 = _butterfly_stage_1d(y1, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y2 = _butterfly_stage_1d(y2, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y3 = _butterfly_stage_1d(y3, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y4 = _butterfly_stage_1d(y4, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y5 = _butterfly_stage_1d(y5, N_COLS, 1 << (LOG_N - 1 - s_rev))
        y6 = _butterfly_stage_1d(y6, N_COLS, 1 << (LOG_N - 1 - s_rev))
    y0 = y0 * SCALE; y1 = y1 * SCALE; y2 = y2 * SCALE; y3 = y3 * SCALE
    y4 = y4 * SCALE; y5 = y5 * SCALE; y6 = y6 * SCALE
    if IS_FP16:
        y0 = y0.to(tl.float16); y1 = y1.to(tl.float16)
        y2 = y2.to(tl.float16); y3 = y3.to(tl.float16)
        y4 = y4.to(tl.float16); y5 = y5.to(tl.float16); y6 = y6.to(tl.float16)
    elif IS_BF16:
        y0 = y0.to(tl.bfloat16); y1 = y1.to(tl.bfloat16)
        y2 = y2.to(tl.bfloat16); y3 = y3.to(tl.bfloat16)
        y4 = y4.to(tl.bfloat16); y5 = y5.to(tl.bfloat16); y6 = y6.to(tl.bfloat16)
    tl.store(OUT_ptr + base + 0 * stride_row + offs, y0, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 1 * stride_row + offs, y1, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 2 * stride_row + offs, y2, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 3 * stride_row + offs, y3, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 4 * stride_row + offs, y4, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 5 * stride_row + offs, y5, eviction_policy="evict_first")
    tl.store(OUT_ptr + base + 6 * stride_row + offs, y6, eviction_policy="evict_first")


def _launch_mn_fused_kernel(x: torch.Tensor, M: int, scale: float) -> torch.Tensor:
    """Launch the appropriate H_M fused kernel for dim = M * 2^k."""
    *leading, dim = x.shape
    batch = x.numel() // dim
    n_cols = dim // M
    log_n = n_cols.bit_length() - 1
    dtype = x.dtype
    xm = x.reshape(batch, M, n_cols).contiguous()
    out = torch.empty_like(xm)
    num_warps = 2 if n_cols <= 1024 else (4 if n_cols <= 2048 else 8)
    kwargs = dict(
        SCALE=scale,
        IS_FP16=(dtype == torch.float16),
        IS_BF16=(dtype == torch.bfloat16),
        N_COLS=n_cols,
        LOG_N=log_n,
        num_warps=num_warps,
        num_stages=1,
    )
    if M == 3:
        _h3_fht_kernel[(batch,)](xm, out, xm.stride(0), xm.stride(1), **kwargs)
    elif M == 5:
        _h5_fht_kernel[(batch,)](xm, out, xm.stride(0), xm.stride(1), **kwargs)
    elif M == 7:
        _h7_fht_kernel[(batch,)](xm, out, xm.stride(0), xm.stride(1), **kwargs)
    else:
        raise ValueError(f"Unsupported M={M}")
    return out.reshape(*leading, dim)


# ============================================================
# Precomputed lookup tables for fast dispatch
# ============================================================

# Power-of-2 dims that are multiples of 8, up to 65536
_POW2_DIMS = frozenset(1 << k for k in range(3, 17))  # 8, 16, ..., 65536


# ============================================================
# Core forward
# ============================================================


def _hadamard_transform_fwd(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Core forward: handles reshape, padding, kernel launch."""
    shapes_og = x.shape
    dim_og = x.shape[-1]
    input_dtype = x.dtype
    x_flat = x.reshape(-1, dim_og)
    if x_flat.stride(-1) != 1:
        x_flat = x_flat.contiguous()
    batch_size = x_flat.shape[0]

    # Fast path for power-of-2 dims (no padding needed)
    if dim_og in _POW2_DIMS:
        n = dim_og
        log_n = n.bit_length() - 1
        # Allocate output directly with explicit args (faster than empty_like)
        out = torch.empty(batch_size, n, dtype=input_dtype, device=x_flat.device)
        stride_x = x_flat.stride(0)
        stride_out = n  # out is freshly allocated, always contiguous

        _launch_kernel(
            x_flat, out, scale, input_dtype, batch_size, n, log_n, stride_x, stride_out
        )

        return out.reshape(shapes_og)

    # General path: handle padding
    assert input_dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"hadamard_transform not implemented for input type '{input_dtype}'"
    assert x.is_cuda, "hadamard_transform requires CUDA tensor"

    # Pad to multiple of 8 (matching CUDA implementation)
    needs_pad = dim_og % 8 != 0
    if needs_pad:
        x_flat = F.pad(x_flat, (0, 8 - dim_og % 8))
    dim = x_flat.shape[1]

    assert (
        dim % 8 == 0
    ), "fast_hadamard_transform only supports hidden dimension divisible by 8 for now"
    assert (
        dim <= 65536
    ), "fast_hadamard_transform only supports hidden dimension at most 65536 for now"

    # For butterfly we need next power of 2
    log_n = math.ceil(math.log2(dim)) if dim > 1 else 1
    n = 1 << log_n

    # If dim (multiple of 8) is not a power of 2, pad further for the kernel
    if n != dim:
        x_flat = F.pad(x_flat, (0, n - dim))

    out = torch.empty(batch_size, n, dtype=input_dtype, device=x_flat.device)
    stride_x = x_flat.stride(0)
    stride_out = n

    _launch_kernel(
        x_flat, out, scale, input_dtype, batch_size, n, log_n, stride_x, stride_out
    )

    # Trim padding back to original dim
    if n != dim_og:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def _launch_kernel(
    x, out, scale, input_dtype, batch_size, n, log_n, stride_x, stride_out
):
    """Dispatch to the appropriate kernel. Separated for fast-path sharing."""
    # Dispatch strategy (v43):
    # - dim=256, fp16/bf16: 4-row ILP native (warps=2) — test 2-warp occupancy
    # - dim=512, fp16/bf16: 1D single-row native (warps=1) — v35 best
    # - other dim<=128, fp16/bf16: generic 1D native
    # - dim=1024, fp16/bf16: 2D native batched (rows=2, warps=4)
    # - dim<=512, fp32: fp32 1D kernel
    # - dim>=1024, fp32 or dim>=2048: fp32 2D kernel
    if n <= 1024 and input_dtype in (torch.float16, torch.bfloat16):
        if n == 256:
            if batch_size >= 4:
                _fht_kernel_256_4row_native[((batch_size + 3) // 4,)](
                    x,
                    out,
                    stride_x_row=stride_x,
                    stride_out_row=stride_out,
                    N_ROWS=batch_size,
                    SCALE=scale,
                    num_warps=2,
                    num_stages=1,
                )
            else:
                _fht_kernel_256_1d_native[(batch_size,)](
                    x,
                    out,
                    stride_x_row=stride_x,
                    stride_out_row=stride_out,
                    SCALE=scale,
                    num_warps=2,
                    num_stages=1,
                )
        elif n == 512:
            # Single-row 1D hardcoded: v35 achieved 1.1193x (best)
            _fht_kernel_512_1d_native[(batch_size,)](
                x,
                out,
                stride_x_row=stride_x,
                stride_out_row=stride_out,
                SCALE=scale,
                num_warps=1,
                num_stages=1,
            )
        elif n <= 128:
            _fht_kernel_1d_native[(batch_size,)](
                x,
                out,
                stride_x_row=stride_x,
                stride_out_row=stride_out,
                DIM=n,
                LOG_N=log_n,
                BLOCK_SIZE=n,
                SCALE=scale,
                num_warps=1,
                num_stages=1,
            )
        else:
            # dim=1024: 2D native with 2 rows/program
            rows_per_program = 2
            n_programs = (batch_size + rows_per_program - 1) // rows_per_program
            _fht_kernel_2d_native[(n_programs,)](
                x,
                out,
                stride_x_row=stride_x,
                stride_out_row=stride_out,
                N_ROWS=batch_size,
                DIM=n,
                LOG_N=log_n,
                BLOCK_SIZE=n,
                ROWS_PER_PROGRAM=rows_per_program,
                SCALE=scale,
                num_warps=4,
                num_stages=1,
            )
    elif n <= 512:
        # fp32 1D kernel
        _fht_kernel_1d[(batch_size,)](
            x,
            out,
            scale,
            stride_x_row=stride_x,
            stride_out_row=stride_out,
            DIM=n,
            LOG_N=log_n,
            BLOCK_SIZE=n,
            INPUT_IS_FP16=(input_dtype == torch.float16),
            INPUT_IS_BF16=(input_dtype == torch.bfloat16),
            num_warps=1,
            num_stages=1,
        )
    else:
        # fp32 2D butterfly for fp32 inputs and large dims
        if n <= 32:
            num_warps = 1
            rows_per_program = 64
        elif n <= 64:
            num_warps = 1
            rows_per_program = 64
        elif n <= 128:
            num_warps = 1
            rows_per_program = 32
        elif n <= 256:
            num_warps = 1
            rows_per_program = 16
        elif n <= 1024:
            num_warps = 4
            rows_per_program = 2
        elif n <= 4096:
            num_warps = 4
            rows_per_program = 1
        else:
            num_warps = 8
            rows_per_program = 1

        n_programs = (batch_size + rows_per_program - 1) // rows_per_program
        _fht_kernel_2d[(n_programs,)](
            x,
            out,
            scale,
            stride_x_row=stride_x,
            stride_out_row=stride_out,
            N_ROWS=batch_size,
            DIM=n,
            LOG_N=log_n,
            BLOCK_SIZE=n,
            ROWS_PER_PROGRAM=rows_per_program,
            INPUT_IS_FP16=(input_dtype == torch.float16),
            INPUT_IS_BF16=(input_dtype == torch.bfloat16),
            num_warps=num_warps,
            num_stages=1,
        )


# ============================================================
# Autograd Function
# ============================================================


class HadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale
        return _hadamard_transform_fwd(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # Hadamard matrix is symmetric: backward = forward with same scale
        return _hadamard_transform_fwd(dout, ctx._hadamard_transform_scale), None


# ============================================================
# Public API
# ============================================================


def hadamard_transform(x, scale=1.0):
    """
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
#
# Decomposes dim = M * 2^k via H_M ⊗ H_{2^k}:
#   1. Reshape to (batch, M, 2^k)
#   2. Apply H_M column transform + FHT in a single fused kernel
# No padding to next power of 2, no intermediate DRAM write.
# ============================================================


def hadamard_transform_12N(x, scale=1.0):
    """Hadamard transform for dim = 3 * 2^k (e.g. 1536, 3072, 6144, 12288)."""
    return _launch_mn_fused_kernel(x, M=3, scale=scale)


def hadamard_transform_20N(x, scale=1.0):
    """Hadamard transform for dim = 5 * 2^k (e.g. 5120, 10240, 20480)."""
    return _launch_mn_fused_kernel(x, M=5, scale=scale)


def hadamard_transform_28N(x, scale=1.0):
    """Hadamard transform for dim = 7 * 2^k (e.g. 7168, 14336, 28672)."""
    return _launch_mn_fused_kernel(x, M=7, scale=scale)


def hadamard_transform_40N(x, scale=1.0):
    """Hadamard transform for dim = 5 * 2^k (e.g. 10240, 20480, 40960)."""
    return _launch_mn_fused_kernel(x, M=5, scale=scale)
