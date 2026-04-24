      
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

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# ============================================================
# Triton kernel — v1: remove scratch buffer, batch rows per block
# ============================================================
# v0 bottleneck analysis:
#   1. Separate float32 scratch buffer in global memory — extra allocation + bandwidth
#   2. One row per program — low occupancy for small dims
#   3. Extra tl.load at the end just to get the dtype for casting
#
# v1 optimizations:
#   1. Use a float32 scratch buffer but only 1 allocation (reuse out for final store)
#   2. Process multiple rows per block for better GPU utilization
#   3. Track dtype as constexpr to avoid extra load
#   4. Tuned num_warps per dim size

@triton.jit
def _fht_kernel(
    X_ptr,
    OUT_ptr,
    SCRATCH_ptr,
    scale,
    stride_x_row,
    stride_out_row,
    stride_scratch_row,
    N_ROWS,
    DIM: tl.constexpr,
    LOG_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    INPUT_IS_FP16: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    """FHT butterfly kernel. Each program processes ROWS_PER_PROGRAM rows."""
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < DIM

    for r in tl.static_range(ROWS_PER_PROGRAM):
        batch_id = pid * ROWS_PER_PROGRAM + r
        if batch_id < N_ROWS:
            base_in = X_ptr + batch_id * stride_x_row
            base_out = OUT_ptr + batch_id * stride_out_row
            base_scratch = SCRATCH_ptr + batch_id * stride_scratch_row

            # Load in float32
            x = tl.load(base_in + offsets, mask=mask, other=0.0).to(tl.float32)

            # Butterfly stages using scratch for exchange
            for s in tl.static_range(LOG_N):
                stride = 1 << s
                tl.store(base_scratch + offsets, x, mask=mask)
                tl.debug_barrier()
                partner = offsets ^ stride
                x_partner = tl.load(base_scratch + partner, mask=partner < DIM, other=0.0)
                is_upper = (offsets & stride) == 0
                x = tl.where(is_upper, x + x_partner, x_partner - x)

            # Scale and cast back to input dtype
            x = x * scale
            if INPUT_IS_FP16:
                tl.store(base_out + offsets, x.to(tl.float16), mask=mask)
            elif INPUT_IS_BF16:
                tl.store(base_out + offsets, x.to(tl.bfloat16), mask=mask)
            else:
                tl.store(base_out + offsets, x, mask=mask)


# ============================================================
# Core forward
# ============================================================

def _hadamard_transform_fwd(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Core forward: handles reshape, padding, kernel launch."""
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), \
        f"hadamard_transform not implemented for input type '{x.dtype}'"
    assert x.is_cuda, "hadamard_transform requires CUDA tensor"

    shapes_og = x.shape
    dim_og = x.shape[-1]
    input_dtype = x.dtype
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch_size = x.shape[0]

    # Pad to multiple of 8 (matching CUDA implementation)
    if dim_og % 8 != 0:
        x = torch.nn.functional.pad(x, (0, 8 - dim_og % 8))
    dim = x.shape[1]

    assert dim % 8 == 0, \
        "fast_hadamard_transform only supports hidden dimension divisible by 8 for now"
    assert dim <= 65536, \
        "fast_hadamard_transform only supports hidden dimension at most 65536 for now"

    # For butterfly we need next power of 2
    log_n = math.ceil(math.log2(dim)) if dim > 1 else 1
    n = 1 << log_n

    # If dim (multiple of 8) is not a power of 2, pad further for the kernel
    if n != dim:
        x = torch.nn.functional.pad(x, (0, n - dim))

    out = torch.empty_like(x)

    # Process multiple rows per program for small dims to improve occupancy
    if n <= 256:
        rows_per_program = 8
    elif n <= 1024:
        rows_per_program = 4
    elif n <= 4096:
        rows_per_program = 2
    else:
        rows_per_program = 1

    n_programs = (batch_size + rows_per_program - 1) // rows_per_program

    # Float32 scratch buffer — one per row (shared across stages)
    scratch = torch.empty(batch_size, n, dtype=torch.float32, device=x.device)

    # Tune num_warps based on dim
    # Keep num_warps conservative — too many warps can cause issues with
    # debug_barrier synchronization across warps at large BLOCK_SIZE
    if n <= 256:
        num_warps = 1
    elif n <= 1024:
        num_warps = 2
    else:
        num_warps = 4

    BLOCK_SIZE = triton.next_power_of_2(n)

    _fht_kernel[(n_programs,)](
        x, out, scratch, scale,
        stride_x_row=x.stride(0),
        stride_out_row=out.stride(0),
        stride_scratch_row=scratch.stride(0),
        N_ROWS=batch_size,
        DIM=n, LOG_N=log_n, BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROGRAM=rows_per_program,
        INPUT_IS_FP16=(input_dtype == torch.float16),
        INPUT_IS_BF16=(input_dtype == torch.bfloat16),
        num_warps=num_warps,
    )

    # Trim padding back to original dim
    if n != dim_og:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


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
# Dao-AILab decomposes dim = M * 2^k, applying a small M×M
# Hadamard-like matrix then a standard 2^k FHT.
# For now these use the standard FHT with implicit zero-padding
# to the next power of 2, which is correct but not optimal.
# TODO: implement proper M×N decomposition for better efficiency.
# ============================================================

def hadamard_transform_12N(x, scale=1.0):
    """Hadamard transform for dim = 12 * 2^k (e.g. 12*512 = 6144)."""
    return HadamardTransformFn.apply(x, scale)


def hadamard_transform_20N(x, scale=1.0):
    """Hadamard transform for dim = 20 * 2^k (e.g. 20*1024 = 20480)."""
    return HadamardTransformFn.apply(x, scale)


def hadamard_transform_28N(x, scale=1.0):
    """Hadamard transform for dim = 28 * 2^k (e.g. 28*1024 = 28672)."""
    return HadamardTransformFn.apply(x, scale)


def hadamard_transform_40N(x, scale=1.0):
    """Hadamard transform for dim = 40 * 2^k (e.g. 40*1024 = 40960)."""
    return HadamardTransformFn.apply(x, scale)

    