import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from ..utils.shape_utils import dim_compress


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, mid_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """
    Stage 1: Count non-zero elements within each block.
    Uses 2D tensor mode to avoid scalar operations (see mean operator fix).

    Problem: In the original implementation, storing after tl.sum() returns a scalar triggers ppl.get_value,
    causing direct pointer dereference to fail on TPU.

    Fix: Use 2D tensor operations to keep all intermediate results in tensor form.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < numel

    # Load data and convert to non-zero mask [BLOCK_SIZE]
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(
        tl.float32
    )  # Use FP32 (PPL avgPool2D does not support INT32)

    # Convert to 2D tensor [1, BLOCK_SIZE]
    is_nonzero_2d = is_nonzero[None, :]

    # Sum along axis=1, result is [1] tensor (not scalar)
    # This avoids scalar operations triggering ppl.get_value
    local_count = tl.sum(is_nonzero_2d, axis=1)  # shape: [1]

    # Use 1-element tensor store pattern
    # Construct store address as tensor (avoid scalar store triggering ppl.get_value)
    store_offset = tl.arange(0, 1)  # [0]
    store_mask = store_offset < 1  # [True]
    store_addr = mid_ptr + pid + store_offset

    # Store [1] tensor
    tl.store(store_addr, local_count, mask=store_mask)


@libentry()
@triton.jit
def count_nonzero_kernel_2(mid_ptr, out_ptr, mid_size, BLOCK_MID: tl.constexpr):
    """
    Stage 2: Aggregate intermediate results with int32 accumulation.

    Uses chunked float32 sum (each chunk sum < 2^24 for exact integer
    representation) and int32 element-wise addition to avoid precision
    loss when the total count exceeds float32's precise integer range.
    """
    total = tl.zeros([1], dtype=tl.int32)

    for i in range(0, mid_size, BLOCK_MID):
        offset = i + tl.arange(0, BLOCK_MID)
        mask = offset < mid_size
        mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)
        chunk_sum = tl.sum(mid_val[None, :], axis=1)
        total = total + chunk_sum.to(tl.int32)

    store_offset = tl.arange(0, 1)
    tl.store(out_ptr + store_offset, total, mask=store_offset < 1)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero_dim"), key=["M", "N"])
@triton.jit
def count_nonzero_dim_kernel(
    X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    TPU-adapted version: count non-zero elements per dimension.
    Uses 2D tensor mode to avoid scalar operations (see mean_dim_kernel).

    Uses [BLOCK_M, 1] int32 accumulator to avoid float32 precision loss
    when the reduced dimension exceeds 2^24 elements.
    """
    # Process BLOCK_M rows (2D mode)
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    # int32 accumulator to avoid float32 precision loss when row count > 2^24.
    # Each chunk sum is < BLOCK_N, well within float32 precise range.
    counts = tl.zeros([BLOCK_M, 1], dtype=tl.int32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask=mask, other=0.0)
        is_nonzero = (x != 0).to(tl.float32)
        counts += tl.sum(is_nonzero, axis=1).to(tl.int32)[:, None]

    tl.store(Out, counts, row_mask)


def count_nonzero(x, dim=None):
    """
    TPU-adapted version:
    - dim=None: Uses two-stage kernel + 2D tensor mode, fully completed in Triton.
    - dim=N: Uses count_nonzero_dim_kernel + 2D tensor mode (see mean_dim).
    """
    logging.debug("GEMS_sophgo COUNT NONZERO")
    if dim is not None:
        # Use 2D tensor mode for dimension reduction (similar to mean_dim)
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        shape = list(x.shape)
        dim = dim % x.ndim
        x = dim_compress(x, dim)
        N = shape[dim]
        out_shape = list(shape)
        del out_shape[dim]

        if N == 0:
            return torch.zeros(out_shape, dtype=torch.int32, device=x.device)

        M = x.numel() // N
        out = torch.zeros(out_shape, dtype=torch.int32, device=x.device)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            count_nonzero_dim_kernel[grid](x.flatten(), out, M, N)
        return out
    else:
        # Full tensor case: two-stage reduction (fully completed in Triton)
        # Reference the mean operator fix approach
        x = x.contiguous().flatten()
        numel = x.numel()

        if numel == 0:
            return torch.tensor(0, dtype=torch.int32, device=x.device)

        # Calculate block size and number of programs (same as mean operator)
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(numel)))
        mid_size = triton.cdiv(numel, block_size)

        # Safe BLOCK_MID: each chunk's float32 sum must be < 2^24
        # Stage 1 partial count per block ≤ block_size, so
        # BLOCK_MID * block_size < 2^24 for exact int representation.
        # Must also be a power of 2 (tl.arange requirement).
        max_safe_mid = max(1, (2**24 - 1) // block_size)
        raw_mid = min(mid_size, max_safe_mid)
        block_mid = 1 << (raw_mid.bit_length() - 1)

        # Allocate intermediate results and output
        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty([], dtype=torch.int32, device=x.device)

        with torch_device_fn.device(x.device):
            # Stage 1: compute non-zero count for each block
            count_nonzero_kernel_1[(mid_size, 1, 1)](x, mid, numel, block_size)

            # Stage 2: aggregate with int32 accumulation
            count_nonzero_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

        return out
