import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle


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
    Stage 2: Aggregate intermediate results.
    Uses 2D tensor mode to avoid scalar operations.
    """
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size

    # Load intermediate results [BLOCK_MID]
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)

    # Convert to 2D tensor [1, BLOCK_MID]
    mid_val_2d = mid_val[None, :]

    # Sum along axis=1, result is [1] tensor
    total_count = tl.sum(mid_val_2d, axis=1)  # shape: [1]

    # 1-element tensor store
    store_offset = tl.arange(0, 1)
    store_mask = store_offset < 1
    tl.store(out_ptr + store_offset, total_count, mask=store_mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero_dim"), key=["M", "N"])
@triton.jit
def count_nonzero_dim_kernel(
    X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    TPU-adapted version: count non-zero elements per dimension.
    Uses 2D tensor mode to avoid scalar operations (see mean_dim_kernel).

    Problem: The original count_nonzero_kernel uses tl.sum() returning a scalar,
    causing tl.store() to trigger ppl.get_value which fails on TPU.

    Fix: Use [BLOCK_M, BLOCK_N] 2D accumulator,
    tl.sum(axis=1) returns [BLOCK_M] tensor instead of scalar.
    """
    # Process BLOCK_M rows (2D mode)
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    # 2D accumulator [BLOCK_M, BLOCK_N]
    counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask=mask, other=0.0)
        is_nonzero = (x != 0).to(tl.float32)
        counts += is_nonzero

    # Sum along axis=1 → [BLOCK_M] tensor (not scalar)
    row_counts = tl.sum(counts, axis=1)
    # Convert to 2D for storage [BLOCK_M, 1]
    row_counts = row_counts[:, None]
    tl.store(Out, row_counts.to(tl.int32), row_mask)


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
        M = x.numel() // N

        out_shape = list(shape)
        del out_shape[dim]
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

        # Calculate block size and number of programs (same as mean operator)
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(numel)))
        mid_size = triton.cdiv(numel, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        # Allocate intermediate results and output
        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty([], dtype=torch.float32, device=x.device)

        with torch_device_fn.device(x.device):
            # Stage 1: compute non-zero count for each block
            count_nonzero_kernel_1[(mid_size, 1, 1)](x, mid, numel, block_size)

            # Stage 2: aggregate all intermediate results
            count_nonzero_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

        return out.to(torch.int32)
