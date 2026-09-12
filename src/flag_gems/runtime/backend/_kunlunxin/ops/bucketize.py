# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of bucketize.
#
# Root cause: the generic binary-search kernel
# (flag_gems/ops/bucketize.py) trips the XPU MLIR backend:
#   error: 'arith.addi' op requires the same type for all operands
#   -> PassManager::run failed / OutOfResources.
# The mixed-width int arithmetic inside the `(lo + hi) // 2` binary search
# does not lower on XPU (62 fp16/bf16/fp32 + int32 + boundary cases fail).
#
# Fix: replace the binary search with a straight linear scan over the
# boundaries (they are tiny -- <= 32 in the suite and sorted ascending).
# For each boundary we accumulate a `tl.where` count; no mixed-int divide,
# no data-dependent loop bound, so XPU codegen is happy.
#   right=False : idx = #{ b : b <  v }
#   right=True  : idx = #{ b : b <= v }
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def bucketize_kernel(
    inp_ptr,
    boundaries_ptr,
    out_ptr,
    n_elements,
    N_BOUNDARIES: tl.constexpr,
    right: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    v = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in tl.static_range(N_BOUNDARIES):
        b = tl.load(boundaries_ptr + i).to(tl.float32)
        if right:
            cond = b <= v
        else:
            cond = b < v
        idx = tl.where(cond, i + 1, idx)

    tl.store(out_ptr + offsets, idx.to(tl.int64), mask=mask)


def bucketize(input, boundaries, *, out_int32=False, right=False):
    logger.debug("GEMS_KUNLUNXIN BUCKETIZE")
    output_dtype = torch.int32 if out_int32 else torch.int64

    if boundaries.numel() == 0:
        return torch.zeros_like(input, dtype=output_dtype)

    output = torch.empty_like(input, dtype=torch.int64)

    n_elements = input.numel()
    n_boundaries = boundaries.numel()

    input_flat = input.contiguous().flatten()
    output_flat = output.flatten()
    boundaries = boundaries.contiguous()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)

    bucketize_kernel[grid](
        input_flat,
        boundaries,
        output_flat,
        n_elements,
        n_boundaries,
        right,
        BLOCK_SIZE,
        num_warps=1,
        buffer_size_limit=2048,
        isCloseVectorization=True,
    )

    output = output.reshape(input.shape)
    if out_int32:
        output = output.to(torch.int32)
    return output
