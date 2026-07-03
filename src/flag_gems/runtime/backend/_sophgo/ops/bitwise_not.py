import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64


@triton.jit
def bitwise_not_kernel_fast(
    x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs, ~tl.load(x_ptr + offs))


@triton.jit
def bitwise_not_kernel_masked(
    x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(out_ptr + offs, ~tl.load(x_ptr + offs, mask=mask), mask=mask)


def bitwise_not(A):
    logger.debug("GEMS BITWISE_NOT (sophgo_tpu)")
    if A.dtype == torch.bool:
        # raw `~x` on bool yields the full bitwise complement, not the logical
        # flip torch.bitwise_not does; delegate bool to the generic op.
        from flag_gems.ops.bitwise_not import bitwise_not as _generic_bitwise_not

        return _generic_bitwise_not(A)
    out = torch.empty_like(A)
    n = A.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    if n % BLOCK_SIZE == 0:
        bitwise_not_kernel_fast[(grid,)](A, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    else:
        bitwise_not_kernel_masked[(grid,)](A, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out


def bitwise_not_(A):
    logger.debug("GEMS BITWISE_NOT_ (sophgo_tpu)")
    if A.dtype == torch.bool:
        from flag_gems.ops.bitwise_not import bitwise_not_ as _generic_bitwise_not_

        return _generic_bitwise_not_(A)
    n = A.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    if n % BLOCK_SIZE == 0:
        bitwise_not_kernel_fast[(grid,)](A, A, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    else:
        bitwise_not_kernel_masked[(grid,)](A, A, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return A
