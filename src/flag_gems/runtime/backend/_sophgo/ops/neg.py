import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64


@triton.jit
def neg_kernel_fast(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs, -tl.load(x_ptr + offs))


@triton.jit
def neg_kernel_masked(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(out_ptr + offs, -tl.load(x_ptr + offs, mask=mask), mask=mask)


def neg(A):
    logger.debug("GEMS NEG (sophgo_tpu)")
    out = torch.empty_like(A)
    n = A.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    if n % BLOCK_SIZE == 0:
        neg_kernel_fast[(grid,)](A, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    else:
        neg_kernel_masked[(grid,)](A, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out


def neg_(A):
    logger.debug("GEMS NEG_ (sophgo_tpu)")
    n = A.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    if n % BLOCK_SIZE == 0:
        neg_kernel_fast[(grid,)](A, A, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    else:
        neg_kernel_masked[(grid,)](A, A, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return A
