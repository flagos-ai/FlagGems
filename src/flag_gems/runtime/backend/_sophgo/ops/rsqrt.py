import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAX_GRID = 64


@triton.jit
def rsqrt_kernel_fast(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / tl.sqrt(x))


@triton.jit
def rsqrt_kernel_masked(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / tl.sqrt(x), mask=mask)


def _select_bs(n):
    if n <= 4096:
        return 4096
    return 8192


def _dispatch_params(n):
    bs = _select_bs(n)
    num_tiles = math.ceil(n / bs)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    return bs, grid, tpb


def rsqrt(A):
    logger.debug("GEMS RSQRT (sophgo_tpu)")
    out = torch.empty_like(A)
    n = A.numel()
    bs, grid, tpb = _dispatch_params(n)
    if n % bs == 0:
        rsqrt_kernel_fast[(grid,)](A, out, n, BLOCK_SIZE=bs, TPB=tpb)
    else:
        rsqrt_kernel_masked[(grid,)](A, out, n, BLOCK_SIZE=bs, TPB=tpb)
    return out


def rsqrt_(A):
    logger.debug("GEMS RSQRT_ (sophgo_tpu)")
    n = A.numel()
    bs, grid, tpb = _dispatch_params(n)
    if n % bs == 0:
        rsqrt_kernel_fast[(grid,)](A, A, n, BLOCK_SIZE=bs, TPB=tpb)
    else:
        rsqrt_kernel_masked[(grid,)](A, A, n, BLOCK_SIZE=bs, TPB=tpb)
    return A
