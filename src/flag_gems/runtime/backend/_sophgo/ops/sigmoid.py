import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAX_GRID = 64


@triton.jit
def sigmoid_kernel_fast(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / (1.0 + tl.exp(-x)))


@triton.jit
def sigmoid_kernel_masked(
    x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / (1.0 + tl.exp(-x)), mask=mask)


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


def sigmoid(self):
    logger.debug("GEMS SIGMOID FORWARD (sophgo_tpu)")
    out = torch.empty_like(self)
    n = self.numel()
    bs, grid, tpb = _dispatch_params(n)
    if n % bs == 0:
        sigmoid_kernel_fast[(grid,)](self, out, n, BLOCK_SIZE=bs, TPB=tpb)
    else:
        sigmoid_kernel_masked[(grid,)](self, out, n, BLOCK_SIZE=bs, TPB=tpb)
    return out


def sigmoid_(A):
    logger.debug("GEMS SIGMOID_ FORWARD (sophgo_tpu)")
    n = A.numel()
    bs, grid, tpb = _dispatch_params(n)
    if n % bs == 0:
        sigmoid_kernel_fast[(grid,)](A, A, n, BLOCK_SIZE=bs, TPB=tpb)
    else:
        sigmoid_kernel_masked[(grid,)](A, A, n, BLOCK_SIZE=bs, TPB=tpb)
    return A
