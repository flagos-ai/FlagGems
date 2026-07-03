import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64

# isinf(x): x is +/-inf  <=>  (x == inf) | (x == -inf). Same logic as the 智源
# pointwise_dynamic override (inf literals work in kernels), wrapped in the
# user's grid-cap + no-mask fast-path perf structure. Note: do NOT use `x == x`
# for the NaN case — this TPU returns True for `nan == nan`, so NaN would be
# misclassified; the inf-literal compare correctly yields False for NaN.


@triton.jit
def isinf_kernel_fast(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, (x == float("inf")) | (x == float("-inf")))


@triton.jit
def isinf_kernel_masked(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, (x == float("inf")) | (x == float("-inf")), mask=mask)


def isinf(input):
    logger.debug("GEMS ISINF (sophgo_tpu)")
    if not input.is_floating_point():
        return torch.full(input.shape, False, dtype=torch.bool, device=input.device)
    out = torch.empty(input.shape, dtype=torch.bool, device=input.device)
    n = input.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    if n % BLOCK_SIZE == 0:
        isinf_kernel_fast[(grid,)](input, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    else:
        isinf_kernel_masked[(grid,)](input, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out
