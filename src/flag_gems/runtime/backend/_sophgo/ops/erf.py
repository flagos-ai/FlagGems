import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAX_GRID = 64

# Merge: the user's raw-kernel perf structure (grid-cap + no-mask fast path)
# carrying 智源's Abramowitz-Stegun polynomial erf body. The hw `tl.erf`
# intrinsic is broken on the Sophgo TPU (fixed by 智源 commit a8019217), so the
# polynomial is the correctness contract; it must NOT be replaced by tl.erf.


@triton.jit
def erf_kernel_fast(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_fp32 = tl.load(x_ptr + offs).to(tl.float32)
        abs_x = tl.abs(x_fp32)
        tt = 1.0 / (1.0 + 0.3275911 * abs_x)
        tt2 = tt * tt
        tt3 = tt2 * tt
        tt4 = tt3 * tt
        tt5 = tt4 * tt
        poly = (
            0.254829592 * tt
            + (-0.284496736) * tt2
            + 1.421413741 * tt3
            + (-1.453152027) * tt4
            + 1.061405429 * tt5
        )
        result = 1.0 - poly * tl.exp(-abs_x * abs_x)
        tl.store(out_ptr + offs, tl.where(x_fp32 >= 0.0, result, -result))


@triton.jit
def erf_kernel_masked(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x_fp32 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        abs_x = tl.abs(x_fp32)
        tt = 1.0 / (1.0 + 0.3275911 * abs_x)
        tt2 = tt * tt
        tt3 = tt2 * tt
        tt4 = tt3 * tt
        tt5 = tt4 * tt
        poly = (
            0.254829592 * tt
            + (-0.284496736) * tt2
            + 1.421413741 * tt3
            + (-1.453152027) * tt4
            + 1.061405429 * tt5
        )
        result = 1.0 - poly * tl.exp(-abs_x * abs_x)
        tl.store(out_ptr + offs, tl.where(x_fp32 >= 0.0, result, -result), mask=mask)


def _select_bs(n):
    # erf's polynomial keeps ~8 live fp32 tiles; a large BLOCK_SIZE (4096/8192)
    # overflows TPU local memory ("run ppl AddressAssign pass failed"). Cap it
    # so the working set fits in SRAM.
    return 2048


def _dispatch_params(n):
    bs = _select_bs(n)
    num_tiles = math.ceil(n / bs)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    return bs, grid, tpb


def erf(A):
    logger.debug("GEMS ERF (sophgo_tpu)")
    out = torch.empty_like(A)
    n = A.numel()
    bs, grid, tpb = _dispatch_params(n)
    if n % bs == 0:
        erf_kernel_fast[(grid,)](A, out, n, BLOCK_SIZE=bs, TPB=tpb)
    else:
        erf_kernel_masked[(grid,)](A, out, n, BLOCK_SIZE=bs, TPB=tpb)
    return out


def erf_(A):
    logger.debug("GEMS ERF_ (sophgo_tpu)")
    n = A.numel()
    bs, grid, tpb = _dispatch_params(n)
    if n % bs == 0:
        erf_kernel_fast[(grid,)](A, A, n, BLOCK_SIZE=bs, TPB=tpb)
    else:
        erf_kernel_masked[(grid,)](A, A, n, BLOCK_SIZE=bs, TPB=tpb)
    return A
