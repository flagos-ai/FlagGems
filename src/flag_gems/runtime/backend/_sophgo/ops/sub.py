import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.sub import sub as _generic_sub
from flag_gems.ops.sub import sub_ as _generic_sub_

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64

# Raw-kernel sophgo fast path (grid-cap + no-mask) for the common case:
# same-shape contiguous floating-point tensor - tensor. Everything else
# (scalar operand, broadcasting, non-contiguous, integer) falls back to the
# generic pointwise_dynamic op.
#
# Why scalar operands fall back: the PPL backend mis-lowers a `tensor - scalar`
# kernel for fp16/bf16 output (`apply arith to ppl conversion failed` -> NaN),
# whereas tensor - tensor lowers fine and the generic kernel handles the scalar
# case correctly. Compute is in fp32 (store downcasts).


def _dispatch(n):
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    return grid, tpb


def _can_fast(*tensors):
    ref = tensors[0]
    return all(
        isinstance(t, torch.Tensor)
        and t.is_contiguous()
        and t.shape == ref.shape
        and t.is_floating_point()
        for t in tensors
    )


@triton.jit
def sub_tt_kernel_fast(
    x_ptr, y_ptr, out_ptr, alpha, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, x - y * alpha)


@triton.jit
def sub_tt_kernel_masked(
    x_ptr, y_ptr, out_ptr, alpha, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, x - y * alpha, mask=mask)


def _run_tt(a, b, alpha, out):
    n = a.numel()
    grid, tpb = _dispatch(n)
    k = sub_tt_kernel_fast if n % BLOCK_SIZE == 0 else sub_tt_kernel_masked
    k[(grid,)](a, b, out, alpha, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out


def sub(A, B, *, alpha=1):
    logger.debug("GEMS SUB (sophgo_tpu)")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and _can_fast(A, B):
        return _run_tt(A, B, alpha, torch.empty_like(A))
    return _generic_sub(A, B, alpha=alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS SUB_ (sophgo_tpu)")
    if isinstance(B, torch.Tensor) and _can_fast(A, B):
        return _run_tt(A, B, alpha, A)
    return _generic_sub_(A, B, alpha=alpha)
