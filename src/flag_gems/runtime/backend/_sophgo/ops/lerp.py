import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.lerp import lerp_scalar as _generic_lerp_scalar
from flag_gems.ops.lerp import lerp_scalar_ as _generic_lerp_scalar_
from flag_gems.ops.lerp import lerp_tensor as _generic_lerp_tensor
from flag_gems.ops.lerp import lerp_tensor_ as _generic_lerp_tensor_

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64

# lerp(input, end, weight) = input + weight * (end - input), computed with the
# same numerically-stable two-branch form torch uses (head when |w|<0.5, tail
# otherwise) so extrapolation (|w|>=0.5) matches the reference.
#
# This is the user's raw-kernel sophgo pattern (grid-cap + no-mask fast path),
# replacing the generic pointwise_dynamic dispatch. The flat-index kernels only
# handle same-shape contiguous tensors; anything else (broadcasting,
# non-contiguous, integer weight tensor) falls back to the generic op.


def _can_fast(*tensors):
    ref = tensors[0]
    return all(
        t.is_contiguous() and t.shape == ref.shape and t.is_floating_point()
        for t in tensors
    )


def _dispatch(n):
    num_tiles = triton.cdiv(n, BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = triton.cdiv(num_tiles, grid)
    return grid, tpb


# ---- scalar weight ---------------------------------------------------------
@triton.jit
def _lerp_scalar_head_fast(
    in_ptr, end_ptr, out_ptr, w, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(in_ptr + offs).to(tl.float32)
        b = tl.load(end_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, a + w * (b - a))


@triton.jit
def _lerp_scalar_head_masked(
    in_ptr, end_ptr, out_ptr, w, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        b = tl.load(end_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, a + w * (b - a), mask=mask)


@triton.jit
def _lerp_scalar_tail_fast(
    in_ptr, end_ptr, out_ptr, w, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(in_ptr + offs).to(tl.float32)
        b = tl.load(end_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, b - (b - a) * (1.0 - w))


@triton.jit
def _lerp_scalar_tail_masked(
    in_ptr, end_ptr, out_ptr, w, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        b = tl.load(end_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, b - (b - a) * (1.0 - w), mask=mask)


def _run_scalar(input, end, weight, out):
    n = input.numel()
    grid, tpb = _dispatch(n)
    head = weight < 0.5
    if n % BLOCK_SIZE == 0:
        k = _lerp_scalar_head_fast if head else _lerp_scalar_tail_fast
    else:
        k = _lerp_scalar_head_masked if head else _lerp_scalar_tail_masked
    k[(grid,)](input, end, out, float(weight), n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out


def lerp_scalar(input, end, weight):
    logger.debug("GEMS LERP SCALAR (sophgo_tpu)")
    if not _can_fast(input, end):
        return _generic_lerp_scalar(input, end, weight)
    return _run_scalar(input, end, weight, torch.empty_like(input))


def lerp_scalar_(input, end, weight):
    logger.debug("GEMS LERP SCALAR_ (sophgo_tpu)")
    if not _can_fast(input, end):
        return _generic_lerp_scalar_(input, end, weight)
    return _run_scalar(input, end, weight, input)


# ---- tensor weight ---------------------------------------------------------
@triton.jit
def _lerp_tensor_fast(
    in_ptr, end_ptr, w_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(in_ptr + offs).to(tl.float32)
        b = tl.load(end_ptr + offs).to(tl.float32)
        w = tl.load(w_ptr + offs).to(tl.float32)
        r = tl.where(tl.abs(w) < 0.5, a + w * (b - a), b - (b - a) * (1.0 - w))
        tl.store(out_ptr + offs, r)


@triton.jit
def _lerp_tensor_masked(
    in_ptr, end_ptr, w_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        b = tl.load(end_ptr + offs, mask=mask).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
        r = tl.where(tl.abs(w) < 0.5, a + w * (b - a), b - (b - a) * (1.0 - w))
        tl.store(out_ptr + offs, r, mask=mask)


def _run_tensor(input, end, weight, out):
    n = input.numel()
    grid, tpb = _dispatch(n)
    k = _lerp_tensor_fast if n % BLOCK_SIZE == 0 else _lerp_tensor_masked
    k[(grid,)](input, end, weight, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out


def lerp_tensor(input, end, weight):
    logger.debug("GEMS LERP TENSOR (sophgo_tpu)")
    if not _can_fast(input, end, weight):
        return _generic_lerp_tensor(input, end, weight)
    return _run_tensor(input, end, weight, torch.empty_like(input))


def lerp_tensor_(input, end, weight):
    logger.debug("GEMS LERP TENSOR_ (sophgo_tpu)")
    if not _can_fast(input, end, weight):
        return _generic_lerp_tensor_(input, end, weight)
    return _run_tensor(input, end, weight, input)
