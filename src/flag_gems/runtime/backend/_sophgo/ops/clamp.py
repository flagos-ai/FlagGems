import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64


@triton.jit
def clamp_kernel_fast(
    x_ptr, out_ptr, mini, maxi, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, tl.minimum(maxi, tl.maximum(mini, x)))


@triton.jit
def clamp_kernel_masked(
    x_ptr, out_ptr, mini, maxi, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, tl.minimum(maxi, tl.maximum(mini, x)), mask=mask)


@triton.jit
def clamp_min_kernel_fast(
    x_ptr, out_ptr, mini, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs, tl.maximum(mini, tl.load(x_ptr + offs)))


@triton.jit
def clamp_min_kernel_masked(
    x_ptr, out_ptr, mini, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(
            out_ptr + offs,
            tl.maximum(mini, tl.load(x_ptr + offs, mask=mask)),
            mask=mask,
        )


@triton.jit
def clamp_max_kernel_fast(
    x_ptr, out_ptr, maxi, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs, tl.minimum(maxi, tl.load(x_ptr + offs)))


@triton.jit
def clamp_max_kernel_masked(
    x_ptr, out_ptr, maxi, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(
            out_ptr + offs,
            tl.minimum(maxi, tl.load(x_ptr + offs, mask=mask)),
            mask=mask,
        )


def _clamp_scalar(A, out, mini, maxi):
    n = A.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    if mini is not None and maxi is not None:
        if n % BLOCK_SIZE == 0:
            clamp_kernel_fast[(grid,)](
                A, out, mini, maxi, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb
            )
        else:
            clamp_kernel_masked[(grid,)](
                A, out, mini, maxi, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb
            )
    elif mini is not None:
        if n % BLOCK_SIZE == 0:
            clamp_min_kernel_fast[(grid,)](
                A, out, mini, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb
            )
        else:
            clamp_min_kernel_masked[(grid,)](
                A, out, mini, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb
            )
    else:
        if n % BLOCK_SIZE == 0:
            clamp_max_kernel_fast[(grid,)](
                A, out, maxi, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb
            )
        else:
            clamp_max_kernel_masked[(grid,)](
                A, out, maxi, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb
            )


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS CLAMP (sophgo_tpu)")
    # Tensor bounds: fall back to generic (rare, complex broadcasting)
    if isinstance(mini, torch.Tensor) or isinstance(maxi, torch.Tensor):
        from flag_gems.ops.clamp import clamp_tensor

        return clamp_tensor(A, mini, maxi)
    out = torch.empty_like(A)
    _clamp_scalar(A, out, mini, maxi)
    return out


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS CLAMP_ (sophgo_tpu)")
    if isinstance(mini, torch.Tensor) or isinstance(maxi, torch.Tensor):
        from flag_gems.ops.clamp import clamp_tensor_

        return clamp_tensor_(A, mini, maxi)
    _clamp_scalar(A, A, mini, maxi)
    return A
