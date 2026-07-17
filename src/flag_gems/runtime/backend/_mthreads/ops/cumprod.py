import logging

import torch
import triton
import triton.language as tl
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SCAN_BLOCK = 1024


@triton.jit
def _reduce_mul(left, right):
    return left * right


@libentry()
@triton.jit
def _local_scan_kernel(
    inp,
    local_prefix,
    block_products,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offsets = row * N + cols

    compute_dtype: tl.constexpr = local_prefix.type.element_ty
    values = tl.load(inp + offsets, mask=mask, other=1).to(compute_dtype)
    prefix = tl.cumprod(values, axis=0)
    product = tl.reduce(values, axis=0, combine_fn=_reduce_mul)

    tl.store(local_prefix + offsets, prefix, mask=mask)
    tl.store(block_products + row * NUM_BLOCKS + block, product)


@libentry()
@triton.jit
def _scan_block_products_kernel(
    block_products,
    block_prefix,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    running = tl.full((), 1, dtype=block_prefix.type.element_ty)

    for start in range(0, NUM_BLOCKS, BLOCK_SIZE):
        block_offsets = start + offsets
        mask = block_offsets < NUM_BLOCKS
        values = tl.load(
            block_products + row * NUM_BLOCKS + block_offsets,
            mask=mask,
            other=1,
        )
        scanned = running * tl.cumprod(values, axis=0)
        tl.store(
            block_prefix + row * NUM_BLOCKS + block_offsets,
            scanned,
            mask=mask,
        )
        running *= tl.reduce(values, axis=0, combine_fn=_reduce_mul)


@libentry()
@triton.jit
def _apply_block_prefix_kernel(
    local_prefix,
    block_prefix,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offsets = row * N + cols
    previous = tl.load(
        block_prefix + row * NUM_BLOCKS + block - 1,
        mask=block > 0,
        other=1,
    )
    values = tl.load(local_prefix + offsets, mask=mask, other=1)
    tl.store(local_prefix + offsets, values * previous, mask=mask)


def _compute_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if is_integer_dtype(dtype):
        return torch.int64
    return dtype


def _cumprod_contiguous(inp, dim):
    moved = torch.movedim(inp, dim, -1).contiguous()
    N = moved.shape[-1]
    if moved.numel() == 0:
        return moved

    rows = moved.numel() // N
    compute_dtype = _compute_dtype(inp.dtype)
    local_prefix = torch.empty_like(moved, dtype=compute_dtype)
    num_blocks = triton.cdiv(N, _SCAN_BLOCK)
    block_products = torch.empty(
        (rows, num_blocks), dtype=compute_dtype, device=inp.device
    )
    block_prefix = torch.empty_like(block_products)
    block_scan_size = min(_SCAN_BLOCK, triton.next_power_of_2(max(1, num_blocks)))

    with torch_device_fn.device(inp.device):
        _local_scan_kernel[(rows, num_blocks)](
            moved,
            local_prefix,
            block_products,
            N,
            num_blocks,
            _SCAN_BLOCK,
            num_warps=8,
            num_stages=1,
        )
        _scan_block_products_kernel[(rows,)](
            block_products,
            block_prefix,
            num_blocks,
            block_scan_size,
            num_warps=8 if block_scan_size >= 1024 else 4,
            num_stages=1,
        )
        if num_blocks > 1:
            _apply_block_prefix_kernel[(rows, num_blocks)](
                local_prefix,
                block_prefix,
                N,
                num_blocks,
                _SCAN_BLOCK,
                num_warps=8,
                num_stages=1,
            )

    original_order = torch.movedim(local_prefix, -1, dim)
    return original_order.reshape(inp.shape)


def cumprod_(inp, dim, *, dtype=None):
    logger.debug("GEMS_MTHREADS CUMPROD_")
    if dtype is not None and dtype != inp.dtype:
        raise RuntimeError(
            "Bad in-place call: input tensor dtype and output tensor dtype should match"
        )
    if is_boolean_dtype(inp.dtype):
        raise NotImplementedError(
            "In-place cumprod is not supported for boolean tensors"
        )
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )

    dim %= inp.ndim
    if inp.numel() == 0:
        return inp

    result = _cumprod_contiguous(inp, dim)
    inp.copy_(result)
    return inp
