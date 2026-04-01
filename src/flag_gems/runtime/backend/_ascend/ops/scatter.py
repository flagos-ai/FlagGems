import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

# Hardware specification: Atlas 800T/I A2 on-chip memory capacity is 192KB
BLOCK_SIZE_SUB = 1024


def _compute_block_size(N):
    """Compute BLOCK_SIZE ensuring grid dim (coreDim) <= 65535.
    BLOCK_SIZE is always a multiple of BLOCK_SIZE_SUB."""
    block_size = BLOCK_SIZE_SUB
    while triton.cdiv(N, block_size) > 65535:
        block_size *= 2
    return block_size


# ─── 2D kernel (no-reduce / add) ───────────────────────────────────────────

@libentry()
@triton.jit
def _scatter_2d_kernel(
    src,
    index,
    out,
    src_stride0,
    src_stride1,
    index_stride0,
    index_stride1,
    out_stride0,
    out_stride1,
    index_shape1,
    dim: tl.constexpr,
    dim_stride,
    N,
    IS_ADD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        offset = base + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offset < N

        idx1 = offset % index_shape1
        idx0 = offset // index_shape1

        src_offset = idx0 * src_stride0 + idx1 * src_stride1
        cur_src = tl.load(src + src_offset, mask=mask, other=0)

        index_offset = idx0 * index_stride0 + idx1 * index_stride1
        cur_index = tl.load(index + index_offset, mask=mask, other=0)

        if dim == 0:
            out_offset = cur_index * dim_stride + idx1 * out_stride1
        else:
            out_offset = idx0 * out_stride0 + cur_index * dim_stride

        if IS_ADD:
            tl.atomic_add(out + out_offset, cur_src, mask=mask, sem="relaxed")
        else:
            tl.store(out + out_offset, cur_src, mask=mask)


# ─── 3D kernel (no-reduce / add) ───────────────────────────────────────────

@libentry()
@triton.jit
def _scatter_3d_kernel(
    src,
    index,
    out,
    src_stride0,
    src_stride1,
    src_stride2,
    index_stride0,
    index_stride1,
    index_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    index_shape1,
    index_shape2,
    dim: tl.constexpr,
    dim_stride,
    N,
    IS_ADD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        offset = base + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offset < N

        idx2 = offset % index_shape2
        tmp = offset // index_shape2
        idx1 = tmp % index_shape1
        idx0 = tmp // index_shape1

        src_offset = idx0 * src_stride0 + idx1 * src_stride1 + idx2 * src_stride2
        cur_src = tl.load(src + src_offset, mask=mask, other=0)

        index_offset = (
            idx0 * index_stride0 + idx1 * index_stride1 + idx2 * index_stride2
        )
        cur_index = tl.load(index + index_offset, mask=mask, other=0)

        if dim == 0:
            out_offset = (
                cur_index * dim_stride + idx1 * out_stride1 + idx2 * out_stride2
            )
        elif dim == 1:
            out_offset = (
                idx0 * out_stride0 + cur_index * dim_stride + idx2 * out_stride2
            )
        else:
            out_offset = (
                idx0 * out_stride0 + idx1 * out_stride1 + cur_index * dim_stride
            )

        if IS_ADD:
            tl.atomic_add(out + out_offset, cur_src, mask=mask, sem="relaxed")
        else:
            tl.store(out + out_offset, cur_src, mask=mask)


# ─── Flat kernel (no-reduce / add) for rank != 2,3 ─────────────────────────

@libentry()
@triton.jit
def _scatter_flat_kernel(
    src,
    index,
    out,
    base_offset,
    out_dim_stride,
    N,
    IS_ADD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        offset = base + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offset < N

        cur_src = tl.load(src + offset, mask=mask, other=0)
        cur_index = tl.load(index + offset, mask=mask, other=0)
        base_off = tl.load(base_offset + offset, mask=mask, other=0)

        out_offset = base_off + cur_index * out_dim_stride

        if IS_ADD:
            tl.atomic_add(out + out_offset, cur_src, mask=mask, sem="relaxed")
        else:
            tl.store(out + out_offset, cur_src, mask=mask)


# ─── Serial multiply kernel ────────────────────────────────────────────────
# tl.atomic_cas on float is unreliable on Ascend NPU.
# For reduce="multiply", we use a single-core serial kernel that processes
# one element at a time with scalar load-multiply-store, avoiding atomics.

@libentry()
@triton.jit
def _scatter_mul_serial_kernel(
    src,
    index,
    out,
    src_offsets,
    index_offsets,
    out_base_offsets,
    out_dim_stride,
    N,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    # Single core processes all elements serially in sub-blocks
    for start in range(0, N, BLOCK_SIZE_SUB):
        offset = start + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offset < N

        s_off = tl.load(src_offsets + offset, mask=mask, other=0)
        cur_src = tl.load(src + s_off, mask=mask, other=0)

        i_off = tl.load(index_offsets + offset, mask=mask, other=0)
        cur_index = tl.load(index + i_off, mask=mask, other=0)

        o_base = tl.load(out_base_offsets + offset, mask=mask, other=0)
        out_off = o_base + cur_index * out_dim_stride

        cur_out = tl.load(out + out_off, mask=mask, other=0)
        tl.store(out + out_off, cur_out * cur_src, mask=mask)


# ─── Dispatch logic ────────────────────────────────────────────────────────

def _precompute_offsets(index, src, out, dim, rank):
    """Precompute flat offset arrays for src, index, and out base (excl. dim)."""
    N = index.numel()
    device = index.device
    shape = index.shape

    remaining = torch.arange(N, dtype=torch.int64, device=device)
    src_offsets = torch.zeros(N, dtype=torch.int64, device=device)
    index_offsets = torch.zeros(N, dtype=torch.int64, device=device)
    out_base_offsets = torch.zeros(N, dtype=torch.int64, device=device)

    src_strides = list(src.stride())
    index_strides = list(index.stride())
    out_strides = list(out.stride())

    for i in range(rank - 1, -1, -1):
        coord = remaining % shape[i]
        remaining = remaining // shape[i]
        src_offsets += coord * src_strides[i]
        index_offsets += coord * index_strides[i]
        if i != dim:
            out_base_offsets += coord * out_strides[i]

    return src_offsets, index_offsets, out_base_offsets


def scatter_nd(inp, dim, index, src, out, reduce=None):
    N = index.numel()
    if N == 0:
        return

    is_add = reduce == "add"
    is_mul = reduce == "multiply"
    rank = inp.dim()
    dim_stride = inp.stride(dim)

    # ── multiply: serial single-core kernel ──
    if is_mul:
        src_offsets, index_offsets, out_base_offsets = _precompute_offsets(
            index, src, out, dim, rank
        )
        _scatter_mul_serial_kernel[(1,)](
            src,
            index,
            out,
            src_offsets,
            index_offsets,
            out_base_offsets,
            dim_stride,
            N,
            BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        )
        return

    # ── no-reduce / add: parallel multi-core kernels ──
    BLOCK_SIZE = _compute_block_size(N)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    if rank == 2:
        _scatter_2d_kernel[grid](
            src,
            index,
            out,
            src.stride(0),
            src.stride(1),
            index.stride(0),
            index.stride(1),
            out.stride(0),
            out.stride(1),
            index.shape[1],
            dim,
            dim_stride,
            N,
            IS_ADD=is_add,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        )
    elif rank == 3:
        _scatter_3d_kernel[grid](
            src,
            index,
            out,
            src.stride(0),
            src.stride(1),
            src.stride(2),
            index.stride(0),
            index.stride(1),
            index.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            index.shape[1],
            index.shape[2],
            dim,
            dim_stride,
            N,
            IS_ADD=is_add,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        )
    else:
        # Flat fallback for rank > 3 (and rank == 1)
        index_flat = index.contiguous().view(-1)
        src_flat = src.contiguous().view(-1)

        shape = index.shape
        strides = list(out.stride())
        base_offset = torch.zeros(N, dtype=torch.int64, device=inp.device)
        remaining = torch.arange(N, dtype=torch.int64, device=inp.device)
        for i in range(rank - 1, -1, -1):
            coord = remaining % shape[i]
            remaining = remaining // shape[i]
            if i != dim:
                base_offset += coord * strides[i]

        _scatter_flat_kernel[grid](
            src_flat,
            index_flat,
            out,
            base_offset,
            dim_stride,
            N,
            IS_ADD=is_add,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        )


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_ASCEND SCATTER_")
    dim = dim % inp.dim()

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        inp = inp.contiguous()

    src_restrided = src.as_strided(index.shape, src.stride())
    out = restride_dim(inp, dim, index.shape)

    scatter_nd(inp, dim, index, src_restrided, out, reduce)
    return inp


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_ASCEND SCATTER")
    dim = dim % inp.dim()
    result = inp.clone()

    src_restrided = src.as_strided(index.shape, src.stride())
    out = restride_dim(result, dim, index.shape)

    scatter_nd(result, dim, index, src_restrided, out, reduce)
    return result
