import logging

import torch
import triton
import triton.language as tl

from .scatter import scatter_
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')
# Hardware specification: Atlas 800T/I A2 product's on-chip memory capacity is 192KB
UB_SIZE_BYTES = 192 * 1024
# Max elements per sub-iteration to fit in UB (192KB)
# Each element needs ~7 int64 temporaries = 56 bytes, with multi-buffer ~112 bytes
# 192KB / 112 ≈ 1750, round down to power of 2
BLOCK_SIZE_SUB = 1024


@libentry()
@triton.jit
def _gather_2d_kernel(
    inp,
    index,
    out,
    inp_stride0,
    inp_stride1,
    index_stride0,
    index_stride1,
    out_stride0,
    out_stride1,
    index_shape1,
    dim: tl.constexpr,
    dim_stride,
    N,
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

        index_offset = idx0 * index_stride0 + idx1 * index_stride1
        cur_index = tl.load(index + index_offset, mask=mask, other=0)

        if dim == 0:
            inp_offset = idx1 * inp_stride1 + cur_index * dim_stride
        else:
            inp_offset = idx0 * inp_stride0 + cur_index * dim_stride

        val = tl.load(inp + inp_offset, mask=mask, other=0)

        out_offset = idx0 * out_stride0 + idx1 * out_stride1
        tl.store(out + out_offset, val, mask=mask)


@libentry()
@triton.jit
def _gather_3d_kernel(
    inp,
    index,
    out,
    inp_stride0,
    inp_stride1,
    inp_stride2,
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

        index_offset = idx0 * index_stride0 + idx1 * index_stride1 + idx2 * index_stride2
        cur_index = tl.load(index + index_offset, mask=mask, other=0)

        if dim == 0:
            inp_offset = cur_index * dim_stride + idx1 * inp_stride1 + idx2 * inp_stride2
        elif dim == 1:
            inp_offset = idx0 * inp_stride0 + cur_index * dim_stride + idx2 * inp_stride2
        else:
            inp_offset = idx0 * inp_stride0 + idx1 * inp_stride1 + cur_index * dim_stride

        val = tl.load(inp + inp_offset, mask=mask, other=0)

        out_offset = idx0 * out_stride0 + idx1 * out_stride1 + idx2 * out_stride2
        tl.store(out + out_offset, val, mask=mask)


@libentry()
@triton.jit
def _gather_flat_kernel(
    inp,
    index,
    out,
    base_offset,
    inp_dim_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        offset = base + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offset < N

        cur_index = tl.load(index + offset, mask=mask, other=0)
        base_off = tl.load(base_offset + offset, mask=mask, other=0)

        inp_offset = base_off + cur_index * inp_dim_stride

        val = tl.load(inp + inp_offset, mask=mask, other=0)
        tl.store(out + offset, val, mask=mask)


@triton.jit
def _gather_high_perf_kernel(
    x_ptr,
    idx_ptr,
    out_ptr,
    stride_x_rows,
    stride_x_feats,
    stride_idx_rows,
    stride_idx_cols,
    stride_out_rows,
    stride_out_cols,
    num_indices: tl.constexpr,
    x_size: tl.constexpr,
):
    row_id = tl.program_id(0)

    offs_idx = tl.arange(0, num_indices)
    offs_x = tl.arange(0, x_size)

    idx_ptrs = idx_ptr + row_id * stride_idx_rows + offs_idx * stride_idx_cols
    indices = tl.load(idx_ptrs)

    x_ptrs = x_ptr + row_id * stride_x_rows + offs_x * stride_x_feats
    x_vals = tl.load(x_ptrs)

    out_vals = tl.gather(x_vals, indices, 0)

    out_ptrs = out_ptr + row_id * stride_out_rows + offs_idx * stride_out_cols
    tl.store(out_ptrs, out_vals)


def gather_high_perf(inp: torch.Tensor, index: torch.Tensor, out=None):
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    x_size = inp.shape[-1]
    num_indices = index.shape[-1]
    num_rows = index.shape[0]

    grid = (num_rows,)
    _gather_high_perf_kernel[grid](
        inp,
        index,
        out,
        inp.stride(0),
        inp.stride(1),
        index.stride(0),
        index.stride(1),
        out.stride(0),
        out.stride(1),
        num_indices=triton.next_power_of_2(num_indices),
        x_size=triton.next_power_of_2(x_size),
    )
    return out


def _compute_block_size(N):
    """Compute BLOCK_SIZE ensuring grid dim (coreDim) <= 65535.
    BLOCK_SIZE is always a multiple of BLOCK_SIZE_SUB."""
    block_size = BLOCK_SIZE_SUB
    while triton.cdiv(N, block_size) > 65535:
        block_size *= 2
    return block_size


def gather_nd(inp: torch.Tensor, dim: int, index: torch.Tensor, out=None):
    """General gather using rank-specific Triton kernels."""
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    N = index.numel()
    inp_strided = restride_dim(inp, dim, index.shape)
    dim_stride = inp.stride(dim)
    rank = inp.dim()

    BLOCK_SIZE = _compute_block_size(N)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

    if rank == 2:
        _gather_2d_kernel[grid](
            inp_strided,
            index,
            out,
            inp_strided.stride(0),
            inp_strided.stride(1),
            index.stride(0),
            index.stride(1),
            out.stride(0),
            out.stride(1),
            index.shape[1],
            dim=dim,
            dim_stride=dim_stride,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        )
    elif rank == 3:
        _gather_3d_kernel[grid](
            inp_strided,
            index,
            out,
            inp_strided.stride(0),
            inp_strided.stride(1),
            inp_strided.stride(2),
            index.stride(0),
            index.stride(1),
            index.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            index.shape[1],
            index.shape[2],
            dim=dim,
            dim_stride=dim_stride,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
        )
    else:
        _gather_fallback(inp_strided, dim, index, out, dim_stride, N, BLOCK_SIZE)

    return out


def _gather_fallback(inp_strided, dim, index, out, dim_stride, N, BLOCK_SIZE):
    """Fallback for ranks > 3: compute base_offset on CPU."""
    shape = index.shape
    strides = inp_strided.stride()
    idx = torch.arange(N, device="cpu")
    coord = torch.empty((len(shape), N), dtype=torch.long, device="cpu")
    for i in reversed(range(len(shape))):
        coord[i] = idx % shape[i]
        idx = idx // shape[i]
    offset = torch.zeros_like(coord[0])
    for i in range(len(shape)):
        if i != dim:
            offset += coord[i] * strides[i]
    base_offset = offset.to(torch.int64).npu()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    _gather_flat_kernel[grid](
        inp_strided,
        index,
        out,
        base_offset,
        dim_stride,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
    )


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_ASCEND GATHER")
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    dim = dim % inp.dim()
    is_last_dim = dim == inp.dim() - 1

    total_bytes = (
        inp.size(-1) * inp.element_size()
        + index.size(-1) * index.element_size()
        + index.size(-1) * inp.element_size()
    )

    # For 2D last-dim cases that fit in UB with reasonable num_rows, use tl.gather path
    if (is_last_dim and inp.dim() == 2
            and total_bytes < UB_SIZE_BYTES
            and index.shape[0] <= 4096):
        return gather_high_perf(inp, index, out)

    return gather_nd(inp, dim, index, out)


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_ASCEND GATHER BACKWARD")
    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")
