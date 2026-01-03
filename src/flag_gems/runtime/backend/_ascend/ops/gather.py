import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter import scatter_
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')
# Hardware specification: Atlas 800T/I A2 product's on-chip memory capacity is 192KB
UB_SIZE_BYTES = 192 * 1024


def compute_base_offset(shape, strides, dim):
    # Given shape/strides and a dimension, output a tensor with the size of 'shape',
    # where each position is the offset of the input (excluding the 'dim' dimension)
    idx = torch.arange(int(torch.prod(torch.tensor(shape))), device="cpu")
    coord = torch.empty((len(shape), idx.numel()), dtype=torch.long, device="cpu")
    for i in reversed(range(len(shape))):
        coord[i] = idx % shape[i]
        idx = idx // shape[i]

    offset = torch.zeros_like(coord[0])
    for i in range(len(shape)):
        if i != dim:
            offset += coord[i] * strides[i]
    return offset


@libentry()
@triton.heuristics({"BLOCK_SIZE": lambda args: 4096})
@triton.jit
def _gather_flat_kernel_fixed(
    inp,
    index,
    out,
    base_offset,
    inp_dim_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    cur_index = tl.load(index + offset, mask=mask, other=0)
    base = tl.load(base_offset + offset, mask=mask, other=0)

    inp_offset = base + cur_index * inp_dim_stride

    val = tl.load(inp + inp_offset, mask=mask, other=0)
    tl.store(out + offset, val, mask=mask)


def gather_flat_fixed(inp: torch.Tensor, dim: int, index: torch.Tensor, out=None):
    logger.debug("GEMS_ASCEND GATHER (fixed version)")

    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    N = index.numel()
    dim_stride = inp.stride(dim)
    inp_strided = restride_dim(inp, dim, index.shape)
    if dim == -1:
        dim = inp_strided.dim() - 1
    base_offset = compute_base_offset(index.shape, inp_strided.stride(), dim).to(
        torch.int64
    )
    base_offset = base_offset.npu()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    _gather_flat_kernel_fixed[grid](
        inp_strided,
        index,
        out,
        base_offset,
        dim_stride,
        N,
    )
    return out


@triton.jit
def _gather_high_perf_kernel(
    # Pointers
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

    # Load indices for this row
    idx_ptrs = idx_ptr + row_id * stride_idx_rows + offs_idx * stride_idx_cols
    indices = tl.load(idx_ptrs)

    # Load feature vector
    x_ptrs = x_ptr + row_id * stride_x_rows + offs_x * stride_x_feats
    x_vals = tl.load(x_ptrs)

    # Perform gather
    out_vals = tl.gather(x_vals, indices, 0)

    # Store result
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
        num_indices=num_indices,
        x_size=x_size,
    )
    return out


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

    if is_last_dim and inp.dim() == 2 and total_bytes < UB_SIZE_BYTES:
        return gather_high_perf(inp, index, out)

    return gather_flat_fixed(inp, dim, index, out)


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_ASCEND GATHER BACKWARD")
    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")
