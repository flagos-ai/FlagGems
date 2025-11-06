import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter import scatter_
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


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


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_ASCEND GATHER")
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    x = gather_flat_fixed(inp, dim, index)
    return x


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_ASCEND GATHER BACKWARD")
    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")
