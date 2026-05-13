import logging
import math

import torch
import triton
import triton.language as tl

_BLOCK = 1024
logger = logging.getLogger(__name__)


@triton.jit
def _select_backward_kernel(
    grad_ptr,
    out_ptr,
    total: tl.constexpr,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    index: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    outer = offs // inner_size
    inner = offs % inner_size

    vals = tl.load(grad_ptr + offs, mask=mask)
    out_offset = outer * dim_stride + index * inner_size + inner

    tl.store(out_ptr + out_offset, vals, mask=mask)


def select_backward(grad, input_sizes, dim, index, out=None):
    logger.debug("GEMS SELECT_BACKWARD")
    dim = int(dim)
    index = int(index)
    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]

    if index < 0:
        index += dim_size
    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    if out is None:
        out = torch.zeros(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

        out.zero_()

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1 :]) if dim < ndim - 1 else 1
    total = outer_size * inner_size

    grad_view = grad.contiguous().view(outer_size, inner_size)
    dim_stride = dim_size * inner_size

    n_elements = outer_size * inner_size
    grid = (triton.cdiv(n_elements, _BLOCK),)

    _select_backward_kernel[grid](
        grad_view,
        out,
        total,
        inner_size,
        dim_stride,
        index,
        BLOCK=_BLOCK,
    )
    return out
