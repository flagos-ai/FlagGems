import math

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024


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


def _ascend_launch_select_backward(grad, input_sizes, dim, index, out=None):
    dim = int(dim)
    index = int(index)

    sizes = tuple(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim
        if dim < 0:
            raise ValueError("invalid dim")
    elif dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = int(sizes[dim])

    if index < 0:
        index += dim_size
        if index < 0:
            raise ValueError("index out of range")
    elif index >= dim_size:
        raise ValueError("index out of range")

    if out is None:
        out = torch.empty(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        if out.shape != sizes:
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    # Ascend 910B: CANN's native zero + strided copy is faster than launching
    # a Triton scatter kernel for this op.
    out.zero_()
    out.select(dim, index).copy_(grad)
    return out


def _launch_select_backward(grad, input_sizes, dim, index, out=None):
    dim = int(dim)
    index = int(index)

    sizes = tuple(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim
        if dim < 0:
            raise ValueError("invalid dim")
    elif dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = int(sizes[dim])

    if index < 0:
        index += dim_size
        if index < 0:
            raise ValueError("index out of range")
    elif index >= dim_size:
        raise ValueError("index out of range")

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1 :]) if dim < ndim - 1 else 1

    outer_size = int(outer_size)
    inner_size = int(inner_size)

    total = outer_size * inner_size

    if out is None:
        out = torch.empty(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        if out.shape != sizes:
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    out.zero_()

    if total == 0:
        return out

    if not grad.is_contiguous():
        grad = grad.contiguous()

    grad_view = grad.view(total)

    grid = (triton.cdiv(total, _BLOCK_SIZE),)
    dim_stride = dim_size * inner_size

    _select_backward_kernel[grid](
        grad_view,
        out,
        total,
        inner_size,
        dim_stride,
        index,
        BLOCK=_BLOCK_SIZE,
    )

    return out


def select_backward(grad, input_sizes, dim, index, out=None):
    if grad.device.type == "npu":
        return _ascend_launch_select_backward(grad, input_sizes, dim, index, out=out)
    return _launch_select_backward(grad, input_sizes, dim, index, out=out)
