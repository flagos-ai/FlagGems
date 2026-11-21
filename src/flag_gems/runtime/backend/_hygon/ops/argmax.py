import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.argmax import (
    argmax as _generic_argmax,
    argmax_kernel_1,
    argmax_kernel_2,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min


@libentry()
@triton.jit
def argmax_lastdim_kernel(
    inp,
    out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)

    dtype: tl.constexpr = inp.dtype.element_ty
    if dtype == tl.float16 or dtype == tl.bfloat16:
        dtype = tl.float32
    minimum = get_dtype_min(dtype)
    sentinel: tl.constexpr = 0x7FFFFFFF

    max_value = tl.full((), minimum, dtype)
    max_index = tl.full((), 0, tl.int32)
    nan_index = tl.full((), sentinel, tl.int32)
    for start in tl.range(0, N, BLOCK):
        col = start + lane
        valid = col < N
        values = tl.load(
            inp + row.to(tl.int64) * N + col,
            mask=valid,
            other=minimum,
        ).to(dtype)
        is_nan = valid & (values != values)
        block_nan_index = tl.min(tl.where(is_nan, col, sentinel), 0)
        nan_index = tl.minimum(nan_index, block_nan_index)
        values = tl.where(is_nan, minimum, values)
        local_value, local_index = tl.max(
            values,
            0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        take = local_value > max_value
        max_value = tl.where(take, local_value, max_value)
        max_index = tl.where(take, start + local_index, max_index)
    max_index = tl.where(nan_index != sentinel, nan_index, max_index)
    tl.store(out + row, max_index)


@triton.jit
def argmax_zero_kernel(out, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offsets, 0, mask=offsets < n_elements)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    numel = inp.numel()
    if numel == 0:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    rank = inp.ndim
    if dim is not None:
        if rank == 0 or dim < -rank or dim >= rank:
            return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
        dim %= rank
        shape = inp.shape
        if shape[dim] == 1:
            if keepdim:
                out_shape = shape[:dim] + (1,) + shape[dim + 1 :]
            else:
                out_shape = shape[:dim] + shape[dim + 1 :]
            out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
            block = 1024
            with torch_device_fn.device(inp.device):
                argmax_zero_kernel[(triton.cdiv(numel, block),)](
                    out,
                    numel,
                    block,
                    num_warps=4,
                )
            return out

    fast_dtype = inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not fast_dtype:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if dim is None:
        size = inp.numel()
        if not inp.is_contiguous() or size > 1 << 24:
            return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
        mid_dtype = inp.dtype if dtype is None else dtype
        block = 2048
        mid_size = triton.cdiv(size, block)
        mid_value = torch.empty((mid_size,), dtype=mid_dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        out_shape = [1] * inp.ndim if keepdim else []
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        with torch_device_fn.device(inp.device):
            argmax_kernel_1[(mid_size,)](
                inp,
                mid_value,
                mid_index,
                size,
                block,
                num_warps=4,
            )
            argmax_kernel_2[(1,)](
                mid_value,
                mid_index,
                out,
                mid_size,
                triton.next_power_of_2(mid_size),
            )
        return out

    if rank == 0 or dim < -rank or dim >= rank:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    dim %= rank

    shape = inp.shape
    n = shape[dim]
    out_shape = list(shape)
    out_shape[dim] = 1
    if not keepdim:
        del out_shape[dim]

    m = math.prod(shape[:dim])
    k = math.prod(shape[dim + 1 :])
    if (
        not inp.is_contiguous()
        or k != 1
        or n < 4096
        or n > 0x7FFFFFFF
        or m < 64
    ):
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    if n == 4096 and m >= 2048:
        block = 4096
        num_warps = 4
    else:
        block = 8192
        num_warps = 8
    with torch_device_fn.device(inp.device):
        argmax_lastdim_kernel[(m,)](
            inp,
            out,
            n,
            block,
            num_warps=num_warps,
        )
    return out
