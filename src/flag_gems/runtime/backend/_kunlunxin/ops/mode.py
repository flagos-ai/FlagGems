import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy
from .sort import radix_sort_low_mem

logger = logging.getLogger(__name__)

ModeResult = namedtuple("mode", ["values", "indices"])


@libentry()
@triton.jit(do_not_specialize=["columns"])
def _mode_sorted_rows_kernel(
    sorted_values,
    sorted_indices,
    output_values,
    output_indices,
    columns,
):
    row = tl.program_id(0)
    row_offset = row * columns
    current_value = tl.load(sorted_values + row_offset)
    current_index = tl.load(sorted_indices + row_offset)
    best_value = current_value
    best_index = current_index
    current_count = 1
    best_count = 1

    column = 1
    while column < columns:
        value = tl.load(sorted_values + row_offset + column)
        index = tl.load(sorted_indices + row_offset + column)
        same_value = value == current_value
        current_count = tl.where(same_value, current_count + 1, 1)
        current_value = tl.where(same_value, current_value, value)
        current_index = index
        better = current_count > best_count
        best_count = tl.where(better, current_count, best_count)
        best_value = tl.where(better, current_value, best_value)
        best_index = tl.where(better, current_index, best_index)
        column += 1

    tl.store(output_values + row, best_value)
    tl.store(output_indices + row, best_index)


@libentry()
@triton.jit
def _mode_fill_first(x_ptr, out_v_ptr, out_i_ptr, RS: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    v = tl.load(x_ptr + pid * RS)
    tl.store(out_v_ptr + pid, v)
    tl.store(out_i_ptr + pid, 0)


def _normalize_dim(dim, ndim):
    if ndim == 0:
        if dim in (0, -1):
            return 0
    elif -ndim <= dim < ndim:
        return dim % ndim
    raise IndexError(
        f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
    )


def _mode_impl(inp, dim, keepdim):
    if inp.ndim == 0:
        values = inp.clone()
        indices = torch.zeros((), dtype=torch.long, device=inp.device)
        return ModeResult(values=values, indices=indices)

    dim = _normalize_dim(dim, inp.ndim)
    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1

    if N == 0:
        if M != 0:
            raise IndexError(
                f"mode(): Expected reduction dim {dim} to have non-zero size."
            )
        values = torch.empty(keepdim_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(keepdim_shape, dtype=torch.long, device=inp.device)
        if not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return ModeResult(values=values, indices=indices)

    values = torch.empty(keepdim_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(keepdim_shape, dtype=torch.long, device=inp.device)

    if M == 0:
        if not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return ModeResult(values=values, indices=indices)

    flat_values = values.reshape(M)
    flat_indices = indices.reshape(M)

    if dim != inp.ndim - 1:
        view = torch.movedim(inp, dim, -1)
        rows = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
        if not tle_copy(view, rows):
            torch.ops.aten._copy_from(view, rows, False)
    else:
        rows = inp.reshape(M, N)

    if N == 1:
        with torch_device_fn.device(inp.device):
            _mode_fill_first[(M,)](rows, flat_values, flat_indices, RS=N, N=N)
    else:
        sorted_v, sorted_i = radix_sort_low_mem(rows, 4, False)
        if sorted_v.dtype == torch.bfloat16:
            sorted_v = sorted_v.to(torch.float32)
        with torch_device_fn.device(inp.device):
            _mode_sorted_rows_kernel[(M,)](
                sorted_v, sorted_i, flat_values, flat_indices, N
            )

    if not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return ModeResult(values=values, indices=indices)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MODE")
    return _mode_impl(inp, dim, keepdim)
