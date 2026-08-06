import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sort import sort_stable

logger = logging.getLogger(__name__)


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
        # ATen mode returns the last occurrence for the selected value.
        current_index = index
        better = current_count > best_count
        best_count = tl.where(better, current_count, best_count)
        best_value = tl.where(better, current_value, best_value)
        best_index = tl.where(better, current_index, best_index)
        column += 1

    tl.store(output_values + row, best_value)
    tl.store(output_indices + row, best_index)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MODE")
    if inp.ndim == 0:
        if dim not in (-1, 0):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
            )
        Mode = namedtuple("mode", ["values", "indices"])
        return Mode(
            values=inp.clone(),
            indices=torch.zeros((), dtype=torch.int64, device=inp.device),
        )
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )

    dim %= inp.ndim
    columns = inp.shape[dim]
    if columns == 0:
        raise IndexError(f"mode(): Expected reduction dim {dim} to have non-zero size.")

    remaining_dims = tuple(index for index in range(inp.ndim) if index != dim)
    work = inp.permute(remaining_dims + (dim,)).contiguous()
    rows = math.prod(work.shape[:-1])
    work = work.reshape(rows, columns)
    sort_work = work.to(torch.float32) if work.dtype == torch.bfloat16 else work
    sorted_values = torch.empty_like(sort_work)
    sorted_indices = torch.empty_like(sort_work, dtype=torch.int64)
    row_chunk = 256
    for row_start in range(0, rows, row_chunk):
        row_end = min(row_start + row_chunk, rows)
        chunk_values, chunk_indices = sort_stable(
            sort_work[row_start:row_end], stable=True, dim=1, descending=False
        )
        sorted_values[row_start:row_end].copy_(chunk_values)
        sorted_indices[row_start:row_end].copy_(chunk_indices)

    output_values = torch.empty(rows, dtype=inp.dtype, device=inp.device)
    output_indices = torch.empty(rows, dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        _mode_sorted_rows_kernel[(rows,)](
            sorted_values,
            sorted_indices,
            output_values,
            output_indices,
            columns,
            isCloseVectorization=True,
            isCloseUnrollControl=True,
            buffer_size_limit=2048,
        )

    output_shape = list(inp.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    Mode = namedtuple("mode", ["values", "indices"])
    return Mode(
        values=output_values.reshape(output_shape),
        indices=output_indices.reshape(output_shape),
    )
