# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_put import _index_put_impl_ as _generic_index_put_impl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _index_put_ordered_rows_kernel(
    inp,
    indices,
    values,
    n_indices: tl.constexpr,
    row_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column = tl.arange(0, BLOCK)
    for update in range(n_indices):
        target_row = tl.load(indices + update)
        offsets = target_row * row_width + column
        value_offsets = update * row_width + column
        value = tl.load(values + value_offsets, mask=column < row_width)
        tl.store(inp + offsets, value, mask=column < row_width)


def _can_use_last_write_rows(inp, indices, values, accumulate):
    if accumulate or len(indices) != 1:
        return False
    index = indices[0]
    if index is None or index.dtype == torch.bool or index.ndim != 1:
        return False
    if not inp.is_contiguous() or not index.is_contiguous() or not values.is_contiguous():
        return False
    if inp.ndim < 2 or values.ndim != inp.ndim:
        return False
    if values.shape[0] != index.numel() or tuple(values.shape[1:]) != tuple(inp.shape[1:]):
        return False
    return index.numel() <= 64 and inp.numel() // inp.shape[0] <= 1024


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    indices = list(indices)
    if not _can_use_last_write_rows(inp, indices, values, accumulate):
        return _generic_index_put_impl(inp, indices, values, accumulate, unsafe)

    row_width = inp.numel() // inp.shape[0]
    block = min(1024, triton.next_power_of_2(row_width))
    with torch_device_fn.device(inp.device):
        _index_put_ordered_rows_kernel[(1,)](
            inp,
            indices[0],
            values,
            indices[0].numel(),
            row_width=row_width,
            BLOCK=block,
        )
    return inp
