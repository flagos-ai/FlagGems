# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import torch

from flag_gems.ops.row_stack import (
    _allocate,
    _cast_tensors,
    _check_out,
    _prepare,
    _strides_for,
)
from flag_gems.ops.row_stack import row_stack as _generic_row_stack
from flag_gems.ops.row_stack import row_stack_out as _generic_row_stack_out

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# Every input contributes one dense run of ``rows * row_width`` elements to the
# output, so the operator is a list of block copies. Below this size the extra
# launch per input dominates and the generic pointwise kernel, which covers a
# group of inputs with a single launch, stays in charge.
_TLE_MIN_BYTES = 1 << 20


def _tle_stack(tensors, out, row_width, dtype):
    """Copy every input block into ``out`` with one transfer per input.

    Returns False, leaving ``out`` partially written and for the caller to
    overwrite, when a transfer cannot be expressed.
    """
    flat = out.view(-1)
    total_rows = 0
    for tensor in tensors:
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        rows = tensor.shape[0]
        if rows == 0:
            continue
        start = total_rows * row_width
        chunk = flat[start : start + rows * row_width].view(tensor.shape)
        if not tle_copy(tensor, chunk):
            return False
        total_rows += rows
    return True


def _dense_contiguous(tensors, memory_format, row_width, shape, strides, dtype, device):
    """Allocate the output and fill it with DMA block copies, or return None."""
    if memory_format != torch.contiguous_format or strides[0] != row_width:
        return None
    total_elements = sum(t.numel() for t in tensors)
    if total_elements * tensors[0].element_size() < _TLE_MIN_BYTES:
        return None
    out = _allocate(shape, strides, dtype, device)
    if not _tle_stack(
        _cast_tensors(tensors, dtype, memory_format), out, row_width, dtype
    ):
        return None
    return out


def row_stack(tensors: list):
    logger.debug("GEMS_KUNLUNXIN ROW_STACK")

    prepared, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    if row_width == 0 or output_shape[0] == 0:
        return _allocate(
            output_shape,
            _strides_for(output_shape, memory_format),
            dtype,
            prepared[0].device,
        )

    strides = _strides_for(output_shape, memory_format)
    out = _dense_contiguous(
        prepared,
        memory_format,
        row_width,
        output_shape,
        strides,
        dtype,
        prepared[0].device,
    )
    if out is not None:
        return out
    return _generic_row_stack(tensors)


def row_stack_out(tensors: list, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ROW_STACK_OUT")

    prepared, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    if (
        row_width != 0
        and output_shape[0] != 0
        and memory_format == torch.contiguous_format
        and list(out.shape) == list(output_shape)
        and out.dtype == dtype
        and out.is_contiguous()
        and out.numel() * out.element_size() >= _TLE_MIN_BYTES
    ):
        # The dtype/overlap contract of ``cat_out`` still has to hold; the shape
        # already matches, so this validates only and never resizes.
        _check_out(out, output_shape, dtype, memory_format, prepared)
        cast = _cast_tensors(prepared, dtype, memory_format)
        if _tle_stack(cast, out, row_width, dtype):
            return out
    return _generic_row_stack_out(tensors, out=out)
