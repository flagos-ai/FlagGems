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

"""Kunlunxin (P800/XPU) override for ``torch.triu_indices``.

The generic implementation in ``flag_gems.ops.triangular_indices`` fills the
output with a Triton kernel that performs *masked strided stores* of a wide
(``BLOCK_SIZE=256``) column vector where only a handful of lanes are valid.
On this TritonXPU backend the ramp kernel is miscompiled: the per-program
scalar index arithmetic (``local_row * (local_row - 1) // 2``,
``matrix_col - row_length``) and the second (column) masked store produce
garbage/leftover column *values* and even corrupt the neighbouring rectangle
region. Observed ``triu_indices(5, 7, -1)`` emitting column indices such as
``4,4,4,...`` instead of ``0,1,2,3,4,5,6`` (row indices stay correct because
they are a broadcast scalar). This is the same class of failure already
handled for the sibling ``tril_indices`` override.

This override keeps the vendor-neutral, overflow-safe host-side validation and
plan computation verbatim (imported from the generic module) and replaces only
the data-generation step with a device-side ``arange`` / ``repeat_interleave``
construction. No CPU/ATen/composite fallback is used: every tensor lives on the
target device and is built from plain allocations plus integer arithmetic. Both
the ramp and rectangle regions are bounded by the true output size, so the
INT64_MAX sparse/far-offset cases (which produce a size-0 or size-1 output) stay
cheap while the overflow/allocation validation still raises exactly as torch
does.
"""

import logging

import torch

from flag_gems.ops.triangular_indices import (
    _make_triu_plan,
    _validate_arguments,
    _validate_output,
)
from flag_gems.runtime import device as runtime_device

logger = logging.getLogger(__name__)


def _fill_rectangle(output, plan, col):
    """Fill the full-width rectangle region (rows all have ``col`` columns)."""
    n = plan.rectangle_rows
    if n == 0:
        return
    dev = output.device
    dtype = output.dtype
    rows = torch.arange(n, device=dev, dtype=torch.int64) + plan.rectangle_row_start
    cols = torch.arange(col, device=dev, dtype=torch.int64)
    row_block = rows.reshape(n, 1).expand(n, col).reshape(-1)
    col_block = cols.reshape(1, col).expand(n, col).reshape(-1)
    off = plan.rectangle_output_offset
    end = off + n * col
    output[0, off:end] = row_block.to(dtype)
    output[1, off:end] = col_block.to(dtype)


def _fill_triu_ramp(output, plan, col):
    """Fill the upper-triangular ramp: row ``r`` has ``first_length - r`` columns.

    For ramp row ``r`` the valid columns are the *trailing* ones of the matrix
    row, i.e. ``[col - length, col - 1]``. The generic kernel emits exactly
    ``column = matrix_col - row_length + element`` for ``element`` in
    ``[0, row_length)``; this reproduces that value stream on device.
    """
    m = plan.ramp_rows
    if m == 0:
        return
    dev = output.device
    dtype = output.dtype
    local_row = torch.arange(m, device=dev, dtype=torch.int64)
    lengths = plan.ramp_first_length - local_row
    total = int(lengths.sum().item())
    starts = torch.cumsum(lengths, 0) - lengths
    matrix_rows = plan.ramp_row_start + local_row
    row_block = torch.repeat_interleave(matrix_rows, lengths)
    starts_rep = torch.repeat_interleave(starts, lengths)
    lengths_rep = torch.repeat_interleave(lengths, lengths)
    within = torch.arange(total, device=dev, dtype=torch.int64) - starts_rep
    col_block = (col - lengths_rep) + within
    off = plan.ramp_output_offset
    end = off + total
    output[0, off:end] = row_block.to(dtype)
    output[1, off:end] = col_block.to(dtype)


def triu_indices(
    row,
    col,
    offset=0,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN TRIU_INDICES")

    row, col, offset, dtype, layout, device, pin_memory = _validate_arguments(
        row, col, offset, dtype, layout, device, pin_memory
    )
    if device is None:
        device = torch.device(runtime_device.name)

    plan = _make_triu_plan(row, col, offset)
    _validate_output(plan, dtype)

    output = torch.empty(
        (2, plan.size),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if plan.size:
        _fill_rectangle(output, plan, col)
        _fill_triu_ramp(output, plan, col)
    return output
