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
import triton
import triton.language as tl

from flag_gems.ops.triangular_indices import (
    _make_tril_plan,
    _validate_arguments,
    _validate_output,
)
from flag_gems.runtime import device as runtime_device

logger = logging.getLogger(__name__)

_IOTA_BLOCK = 1024


@triton.jit
def _iota_kernel(out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tl.store(out_ptr + offs, offs, mask=offs < n)


def _iota(n, device):
    """Device-side ``torch.arange(n, dtype=int64)`` replacement (no fallback)."""
    out = torch.empty(n, device=device, dtype=torch.int64)
    if n > 0:
        grid = (triton.cdiv(n, _IOTA_BLOCK),)
        _iota_kernel[grid](out, n, BLOCK=_IOTA_BLOCK)
    return out


def _fill_rectangle(output, plan, col):
    """Fill the full-width rectangle region (rows all have ``col`` columns)."""
    n = plan.rectangle_rows
    if n == 0:
        return
    dev = output.device
    dtype = output.dtype
    rows = _iota(n, dev) + plan.rectangle_row_start
    cols = _iota(col, dev)
    row_block = rows.reshape(n, 1).expand(n, col).reshape(-1)
    col_block = cols.reshape(1, col).expand(n, col).reshape(-1)
    off = plan.rectangle_output_offset
    end = off + n * col
    output[0, off:end] = row_block.to(dtype)
    output[1, off:end] = col_block.to(dtype)


def _fill_tril_ramp(output, plan):
    """Fill the lower-triangular ramp: row ``r`` has ``first_length + r`` columns."""
    m = plan.ramp_rows
    if m == 0:
        return
    dev = output.device
    dtype = output.dtype
    first_length = plan.ramp_first_length
    local_row = _iota(m, dev)
    lengths = first_length + local_row
    # Closed-form prefix sum of the arithmetic series ``first_length + r``:
    #   total      = m*first_length + m*(m-1)/2
    #   starts[r]  = r*first_length + r*(r-1)/2   (exclusive prefix sum)
    # avoids torch.cumsum / a .sum().item() device sync entirely.
    total = m * first_length + m * (m - 1) // 2
    starts = local_row * first_length + local_row * (local_row - 1) // 2
    matrix_rows = plan.ramp_row_start + local_row
    row_block = torch.repeat_interleave(matrix_rows, lengths)
    starts_rep = torch.repeat_interleave(starts, lengths)
    col_block = _iota(total, dev) - starts_rep
    off = plan.ramp_output_offset
    end = off + total
    output[0, off:end] = row_block.to(dtype)
    output[1, off:end] = col_block.to(dtype)


def tril_indices(
    row,
    col,
    offset=0,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN TRIL_INDICES")

    row, col, offset, dtype, layout, device, pin_memory = _validate_arguments(
        row, col, offset, dtype, layout, device, pin_memory
    )
    if device is None:
        device = torch.device(runtime_device.name)

    plan = _make_tril_plan(row, col, offset)
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
        _fill_tril_ramp(output, plan)
    return output
