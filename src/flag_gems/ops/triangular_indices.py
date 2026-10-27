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
import operator
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.codegen_config_utils import get_codegen_config

logger = logging.getLogger(__name__)

_INT32_MAX = (1 << 31) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_PORTABLE_GRID_LIMIT = 65535
_BLOCK_SIZE = 256


@dataclass(frozen=True)
class _TriangularPlan:
    size: int
    rectangle_row_start: int = 0
    rectangle_rows: int = 0
    rectangle_output_offset: int = 0
    ramp_row_start: int = 0
    ramp_rows: int = 0
    ramp_first_length: int = 0
    ramp_output_offset: int = 0
    max_row_index: int = -1
    max_col_index: int = -1


def _checked_add_nonnegative(lhs, rhs, description):
    if lhs < 0 or rhs < 0 or lhs > _INT64_MAX - rhs:
        raise RuntimeError(f"{description} exceeds the signed 64-bit range")
    return lhs + rhs


def _checked_mul_nonnegative(lhs, rhs, description):
    if lhs < 0 or rhs < 0 or (lhs != 0 and rhs > _INT64_MAX // lhs):
        raise RuntimeError(f"{description} exceeds the signed 64-bit range")
    return lhs * rhs


def _checked_arithmetic_sum(first, count, step, description):
    """Return the sum without overflowing in an intermediate product."""
    if count == 0:
        return 0

    if first <= 0 or count < 0 or step not in (-1, 1):
        raise RuntimeError(f"invalid arithmetic segment while computing {description}")

    if step == 1:
        last = _checked_add_nonnegative(first, count - 1, description)
    else:
        if count > first:
            raise RuntimeError(
                f"invalid decreasing arithmetic segment while computing {description}"
            )
        last = first - count + 1

    if count & 1:
        # first and last have the same parity when count is odd. Dividing the
        # pair before multiplying keeps the intermediate value in int64.
        half_pair = _checked_add_nonnegative(first // 2, last // 2, description)
        if first & 1:
            half_pair = _checked_add_nonnegative(half_pair, 1, description)
        return _checked_mul_nonnegative(count, half_pair, description)

    pair = _checked_add_nonnegative(first, last, description)
    return _checked_mul_nonnegative(count // 2, pair, description)


def _as_int64(value, name):
    try:
        value = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if value < _INT64_MIN or value > _INT64_MAX:
        raise RuntimeError(f"{name} must fit in a signed 64-bit integer")
    return value


def _validate_arguments(row, col, offset, dtype, layout, device, pin_memory):
    row = _as_int64(row, "row")
    col = _as_int64(col, "col")
    offset = _as_int64(offset, "offset")
    if row < 0:
        raise RuntimeError("row must be non-negative")
    if col < 0:
        raise RuntimeError("col must be non-negative")

    if dtype is None:
        dtype = torch.int64
    if dtype not in (torch.int32, torch.int64):
        raise RuntimeError("dtype must be torch.int32 or torch.int64")

    if layout is None:
        layout = torch.strided
    if layout is not torch.strided:
        raise RuntimeError("only strided layout is supported")

    if device is None:
        device = torch.device(runtime_device.name)
    if pin_memory is None:
        pin_memory = False

    return row, col, offset, dtype, layout, device, pin_memory


def _empty_plan():
    return _TriangularPlan(size=0)


def _make_tril_plan(row, col, offset):
    if row == 0 or col == 0 or offset <= -row:
        return _empty_plan()

    if offset >= col - 1:
        size = _checked_mul_nonnegative(row, col, "tril_indices output size")
        return _TriangularPlan(
            size=size,
            rectangle_rows=row,
            max_row_index=row - 1,
            max_col_index=col - 1,
        )

    active_start = -offset if offset < 0 else 0
    available_rows = row - active_start
    first_length = 1 if offset < 0 else offset + 1
    ramp_rows = min(available_rows, col - first_length)
    rectangle_rows = available_rows - ramp_rows

    ramp_size = _checked_arithmetic_sum(
        first_length, ramp_rows, 1, "tril_indices ramp size"
    )
    rectangle_size = _checked_mul_nonnegative(
        rectangle_rows, col, "tril_indices rectangle size"
    )
    size = _checked_add_nonnegative(
        ramp_size, rectangle_size, "tril_indices output size"
    )

    rectangle_row_start = _checked_add_nonnegative(
        active_start, ramp_rows, "tril_indices rectangle row start"
    )
    if rectangle_rows:
        max_col_index = col - 1
    else:
        max_col_index = first_length + ramp_rows - 2

    return _TriangularPlan(
        size=size,
        rectangle_row_start=rectangle_row_start,
        rectangle_rows=rectangle_rows,
        rectangle_output_offset=ramp_size,
        ramp_row_start=active_start,
        ramp_rows=ramp_rows,
        ramp_first_length=first_length,
        max_row_index=row - 1,
        max_col_index=max_col_index,
    )


def _make_triu_plan(row, col, offset):
    if row == 0 or col == 0 or offset >= col:
        return _empty_plan()

    if offset <= 1 - row:
        size = _checked_mul_nonnegative(row, col, "triu_indices output size")
        return _TriangularPlan(
            size=size,
            rectangle_rows=row,
            max_row_index=row - 1,
            max_col_index=col - 1,
        )

    full_rows = min(row, max(0, 1 - offset))
    # col - offset may overflow int64 for a large negative offset. In that
    # region every remaining matrix row is active, so use row directly.
    active_end = row if offset <= col - row else col - offset
    ramp_rows = active_end - full_rows
    first_length = col - 1 if full_rows else col - offset

    rectangle_size = _checked_mul_nonnegative(
        full_rows, col, "triu_indices rectangle size"
    )
    ramp_size = _checked_arithmetic_sum(
        first_length, ramp_rows, -1, "triu_indices ramp size"
    )
    size = _checked_add_nonnegative(
        rectangle_size, ramp_size, "triu_indices output size"
    )

    return _TriangularPlan(
        size=size,
        rectangle_rows=full_rows,
        ramp_row_start=full_rows,
        ramp_rows=ramp_rows,
        ramp_first_length=first_length,
        ramp_output_offset=rectangle_size,
        max_row_index=active_end - 1,
        max_col_index=col - 1,
    )


def _validate_output(plan, dtype):
    if dtype is torch.int32 and (
        plan.max_row_index > _INT32_MAX or plan.max_col_index > _INT32_MAX
    ):
        raise RuntimeError("triangular index value cannot be represented as int32")

    output_numel = _checked_mul_nonnegative(
        2, plan.size, "triangular_indices output element count"
    )
    element_size = 4 if dtype is torch.int32 else 8
    _checked_mul_nonnegative(
        output_numel, element_size, "triangular_indices output allocation size"
    )


def _launch_metadata(row_count, max_length):
    blocks_per_row = (max_length - 1) // _BLOCK_SIZE + 1
    total_tasks = _checked_mul_nonnegative(
        row_count, blocks_per_row, "triangular_indices kernel task count"
    )

    config = get_codegen_config()
    if config is None or not config.max_grid_size:
        raise RuntimeError("unable to determine the kernel grid limit")
    configured_limit = operator.index(config.max_grid_size[0])
    if configured_limit <= 0:
        raise RuntimeError("kernel grid limit must be positive")

    grid_limit = min(configured_limit, _PORTABLE_GRID_LIMIT, _INT32_MAX)
    grid = min(total_tasks, grid_limit)
    if grid <= 0 or grid > configured_limit:
        raise RuntimeError("triangular_indices kernel grid exceeds the backend limit")
    return blocks_per_row, total_tasks, grid < total_tasks, (grid,)


@libentry()
@triton.jit
def _rectangle_indices_kernel(
    output_ptr,
    output_size: tl.constexpr,
    output_offset: tl.constexpr,
    row_start: tl.constexpr,
    row_count: tl.constexpr,
    matrix_col: tl.constexpr,
    blocks_per_row: tl.constexpr,
    total_tasks: tl.constexpr,
    PERSISTENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)

    if PERSISTENT:
        tasks_per_program = total_tasks // num_programs
        extra_tasks = total_tasks - tasks_per_program * num_programs
        task_start = pid * tasks_per_program + tl.minimum(pid, extra_tasks)
        task_end = task_start + tasks_per_program + tl.where(pid < extra_tasks, 1, 0)
    else:
        task_start = pid
        task_end = pid + 1

    for task_id in range(task_start, task_end):
        local_row = task_id // blocks_per_row
        block_index = task_id - local_row * blocks_per_row
        column = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (local_row < row_count) & (column < matrix_col)
        output_index = output_offset + local_row * matrix_col + column

        tl.store(output_ptr + output_index, row_start + local_row, mask=mask)
        tl.store(output_ptr + output_size + output_index, column, mask=mask)


@libentry()
@triton.jit
def _tril_ramp_indices_kernel(
    output_ptr,
    output_size: tl.constexpr,
    output_offset: tl.constexpr,
    row_start: tl.constexpr,
    row_count: tl.constexpr,
    first_length: tl.constexpr,
    blocks_per_row: tl.constexpr,
    total_tasks: tl.constexpr,
    PERSISTENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)

    if PERSISTENT:
        tasks_per_program = total_tasks // num_programs
        extra_tasks = total_tasks - tasks_per_program * num_programs
        task_start = pid * tasks_per_program + tl.minimum(pid, extra_tasks)
        task_end = task_start + tasks_per_program + tl.where(pid < extra_tasks, 1, 0)
    else:
        task_start = pid
        task_end = pid + 1

    for task_id in range(task_start, task_end):
        local_row = task_id // blocks_per_row
        block_index = task_id - local_row * blocks_per_row
        column = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_length = first_length + local_row

        half_row = local_row // 2
        triangular = tl.where(
            (local_row & 1) == 0,
            half_row * (local_row - 1),
            local_row * half_row,
        )
        row_output_offset = local_row * first_length + triangular
        output_index = output_offset + row_output_offset + column
        mask = (local_row < row_count) & (column < row_length)

        tl.store(output_ptr + output_index, row_start + local_row, mask=mask)
        tl.store(output_ptr + output_size + output_index, column, mask=mask)


@libentry()
@triton.jit
def _triu_ramp_indices_kernel(
    output_ptr,
    output_size: tl.constexpr,
    output_offset: tl.constexpr,
    row_start: tl.constexpr,
    row_count: tl.constexpr,
    first_length: tl.constexpr,
    matrix_col: tl.constexpr,
    blocks_per_row: tl.constexpr,
    total_tasks: tl.constexpr,
    PERSISTENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)

    if PERSISTENT:
        tasks_per_program = total_tasks // num_programs
        extra_tasks = total_tasks - tasks_per_program * num_programs
        task_start = pid * tasks_per_program + tl.minimum(pid, extra_tasks)
        task_end = task_start + tasks_per_program + tl.where(pid < extra_tasks, 1, 0)
    else:
        task_start = pid
        task_end = pid + 1

    for task_id in range(task_start, task_end):
        local_row = task_id // blocks_per_row
        block_index = task_id - local_row * blocks_per_row
        element = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_length = first_length - local_row

        half_row = local_row // 2
        triangular = tl.where(
            (local_row & 1) == 0,
            half_row * (local_row - 1),
            local_row * half_row,
        )
        row_output_offset = local_row * first_length - triangular
        column = matrix_col - row_length + element
        output_index = output_offset + row_output_offset + element
        mask = (local_row < row_count) & (element < row_length)

        tl.store(output_ptr + output_index, row_start + local_row, mask=mask)
        tl.store(output_ptr + output_size + output_index, column, mask=mask)


def _launch_plan(output, plan, col, is_lower):
    with torch_device_fn.device(output.device):
        if plan.rectangle_rows:
            blocks, tasks, persistent, grid = _launch_metadata(plan.rectangle_rows, col)
            _rectangle_indices_kernel[grid](
                output,
                plan.size,
                plan.rectangle_output_offset,
                plan.rectangle_row_start,
                plan.rectangle_rows,
                col,
                blocks,
                tasks,
                persistent,
                _BLOCK_SIZE,
            )

        if plan.ramp_rows:
            if is_lower:
                max_length = _checked_add_nonnegative(
                    plan.ramp_first_length,
                    plan.ramp_rows - 1,
                    "tril_indices maximum row length",
                )
                blocks, tasks, persistent, grid = _launch_metadata(
                    plan.ramp_rows, max_length
                )
                _tril_ramp_indices_kernel[grid](
                    output,
                    plan.size,
                    plan.ramp_output_offset,
                    plan.ramp_row_start,
                    plan.ramp_rows,
                    plan.ramp_first_length,
                    blocks,
                    tasks,
                    persistent,
                    _BLOCK_SIZE,
                )
            else:
                blocks, tasks, persistent, grid = _launch_metadata(
                    plan.ramp_rows, plan.ramp_first_length
                )
                _triu_ramp_indices_kernel[grid](
                    output,
                    plan.size,
                    plan.ramp_output_offset,
                    plan.ramp_row_start,
                    plan.ramp_rows,
                    plan.ramp_first_length,
                    col,
                    blocks,
                    tasks,
                    persistent,
                    _BLOCK_SIZE,
                )


def _triangular_indices(
    row,
    col,
    offset,
    *,
    dtype,
    layout,
    device,
    pin_memory,
    is_lower,
):
    row, col, offset, dtype, layout, device, pin_memory = _validate_arguments(
        row, col, offset, dtype, layout, device, pin_memory
    )
    plan = (
        _make_tril_plan(row, col, offset)
        if is_lower
        else _make_triu_plan(row, col, offset)
    )
    _validate_output(plan, dtype)

    output = torch.empty(
        (2, plan.size),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if plan.size:
        _launch_plan(output, plan, col, is_lower)
    return output


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
    logger.debug("GEMS TRIL_INDICES")
    return _triangular_indices(
        row,
        col,
        offset,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        is_lower=True,
    )


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
    logger.debug("GEMS TRIU_INDICES")
    return _triangular_indices(
        row,
        col,
        offset,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        is_lower=False,
    )
