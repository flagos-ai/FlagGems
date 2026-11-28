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
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.device_info import get_device_capability

if torch_device_fn.is_available() and get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + row_g_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    else:
        y_q = tl.clamp((y / _absmax) * fp8_max, fp8_min, fp8_max).to(
            y_q_ptr.dtype.element_ty
        )

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    group_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + group_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += group_id * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    else:
        y_q = tl.clamp((y / _absmax) * fp8_max, fp8_min, fp8_max).to(
            y_q_ptr.dtype.element_ty
        )

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    programs_per_row = groups_per_row // NGROUPS

    pid = tl.program_id(0)
    row = pid // programs_per_row
    program_id = pid % programs_per_row

    start_group = program_id * NGROUPS
    start_gid = row * groups_per_row + start_group

    group_ids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row * y_row_stride
        + start_group * group_size
        + group_ids[:, None] * group_size
        + cols[None, :]
    )
    mask = cols[None, :] < group_size

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    else:
        y_q = tl.clamp((y / _absmax[:, None]) * fp8_max, fp8_min, fp8_max).to(
            y_q_ptr.dtype.element_ty
        )
    output_offsets = (
        start_gid * group_size + group_ids[:, None] * group_size + cols[None, :]
    )

    tl.store(y_q_ptr + output_offsets, y_q, mask=mask)
    tl.store(y_s_ptr + start_gid + group_ids, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    programs_per_row = groups_per_row // NGROUPS

    pid = tl.program_id(0)
    row = pid // programs_per_row
    program_id = pid % programs_per_row

    start_group = program_id * NGROUPS
    start_gid = row * groups_per_row + start_group

    group_ids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row * y_row_stride
        + start_group * group_size
        + group_ids[:, None] * group_size
        + cols[None, :]
    )
    mask = cols[None, :] < group_size

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    else:
        y_q = tl.clamp((y / _absmax[:, None]) * fp8_max, fp8_min, fp8_max).to(
            y_q_ptr.dtype.element_ty
        )
    output_offsets = (
        start_gid * group_size + group_ids[:, None] * group_size + cols[None, :]
    )
    scale_offsets = (start_group + group_ids) * y_s_col_stride + row

    tl.store(y_q_ptr + output_offsets, y_q, mask=mask)
    tl.store(y_s_ptr + scale_offsets, y_s)


@triton.jit
def _per_token_group_quant_fp8_flat(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    num_groups,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    """Flat quantization: treat the whole tensor as num_groups contiguous groups.

    Groups are contiguous in memory (x rows are contiguous and group_size divides
    the last dim), so the divisibility constraint on groups_per_row is unnecessary
    and NGROUPS can be raised to any value to shrink the program count.
    """
    pid = tle.program_id(0) + tle.program_id(1) * tle.num_programs(0)
    group_ids = pid * NGROUPS + tl.arange(0, NGROUPS)
    group_mask = group_ids < num_groups

    cols = tl.arange(0, BLOCK)
    col_mask = cols < group_size
    offsets = group_ids[:, None] * group_size + cols[None, :]
    mask = group_mask[:, None] & col_mask[None, :]

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = absmax / fp8_max
    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    else:
        y_q = tl.clamp((y / absmax[:, None]) * fp8_max, fp8_min, fp8_max).to(
            y_q_ptr.dtype.element_ty
        )

    tl.store(y_q_ptr + offsets, y_q, mask=mask)
    tl.store(y_s_ptr + group_ids, y_s, mask=group_mask)


def _groups_per_program(x: torch.Tensor, group_size: int) -> int:
    groups_per_row = x.shape[-1] // group_size
    for groups in (8, 4, 2):
        if groups_per_row % groups == 0:
            return groups
    return 1


# Element budget of one flat-path program tile (NGROUPS * BLOCK). Measured on
# gcu400: fatter tiles amortize the per-program launch cost until the tile
# exceeds the DSM budget available to a single program.
_FLAT_TILE_ELEMENTS = 32768
# A tile with a single group row degenerates the row-wise reduction into a
# scalar chain, so keep at least this many group rows per tile.
_FLAT_MIN_GROUP_ROWS = 8
# Upper bound on the number of programs issued by the flat path; below this the
# per-program overhead dominates even though each program is tiny.
_FLAT_MAX_PROGRAMS = 8
# Group count from which the fatter tile pays off. Below it the legacy 16-group
# tile already produces enough programs and measuring a gain is not possible
# (the whole launch is shorter than the measurement noise floor).
_FLAT_TILE_MIN_GROUPS = 1024
# Legacy tile used for small inputs.
_FLAT_LEGACY_GROUPS = 16


def _flat_groups_per_program(num_groups: int, block: int) -> int:
    """Groups processed by one flat-path program.

    The flat path sees the tensor as ``num_groups`` contiguous groups, so the
    tile shape is fully under host control. Take the largest power-of-two tile
    that stays below ``_FLAT_TILE_ELEMENTS`` and shrink it until at most
    ``_FLAT_MAX_PROGRAMS`` programs are needed: a few fat programs are much
    cheaper than many thin ones on gcu400.
    """
    if num_groups <= 32:
        # Tiny workloads are launch-bound: one group per program keeps the
        # device busy instead of serializing the work into a single program.
        return 1
    if num_groups < _FLAT_TILE_MIN_GROUPS:
        return _FLAT_LEGACY_GROUPS
    tile_groups = max(_FLAT_MIN_GROUP_ROWS, _FLAT_TILE_ELEMENTS // block)
    programs_target = triton.next_power_of_2(
        max(1, triton.cdiv(num_groups, _FLAT_MAX_PROGRAMS))
    )
    return max(_FLAT_MIN_GROUP_ROWS, min(tile_groups, programs_target))


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_ENFLAME PER_TOKEN_GROUP_QUANT_FP8")
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    num_groups = x.numel() // group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)

    # With contiguous rows (x is a contiguous tensor) and a row-major scale, the
    # whole tensor is num_groups contiguous groups, so the flat kernel can move
    # several groups per program.
    if x.is_contiguous() and not column_major_scales:
        groups_per_program = _flat_groups_per_program(num_groups, block)
        flat_blocks = triton.cdiv(num_groups, groups_per_program)
        flat_grid_y = triton.cdiv(flat_blocks, 65535)
        _per_token_group_quant_fp8_flat[
            (triton.cdiv(flat_blocks, flat_grid_y), flat_grid_y)
        ](
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            num_groups=num_groups,
            BLOCK=block,
            NGROUPS=groups_per_program,
            num_warps=8,
            num_stages=1,
        )
        return x_q, x_s

    groups_per_program = _groups_per_program(x, group_size)
    grid = (num_groups // groups_per_program,)

    if column_major_scales:
        if groups_per_program > 1:
            kernel = _per_token_group_quant_fp8_colmajor_vec
            kernel[grid](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=block,
                NGROUPS=groups_per_program,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            kernel = _per_token_group_quant_fp8_colmajor
            kernel[grid](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=1,
            )
    elif groups_per_program > 1:
        kernel = _per_token_group_quant_fp8_vec
        kernel[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            BLOCK=block,
            NGROUPS=groups_per_program,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        kernel = _per_token_group_quant_fp8
        kernel[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )

    return x_q, x_s
