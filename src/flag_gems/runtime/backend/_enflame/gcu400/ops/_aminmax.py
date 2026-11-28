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

from flag_gems.ops._aminmax import _aminmax as _generic_aminmax
from flag_gems.ops._aminmax import _aminmax_out as _generic_aminmax_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["_aminmax", "_aminmax_out"]

# GCU400 reduction affinity parameters (determined by A/B sweeps):
# - 48 CTA = 24 SIP × 2 programs/SP at full occupancy, doubling memory parallelism over 24 CTA;
# - Accumulator uses the input's native dtype: min/max are lossless, and vector op lane throughput is higher than fp32
#   accumulation by 2x (512M fp16: 2106 GB/s vs 1198 GB/s);
# - The grid shrinks with the data size, so small tensors don't pay accumulator initialization cost for empty CTA.
_GCU400_AMINMAX_USE_SPECIALIZED = True
_GCU400_AMINMAX_GRID_CAP = 48
_GCU400_AMINMAX_BLOCK = 16384
_GCU400_AMINMAX_NUM_WARPS = 4

# (BLOCK, GRID upper bound) per dtype. Both tiles and accumulators consume on-chip DSM (896KB total budget),
# so the accumulator width determines how large a tile can be and how many programs can co-reside:
# - fp16 native accumulation (2B accumulator): 64KB tile + 48 CTA is optimal;
# - bf16 is promoted to fp32 accumulation by the backend (4B accumulator): only a large tile + 1 program/SP;
# - fp32 / integer accumulation: 32KB tile + 48 CTA is optimal.
_GCU400_AMINMAX_LAUNCH_TABLE = (
    (tl.float16, 16384, 48),
    (tl.float32, 8192, 48),
)
_GCU400_AMINMAX_BF16_LAUNCH = (32768, 24)


@triton.jit
def _aminmax_tile_min(acc, values, IS_FLOAT: tl.constexpr):
    """NaN-propagating per-lane min (matching torch.aminmax)."""
    if IS_FLOAT:
        return tl.minimum(acc, values, propagate_nan=tl.PropagateNan.ALL)
    return tl.minimum(acc, values)


@triton.jit
def _aminmax_tile_max(acc, values, IS_FLOAT: tl.constexpr):
    if IS_FLOAT:
        return tl.maximum(acc, values, propagate_nan=tl.PropagateNan.ALL)
    return tl.maximum(acc, values)


@libentry()
@triton.jit
def _gcu400_aminmax_partial_kernel(
    inp,
    min_partial,
    max_partial,
    num_elements,
    num_full_tiles,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """Stage 1: GRID CTAs stride over the whole tensor, each writing one partial extremum.

    The caller guarantees ``GRID <= num_full_tiles``, so every CTA has at least one full tile to
    initialize its accumulator, and no +-inf / extreme-value fill constant is needed (bf16 cannot
    build one). Full tiles run mask-free; only the sub-BLOCK tail is masked, handled by CTA 0.
    """
    program_id = tl.program_id(0)
    lane_offset = tl.arange(0, BLOCK)

    first_offset = program_id * BLOCK + lane_offset
    acc_min = tl.load(inp + first_offset).to(ACC_DTYPE)
    acc_max = acc_min

    for tile_index in range(program_id + GRID, num_full_tiles, GRID):
        element_offset = tile_index * BLOCK + lane_offset
        values = tl.load(inp + element_offset).to(ACC_DTYPE)
        acc_min = _aminmax_tile_min(acc_min, values, IS_FLOAT)
        acc_max = _aminmax_tile_max(acc_max, values, IS_FLOAT)

    tail_start = num_full_tiles * BLOCK
    if tail_start < num_elements:
        if program_id == 0:
            tail_offset = tail_start + lane_offset
            tail_mask = tail_offset < num_elements
            tail_values = tl.load(inp + tail_offset, mask=tail_mask, other=0).to(
                ACC_DTYPE
            )
            # Lanes outside the mask keep their existing accumulated value to avoid introducing out-of-bounds garbage.
            acc_min = _aminmax_tile_min(
                acc_min, tl.where(tail_mask, tail_values, acc_min), IS_FLOAT
            )
            acc_max = _aminmax_tile_max(
                acc_max, tl.where(tail_mask, tail_values, acc_max), IS_FLOAT
            )

    partial_min = tl.min(acc_min)
    partial_max = tl.max(acc_max)
    if IS_FLOAT:
        # tl.min / tl.max drop NaN: if this CTA has seen a NaN, pollute the output with NaN.
        has_nan = tl.sum((acc_min != acc_min).to(tl.int32)) > 0
        nan_value = float("nan")
        partial_min = tl.where(has_nan, nan_value, partial_min)
        partial_max = tl.where(has_nan, nan_value, partial_max)

    tl.store(min_partial + program_id, partial_min.to(min_partial.type.element_ty))
    tl.store(max_partial + program_id, partial_max.to(max_partial.type.element_ty))


@libentry()
@triton.jit
def _gcu400_aminmax_serial_kernel(
    inp,
    min_out,
    max_out,
    num_elements,
    num_full_tiles,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """Single-CTA direct version: one sweep writes the scalar output, dropping stage 2."""
    lane_offset = tl.arange(0, BLOCK)
    # First block with masking: when the input is shorter than one BLOCK, clamp out-of-bounds lanes to element 0
    # (duplicate elements do
    # not affect min/max), so there is still no need to construct dtype extreme constants.
    first_valid = lane_offset < num_elements
    acc_min = tl.load(inp + tl.where(first_valid, lane_offset, 0)).to(ACC_DTYPE)
    acc_max = acc_min

    for tile_index in range(1, num_full_tiles, 1):
        element_offset = tile_index * BLOCK + lane_offset
        values = tl.load(inp + element_offset).to(ACC_DTYPE)
        acc_min = _aminmax_tile_min(acc_min, values, IS_FLOAT)
        acc_max = _aminmax_tile_max(acc_max, values, IS_FLOAT)

    tail_start = num_full_tiles * BLOCK
    if tail_start < num_elements:
        tail_offset = tail_start + lane_offset
        tail_mask = tail_offset < num_elements
        tail_values = tl.load(inp + tail_offset, mask=tail_mask, other=0).to(ACC_DTYPE)
        acc_min = _aminmax_tile_min(
            acc_min, tl.where(tail_mask, tail_values, acc_min), IS_FLOAT
        )
        acc_max = _aminmax_tile_max(
            acc_max, tl.where(tail_mask, tail_values, acc_max), IS_FLOAT
        )

    final_min = tl.min(acc_min)
    final_max = tl.max(acc_max)
    if IS_FLOAT:
        has_nan = tl.sum((acc_min != acc_min).to(tl.int32)) > 0
        nan_value = float("nan")
        final_min = tl.where(has_nan, nan_value, final_min)
        final_max = tl.where(has_nan, nan_value, final_max)

    tl.store(min_out, final_min.to(min_out.type.element_ty))
    tl.store(max_out, final_max.to(max_out.type.element_ty))


@libentry()
@triton.jit
def _gcu400_aminmax_final_kernel(
    min_partial,
    max_partial,
    min_out,
    max_out,
    GRID: tl.constexpr,
    BLOCK_FINAL: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """Stage 2: one CTA reduces the partial extrema from stage 1 (at most GRID elements)."""
    partial_offset = tl.arange(0, BLOCK_FINAL)
    valid_mask = partial_offset < GRID
    # Fill masked-out lanes with the partial value at index 0: it is always valid and does not change the min/max
    # result,
    # so no dtype extreme constant needs to be constructed.
    fill_min = tl.load(min_partial)
    fill_max = tl.load(max_partial)
    partial_mins = tl.load(
        min_partial + partial_offset, mask=valid_mask, other=fill_min
    ).to(ACC_DTYPE)
    partial_maxs = tl.load(
        max_partial + partial_offset, mask=valid_mask, other=fill_max
    ).to(ACC_DTYPE)

    final_min = tl.min(partial_mins)
    final_max = tl.max(partial_maxs)
    if IS_FLOAT:
        # If any partial result is NaN, both outputs must be NaN (matching torch._aminmax).
        has_nan = (
            tl.sum((partial_mins != partial_mins).to(tl.int32))
            + tl.sum((partial_maxs != partial_maxs).to(tl.int32))
        ) > 0
        nan_value = float("nan")
        final_min = tl.where(has_nan, nan_value, final_min)
        final_max = tl.where(has_nan, nan_value, final_max)

    tl.store(min_out, final_min.to(min_out.type.element_ty))
    tl.store(max_out, final_max.to(max_out.type.element_ty))


def _acc_dtype_for(dtype: torch.dtype):
    """Accumulator dtype (triton type).

    fp16 accumulates at native width (min/max lose no precision and vector lane throughput
    doubles); bf16's ``tl.minimum`` is promoted to fp32 by the GCU backend, so the accumulator
    must follow fp32; narrow integers are promoted to int32.
    """
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.int64:
        return tl.int64
    if dtype in (torch.float32, torch.bfloat16):
        return tl.float32
    return tl.int32


def _pick_launch_shape(num_elements: int, input_dtype, acc_dtype):
    """Picks (BLOCK, GRID) from dtype and total element count.

    GRID must not exceed the full-tile count, otherwise some CTA gets no full tile and cannot init.
    """
    block, grid_cap = _GCU400_AMINMAX_BLOCK, _GCU400_AMINMAX_GRID_CAP
    if input_dtype == torch.bfloat16:
        block, grid_cap = _GCU400_AMINMAX_BF16_LAUNCH
    else:
        for (
            candidate_acc,
            candidate_block,
            candidate_cap,
        ) in _GCU400_AMINMAX_LAUNCH_TABLE:
            if candidate_acc == acc_dtype:
                block, grid_cap = candidate_block, candidate_cap
                break
    num_full_tiles = num_elements // block
    grid = min(max(num_full_tiles, 1), grid_cap)
    return block, grid


def _gcu400_aminmax_run(inp, out0=None, out1=None):
    """GCU400-specific driver: handles input/view and ``.out`` semantics, then launches the reduction."""
    if inp.numel() == 0:
        # ATen rejects reduction on empty tensors rather than returning positive/negative infinity.
        raise RuntimeError(
            "aminmax(): cannot compute aminmax over an empty dimension as the "
            "operation has no identity."
        )

    # Kernels index by flat offsets, so non-contiguous / expanded views must first be materialized into dense copies.
    inp = inp.contiguous()
    num_elements = inp.numel()
    dtype = inp.dtype
    is_float = inp.is_floating_point()

    if out0 is None:
        min_out = torch.empty([], dtype=dtype, device=inp.device)
        max_out = torch.empty([], dtype=dtype, device=inp.device)
    else:
        for name, output in (("out0", out0), ("out1", out1)):
            if output.dtype != dtype:
                raise RuntimeError(
                    f"Expected out tensor to have dtype {dtype}, but got "
                    f"{output.dtype} instead"
                )
            if output.device != inp.device:
                raise RuntimeError(
                    f"Expected out tensor to have device {inp.device}, but got "
                    f"{output.device} instead"
                )
        # ATen resizes a shape-mismatched output in place to a scalar and preserves object aliasing.
        if out0.shape != torch.Size([]):
            out0.resize_(())
        if out1.shape != torch.Size([]):
            out1.resize_(())
        min_out, max_out = out0, out1

    acc_dtype = _acc_dtype_for(dtype)
    block, grid = _pick_launch_shape(num_elements, dtype, acc_dtype)
    num_full_tiles = num_elements // block

    if grid == 1:
        # When the data is small enough for a single CTA pass, write the scalar output directly, saving one kernel
        # launch.
        with torch_device_fn.device(inp.device):
            _gcu400_aminmax_serial_kernel[(1,)](
                inp,
                min_out,
                max_out,
                num_elements,
                num_full_tiles,
                block,
                acc_dtype,
                IS_FLOAT=is_float,
                num_warps=_GCU400_AMINMAX_NUM_WARPS,
            )
        return min_out, max_out

    min_partial = torch.empty((grid,), dtype=dtype, device=inp.device)
    max_partial = torch.empty((grid,), dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        _gcu400_aminmax_partial_kernel[(grid,)](
            inp,
            min_partial,
            max_partial,
            num_elements,
            num_full_tiles,
            block,
            grid,
            acc_dtype,
            IS_FLOAT=is_float,
            num_warps=_GCU400_AMINMAX_NUM_WARPS,
        )
        _gcu400_aminmax_final_kernel[(1,)](
            min_partial,
            max_partial,
            min_out,
            max_out,
            grid,
            triton.next_power_of_2(grid),
            acc_dtype,
            IS_FLOAT=is_float,
            num_warps=_GCU400_AMINMAX_NUM_WARPS,
        )
    return min_out, max_out


def _aminmax(inp):
    """gcu400 ``_aminmax``.

    GCU does not support fp64: torch_gcu replaces fp64 aten with float, but fp64 NaN detection
    in the generated Triton kernel cannot be legalized. Reduce over an fp32 view, then restore the input dtype.
    """
    logger.debug("GEMS_ENFLAME _AMINMAX")
    if inp.dtype == torch.float64:
        min_out, max_out = _gcu400_aminmax_run(inp.to(torch.float32))
        return min_out.to(torch.float64), max_out.to(torch.float64)
    if not _GCU400_AMINMAX_USE_SPECIALIZED:
        return _generic_aminmax(inp)
    return _gcu400_aminmax_run(inp)


def _aminmax_out(inp, *, out0, out1):
    logger.debug("GEMS_ENFLAME _AMINMAX_OUT")
    if inp.dtype == torch.float64:
        min_out, max_out = _aminmax(inp)
        out0.resize_(())
        out1.resize_(())
        out0.copy_(min_out)
        out1.copy_(max_out)
        return out0, out1
    if not _GCU400_AMINMAX_USE_SPECIALIZED:
        return _generic_aminmax_out(inp, out0=out0, out1=out1)
    return _gcu400_aminmax_run(inp, out0=out0, out1=out1)
