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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_NPU_BLOCK_SIZE = 256
# The sum/mean backward pass is pure gather (read grad slices, write all of grad_input) and a
# single tile that is too small cannot saturate memory bandwidth. A/B with identical inputs on
# (1024,1024,1024) fp16 (48 CTAs, bandwidth = bytes read plus written over latency):
#   BLOCK=1024  STAGES=4 -> 404.5ms (10.6 GB/s)  <- original config
#   BLOCK=4096  STAGES=4 -> 608.2ms ( 7.1 GB/s)
#   BLOCK=8192  STAGES=2 -> 428.0ms (10.0 GB/s)
#   BLOCK=16384 STAGES=2 -> 242.1ms (17.7 GB/s)  <- best, 1.67x
#   BLOCK=32768 STAGES=2 -> 960.8ms ( 4.5 GB/s, warps=8 regresses)
# So this gather uses 16384 with a 2-stage pipeline, not the 1024 of the generic path.
_UNIFORM_BACKWARD_GATHER_BLOCK = 16384
_UNIFORM_BACKWARD_GATHER_STAGES = 2
# Lowered from 1<<20 (generic) to 1<<12 on GCU400: the generic fallback's
# host-side lengths->offsets prep (_prepare_common) costs ~4.8ms, so
# mid-size uniform tensors (e.g. (2048,256) = 512K elements) were ~4x
# SLOWER than torch on that path; the uniform kernel handles them in
# ~0.02ms. The uniformity check itself is cached per lengths tensor.
_UNIFORM_FAST_PATH_MIN_NUMEL = 1 << 12
_UNIFORM_KERNEL_MAX_SEGMENT_LENGTH = 1024
_UNIFORM_LENGTHS_CACHE = {}
# Fixed Grid on GCU400 (48 CTAs). The generic per-element grids
# (output.numel() / data.numel()) exceed the 65535 grid.x hardware
# limit for large tensors; striding from a fixed grid keeps every
# launch within the limit regardless of tensor size.
_GRID_SIZE = 48
# Uniform forward always launches its own kernels (since 2026-11-18 the host
# no longer falls back to torch.sum or other aten compute entry points):
# under uniform lengths the flat layout of data is exactly a (rows, L, K)
# perfect blocking, so three fast paths -- whole segment (K == 1) / whole
# row (K > 1) / chunked two-pass (overlong segments) -- use only 1D
# contiguous loads with i64 scalar base addresses (the P17 family proved
# that 2D row-major tiles wrap 32-bit addressing once the covered byte
# range reaches 2 GiB and abort the device; flat 1D is measured safe even
# at 4 GiB).

_SUPPORTED_REDUCES = ("sum", "mean", "max", "min", "prod")
_SUPPORTED_DATA_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _prod(shape):
    return math.prod(shape) if shape else 1


def _get_block_size(device):
    return _NPU_BLOCK_SIZE if device.type == "npu" else _BLOCK_SIZE


def _get_uniform_backward_tile_config(device, inner_size, reduce, dtype):
    if device.type == "npu":
        return 1, 16 if inner_size > 1 else 1
    # Alternating A/B retests: 16 rows/CTA is about 2x faster than the original 4 rows on GCU400
    # (5-case total latency 17.57 ms -> 8.26 ms); prod also uses BLOCK_K=64 uniformly,
    # avoiding 256's DSM overflow; the 3D tile total is tightened by BLOCK_N at the call site.
    return 16, 64 if inner_size > 1 else 1


# Element budget for uniform-forward tiles. A/B measurements (2026-11-18,
# after de-where-ization): 32768 elements are optimal on the rows path
# (fp16 (4096,4096) 187->151us); anything larger hits the DSM limit
# (917504-byte hard cap; a 65536-element 2D tile already exceeds it).
_UNIFORM_TILE_ELEMS = 32768
# Chunk size (K == 1 path) when a segment exceeds the single-tile budget
# or its length is not a power of two. fp16 2GiB 9160->1219us (1.76TB/s),
# fp32 4GiB 3785->1848us (2.26TB/s).
_UNIFORM_SPLIT_CHUNK = 65536
# Element budget of the (GROUP, POS, INNER) 3D tile on the K > 1 whole-row
# path (tiered by dtype).
_UNIFORM_INNER_BUDGET_2B = 65536
_UNIFORM_INNER_BUDGET_4B = 131072
# Inner-width cap for the K > 1 generic loop path.
_UNIFORM_LOOP_BLOCK_K = 2048
# dsm hard cap for a single GCU400 CTA (917504 B); 3/4 of it is the (ROWS, SEGMENT) tile budget.
_DSM_BUDGET = 917504 * 3 // 4


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def _prev_pow2(value):
    return 1 << (value.bit_length() - 1)


def _uniform_rows_per_task(segment_length, elem_bytes):
    """Bounds the ROWS of the (ROWS, SEGMENT) 2-D tile so dsm does not overflow (GCU400).

    Measured (2026-11-19, launching _segment_reduce_uniform_rows_kernel directly): ROWS=8192 with
    SEGMENT=4 needs 1048576 B of dsm (sum/mean/prod) and 1179648 B (max/min, the extra 16 B per
    row being the NaN mask), while ROWS=4096 with SEGMENT=4 and ROWS=4096 with SEGMENT=8 both
    compile -- i.e. dsm is billed as "max(SEGMENT, 32) lanes per row (short segments rounded up
    to 32) + 16 B", regardless of whether SEGMENT fills a whole vector. The old code used
    _UNIFORM_TILE_ELEMS // SEGMENT directly as ROWS, so short segments (L=4/8) pushed ROWS to
    8192 and shapes like (2048,256) failed the whole segment_reduce benchmark with
    dsm OutOfResources.
    """
    per_row = max(segment_length, 32) * elem_bytes + 16
    rows_limit = max(1, _DSM_BUDGET // per_row)
    return _prev_pow2(rows_limit)


def _uniform_bcast_tile_config(segment_length, inner_size):
    """Region-aligned tile for the broadcast backward kernels.

    Returns (TILE, tiles_per_region), or None when the shape is not tile
    aligned. Both L and K must be powers of two so that one of TILE/INNER
    divides the other, which keeps every tile inside a single region and makes
    the channel pattern (lane % INNER) exact.
    """
    if not (_is_pow2(segment_length) and _is_pow2(inner_size)):
        return None
    region_len = segment_length * inner_size
    tile = min(_UNIFORM_BCAST_TILE, region_len)
    if inner_size < tile:
        tile = (tile // inner_size) * inner_size
    return tile, region_len // tile


def _uniform_stats_config(segment_length, inner_size):
    """(BLOCK_K, POS, num_k_tiles) of the (POS, BLOCK_K) first-pass tile."""
    block_k = _prev_pow2(min(inner_size, _UNIFORM_STATS_BLOCK_K))
    pos = max(1, _UNIFORM_STATS_TILE // block_k)
    if segment_length < pos:
        pos = _prev_pow2(segment_length)
    return block_k, pos, triton.cdiv(inner_size, block_k)


@triton.jit
def _mul_combine(a, b):
    return a * b


def _all_lengths_equal(lengths, value):
    cache_key = (
        lengths.device.type,
        lengths.data_ptr(),
        tuple(lengths.shape),
        getattr(lengths, "_version", None),
        value,
    )
    is_equal = _UNIFORM_LENGTHS_CACHE.get(cache_key)
    if is_equal is None:
        is_equal = torch.all(lengths.detach().cpu() == value).item()
        if len(_UNIFORM_LENGTHS_CACHE) > 128:
            _UNIFORM_LENGTHS_CACHE.clear()
        _UNIFORM_LENGTHS_CACHE[cache_key] = is_equal
    return is_equal


def _wrap_axis(axis, ndim):
    if ndim == 0:
        raise IndexError(
            "segment_reduce(): input tensor must have at least one dimension."
        )
    if axis < -ndim or axis >= ndim:
        raise IndexError(
            f"segment_reduce(): axis {axis} is out of bounds for tensor of dimension {ndim}."
        )
    return axis % ndim


def _check_reduce_and_dtype(data, reduce):
    if reduce not in _SUPPORTED_REDUCES:
        raise RuntimeError(
            "segment_reduce(): reduce must be one of 'sum', 'mean', 'max', 'min', or 'prod'."
        )
    if data.dtype not in _SUPPORTED_DATA_DTYPES:
        raise NotImplementedError(f'"segment_reduce" not implemented for {data.dtype}.')


def _check_index_tensor(data, index_tensor, name, axis):
    if index_tensor.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise NotImplementedError(f"segment_reduce(): {name} must be int32 or int64.")
    if index_tensor.device != data.device:
        raise RuntimeError(
            f"segment_reduce(): Expected data and {name} on the same device."
        )
    if data.dim() < index_tensor.dim():
        raise RuntimeError(
            f"segment_reduce(): Expected data.dim() >= {name}.dim(), got "
            f"{data.dim()} and {index_tensor.dim()}."
        )
    if axis != index_tensor.dim() - 1:
        raise RuntimeError(
            f"segment_reduce(): Expected axis to be the last dimension of {name} "
            f"but got {axis}."
        )


def _validate_lengths(data, lengths, axis, unsafe):
    _check_index_tensor(data, lengths, "lengths", axis)
    if unsafe:
        return
    lengths_cpu = lengths.detach().cpu()
    if torch.any(lengths_cpu < 0).item():
        raise RuntimeError("lengths contains negative value!")
    valid_lengths = torch.all(lengths_cpu.sum(dim=-1) == data.size(axis)).item()
    if not valid_lengths:
        raise RuntimeError(
            "segment_reduce(): Expected all rows of lengths along axis to sum to "
            "data.size(lengths.dim()-1) when !unsafe."
        )


def _make_initial(reduce, initial):
    if initial is not None:
        return True, initial
    if reduce == "max":
        return False, float("-inf")
    if reduce == "min":
        return False, float("inf")
    if reduce == "prod":
        return False, 1.0
    return False, 0.0


def _get_uniform_segment_length(data, lengths, axis):
    if data.numel() < _UNIFORM_FAST_PATH_MIN_NUMEL:
        return None
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return None
    segment_count = lengths.shape[-1]
    if segment_count <= 0:
        return None
    data_size_axis = data.shape[axis]
    if data_size_axis % segment_count != 0:
        return None
    segment_length = data_size_axis // segment_count
    if segment_length <= 0:
        return None

    if _all_lengths_equal(lengths, segment_length):
        return segment_length
    return None


def _is_unit_lengths(data, lengths, axis):
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return False
    if lengths.shape[-1] != data.shape[axis]:
        return False
    return _all_lengths_equal(lengths, 1)


@libentry()
@triton.jit
def _segment_reduce_uniform_other_backward_kernel(
    grad,
    output,
    data,
    grad_input,
    total_rows,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    num_row_tiles,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    INITIAL_PROD_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GRID_M: tl.constexpr,
):
    pid_k = tle.program_id(1)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    # Fixed grid.x = GRID_M (no more than GCU400's 48 CTAs); each program
    # strides over its own row tiles, avoiding grid.x exceeding the 65535 hardware limit for large tensors.
    for pid_m in tl.range(tle.program_id(0), num_row_tiles, GRID_M):
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None]
        seg_offsets = tl.arange(0, BLOCK_N)[None, :, None]
        k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, None, :]
        output_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        output_k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        row_mask = rows < total_rows
        seg_mask = seg_offsets < segment_length
        k_mask = k_offsets < inner_size
        mask = row_mask & seg_mask & k_mask

        outer_idx = rows // segment_count
        dim_idx = rows - outer_idx * segment_count
        data_offsets = (
            outer_idx * data_size_axis * inner_size
            + (dim_idx * segment_length + seg_offsets) * inner_size
            + k_offsets
        )
        output_offsets = output_rows * inner_size + output_k_offsets
        output_mask = (output_rows < total_rows) & (output_k_offsets < inner_size)

        values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
        grad_value = tl.load(grad + output_offsets, mask=output_mask, other=0.0).to(
            compute_dtype
        )
        output_value = tl.load(output + output_offsets, mask=output_mask, other=0.0).to(
            compute_dtype
        )

        if IS_MAX or IS_MIN:
            match = ((values != values) | (values == output_value[:, None, :])) & mask
            counter = tl.sum(match.to(tl.int64), axis=1)
            store_value = tl.where(
                (counter >= 2) & (grad_value > 0),
                grad_value / counter,
                grad_value,
            )
            tl.store(grad_input + data_offsets, store_value[:, None, :], mask=match)
        elif IS_PROD:
            nan_mask = (values != values) & mask
            zero_mask = (values == 0) & mask & ~nan_mask
            zero_count = tl.sum(zero_mask.to(tl.int64), axis=1)
            nan_count = tl.sum(nan_mask.to(tl.int64), axis=1)
            product_values = tl.where(nan_mask | zero_mask | ~mask, 1.0, values)
            product = tl.reduce(product_values, axis=1, combine_fn=_mul_combine)
            product *= INITIAL_PROD_VALUE

            zero_scalar = tl.full((BLOCK_M, BLOCK_K), 0.0, dtype=compute_dtype)
            nan_scalar = zero_scalar / zero_scalar
            normal_prefix = grad_value * output_value
            normal_grad = normal_prefix[:, None, :] / values
            zero_exclusive = tl.where(
                nan_count > 0,
                nan_scalar,
                tl.where(zero_count > 1, zero_scalar, product),
            )
            nan_exclusive = tl.where(
                nan_count > 1,
                nan_scalar,
                tl.where(zero_count > 0, zero_scalar, product),
            )
            exclusive = tl.where(
                nan_mask, nan_exclusive[:, None, :], zero_exclusive[:, None, :]
            )
            grad_result = tl.where(
                nan_mask | zero_mask,
                grad_value[:, None, :] * exclusive,
                normal_grad,
            )
            tl.store(grad_input + data_offsets, grad_result, mask=mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_sum_mean_backward_kernel(
    grad,
    grad_input,
    total_numel,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_MEAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (total_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    for pid in tl.range(tle.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_numel

        inner_idx = offsets % inner_size
        axis_idx = (offsets // inner_size) % data_size_axis
        outer_idx = offsets // (data_size_axis * inner_size)
        segment_idx = axis_idx // segment_length
        grad_offsets = (
            outer_idx * segment_count + segment_idx
        ) * inner_size + inner_idx

        grad_value = tl.load(grad + grad_offsets, mask=mask, other=0.0)
        if IS_MEAN:
            grad_value = grad_value / segment_length
        tl.store(grad_input + offsets, grad_value, mask=mask)


# ---------------------------------------------------------------------------
# Region-aligned broadcast backward kernels (sum/mean, max/min, prod).
#
# Under uniform lengths the flat layout of data/grad_input is one "region" of
# L*K contiguous elements per (outer, segment) pair, and a region maps to
# exactly one grad/output/count row (row = outer*segment_count + segment). So
# every element of a region shares one small value triple and only its channel
# index (lane % K) selects inside it. When L and K are powers of two a tile of
# TILE consecutive elements therefore never needs a per-element index division:
# the channel pattern is loop invariant and the region index is a scalar derived
# from the program id. The previous kernels recomputed
# offset % K, (offset // K) % axis, offset // (axis*K) and offset // L for every
# element and measured 8-20 GB/s on the same shapes; this form is bound by the
# store stream only.
# ---------------------------------------------------------------------------
_UNIFORM_BCAST_TILE = 16384
# Element budget of the (POS, BLOCK_K) statistics tile used by the first pass of
# the max/min (match count) and prod (product/zero/nan) backward paths.
_UNIFORM_STATS_TILE = 8192
_UNIFORM_STATS_BLOCK_K = 2048


@libentry()
@triton.jit
def _segment_reduce_uniform_bcast_backward_kernel(
    grad,
    grad_input,
    total_numel,
    IS_MEAN: tl.constexpr,
    SEGMENT: tl.constexpr,
    INNER: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_REGION: tl.constexpr,
    SCALAR_SMALL: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # grad_input[r*L*K + p*K + k] = grad[r*K + k] (mean: divided by L), i.e. a
    # pure broadcast of the small grad row over the L positions of the region.
    # Tiles are flat 1-D so the store is one contiguous run; for K == 1 the whole
    # source row is a single scalar which is loaded once and broadcast by the
    # store itself. Loading it per output lane (offset = lane % K) degenerates
    # into an all-duplicate gather: measured 182ms vs 3ms on the 2 GiB K == 1
    # shape. A (REP, WIDTH) 2-D tile with `value + zeros((REP, 1))` was also
    # measured and is slower (9.5ms vs 1.0ms on the fp16 (1024,1024,1024) case).
    lane = tl.arange(0, TILE)
    channel = lane % INNER
    data_dtype = grad.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32
    region_span = SEGMENT * INNER
    num_tile = tl.cdiv(total_numel, TILE)
    for pid in tl.range(tle.program_id(0), num_tile, GRID, num_stages=NUM_STAGES):
        region = pid // TILES_PER_REGION
        sub = (pid % TILES_PER_REGION) * TILE
        mask = sub + lane < region_span
        flat_offset = region.to(tl.int64) * region_span + sub + lane
        if SCALAR_SMALL:
            value = tl.load(grad + region)
            if IS_MEAN:
                value = (value.to(compute_dtype) / SEGMENT).to(data_dtype)
            tl.store(grad_input + flat_offset, value, mask=mask)
        else:
            grad_offset = region.to(tl.int64) * INNER + (sub % INNER) + channel
            value = tl.load(grad + grad_offset, mask=mask, other=0.0)
            if IS_MEAN:
                value = (value.to(compute_dtype) / SEGMENT).to(data_dtype)
            tl.store(grad_input + flat_offset, value, mask=mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_count_backward_kernel(
    data,
    output,
    counts,
    total_rows,
    segment_length,
    INNER: tl.constexpr,
    BLOCK_K: tl.constexpr,
    POS: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # First pass of max/min backward: for every (row, channel) count how many
    # positions of the segment match output (NaN counts as a match). The count is
    # kept as a float: it is exact up to 2^24 and avoids the int64 reductions that
    # dominated the old single-pass kernel.
    cols = tl.arange(0, BLOCK_K)
    pos = tl.arange(0, POS)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32
    num_task = total_rows * NUM_K_TILES
    for task in tl.range(tle.program_id(0), num_task, GRID, num_stages=NUM_STAGES):
        row = task // NUM_K_TILES
        k_tile = task % NUM_K_TILES
        k_mask = k_tile * BLOCK_K + cols < INNER
        small_offset = row.to(tl.int64) * INNER + k_tile * BLOCK_K + cols
        output_value = tl.load(output + small_offset, mask=k_mask, other=0.0).to(
            compute_dtype
        )
        row_base = row.to(tl.int64) * segment_length * INNER + k_tile * BLOCK_K
        counter = tl.zeros((BLOCK_K,), dtype=compute_dtype)
        for p0 in range(0, segment_length, POS):
            segment_offsets = p0 + pos
            mask = (segment_offsets[:, None] < segment_length) & k_mask[None, :]
            offset = (
                row_base + segment_offsets.to(tl.int64)[:, None] * INNER + cols[None, :]
            )
            value = tl.load(data + offset, mask=mask, other=0.0).to(compute_dtype)
            hit = ((value != value) | (value == output_value[None, :])) & mask
            counter += tl.sum(hit.to(compute_dtype), axis=0)
        tl.store(counts + small_offset, counter, mask=k_mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_match_backward_kernel(
    data,
    output,
    grad,
    counts,
    grad_input,
    total_numel,
    SEGMENT: tl.constexpr,
    INNER: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_REGION: tl.constexpr,
    SCALAR_SMALL: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Second pass of max/min backward: every element is written (non-matching
    # positions get 0), so the destination no longer needs pre-zeroing and the
    # store is a full contiguous stream instead of the old predicated store.
    # SCALAR_SMALL handles K == 1, where the small rows are single scalars.
    lane = tl.arange(0, TILE)
    channel = lane % INNER
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32
    region_span = SEGMENT * INNER
    num_tile = tl.cdiv(total_numel, TILE)
    for pid in tl.range(tle.program_id(0), num_tile, GRID, num_stages=NUM_STAGES):
        region = pid // TILES_PER_REGION
        sub = (pid % TILES_PER_REGION) * TILE
        mask = sub + lane < region_span
        flat_offset = region.to(tl.int64) * region_span + sub + lane
        value = tl.load(data + flat_offset, mask=mask, other=0.0).to(compute_dtype)
        if SCALAR_SMALL:
            output_value = tl.load(output + region).to(compute_dtype)
            grad_value = tl.load(grad + region).to(compute_dtype)
            counter = tl.load(counts + region).to(compute_dtype)
        else:
            small_offset = region.to(tl.int64) * INNER + (sub % INNER) + channel
            output_value = tl.load(output + small_offset, mask=mask, other=0.0).to(
                compute_dtype
            )
            grad_value = tl.load(grad + small_offset, mask=mask, other=0.0).to(
                compute_dtype
            )
            counter = tl.load(counts + small_offset, mask=mask, other=0.0).to(
                compute_dtype
            )
        hit = (value != value) | (value == output_value)
        shared = tl.where(
            (counter >= 2) & (grad_value > 0), grad_value / counter, grad_value
        )
        tl.store(
            grad_input + flat_offset,
            tl.where(hit, shared, 0.0).to(data_dtype),
            mask=mask,
        )


@libentry()
@triton.jit
def _segment_reduce_uniform_prod_stats_backward_kernel(
    data,
    products,
    flags,
    total_rows,
    segment_length,
    INITIAL_PROD: tl.constexpr,
    INNER: tl.constexpr,
    BLOCK_K: tl.constexpr,
    POS: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # First pass of prod backward: product of the non-zero non-NaN elements plus
    # the zero/NaN multiplicity flags (bit0 has_nan, bit1 multi_nan, bit2
    # has_zero, bit3 multi_zero) per (row, channel).
    cols = tl.arange(0, BLOCK_K)
    pos = tl.arange(0, POS)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32
    num_task = total_rows * NUM_K_TILES
    for task in tl.range(tle.program_id(0), num_task, GRID, num_stages=NUM_STAGES):
        row = task // NUM_K_TILES
        k_tile = task % NUM_K_TILES
        k_mask = k_tile * BLOCK_K + cols < INNER
        small_offset = row.to(tl.int64) * INNER + k_tile * BLOCK_K + cols
        row_base = row.to(tl.int64) * segment_length * INNER + k_tile * BLOCK_K
        product = tl.full((BLOCK_K,), INITIAL_PROD, dtype=compute_dtype)
        zero_count = tl.zeros((BLOCK_K,), dtype=compute_dtype)
        nan_count = tl.zeros((BLOCK_K,), dtype=compute_dtype)
        for p0 in range(0, segment_length, POS):
            segment_offsets = p0 + pos
            mask = (segment_offsets[:, None] < segment_length) & k_mask[None, :]
            offset = (
                row_base + segment_offsets.to(tl.int64)[:, None] * INNER + cols[None, :]
            )
            value = tl.load(data + offset, mask=mask, other=1.0).to(compute_dtype)
            nan_mask = (value != value) & mask
            zero_mask = (value == 0) & mask & ~nan_mask
            product *= tl.reduce(
                tl.where(nan_mask | zero_mask | ~mask, 1.0, value),
                axis=0,
                combine_fn=_mul_combine,
            )
            zero_count += tl.sum(zero_mask.to(compute_dtype), axis=0)
            nan_count += tl.sum(nan_mask.to(compute_dtype), axis=0)
        flag = (
            (nan_count > 0).to(tl.int32)
            | ((nan_count > 1).to(tl.int32) << 1)
            | ((zero_count > 0).to(tl.int32) << 2)
            | ((zero_count > 1).to(tl.int32) << 3)
        )
        tl.store(products + small_offset, product, mask=k_mask)
        tl.store(flags + small_offset, flag.to(tl.int8), mask=k_mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_prod_backward_kernel(
    data,
    output,
    grad,
    products,
    flags,
    grad_input,
    total_numel,
    SEGMENT: tl.constexpr,
    INNER: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_REGION: tl.constexpr,
    SCALAR_SMALL: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Second pass of prod backward: one full contiguous store per tile, with the
    # exclusive product obtained as grad*output/value for normal elements and
    # from the pass-one product/flags for the zero and NaN elements.
    lane = tl.arange(0, TILE)
    channel = lane % INNER
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32
    region_span = SEGMENT * INNER
    num_tile = tl.cdiv(total_numel, TILE)
    for pid in tl.range(tle.program_id(0), num_tile, GRID, num_stages=NUM_STAGES):
        region = pid // TILES_PER_REGION
        sub = (pid % TILES_PER_REGION) * TILE
        mask = sub + lane < region_span
        flat_offset = region.to(tl.int64) * region_span + sub + lane
        value = tl.load(data + flat_offset, mask=mask, other=1.0).to(compute_dtype)
        if SCALAR_SMALL:
            output_value = tl.load(output + region).to(compute_dtype)
            grad_value = tl.load(grad + region).to(compute_dtype)
            product = tl.load(products + region).to(compute_dtype)
            flag = tl.load(flags + region).to(tl.int32)
        else:
            small_offset = region.to(tl.int64) * INNER + (sub % INNER) + channel
            output_value = tl.load(output + small_offset, mask=mask, other=0.0).to(
                compute_dtype
            )
            grad_value = tl.load(grad + small_offset, mask=mask, other=0.0).to(
                compute_dtype
            )
            product = tl.load(products + small_offset, mask=mask, other=1.0).to(
                compute_dtype
            )
            flag = tl.load(flags + small_offset, mask=mask, other=0).to(tl.int32)
        has_nan = (flag & 1) != 0
        multi_nan = (flag & 2) != 0
        has_zero = (flag & 4) != 0
        multi_zero = (flag & 8) != 0

        nan_mask = value != value
        zero_mask = (value == 0) & ~nan_mask
        zero_scalar = tl.zeros((TILE,), dtype=compute_dtype)
        nan_scalar = zero_scalar / zero_scalar
        normal_grad = (grad_value * output_value) / value
        zero_exclusive = tl.where(
            has_nan, nan_scalar, tl.where(multi_zero, zero_scalar, product)
        )
        nan_exclusive = tl.where(
            multi_nan, nan_scalar, tl.where(has_zero, zero_scalar, product)
        )
        exclusive = tl.where(nan_mask, nan_exclusive, zero_exclusive)
        result = tl.where(nan_mask | zero_mask, grad_value * exclusive, normal_grad)
        tl.store(grad_input + flat_offset, result.to(data_dtype), mask=mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_rows_kernel(
    data,
    output,
    total_numel,
    total_rows,
    segment_length,
    rows_per_task,
    num_tasks,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    ROWS: tl.constexpr,
    SEGMENT: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Whole-segment fast path for K == 1. Under uniform lengths the data of
    # the r-th segment sits exactly at flat offset r * segment_length and
    # adjacent segments are contiguous, so one flat 1D load per task covers
    # rows_per_task complete segments. Everything is a 1D contiguous load
    # with an i64 scalar base: the P17 family proved that 2D row-major
    # tiles (X + m*N + n shapes) wrap around 32-bit addressing once the
    # covered byte range reaches 2 GiB and abort the device outright, while
    # this flat-1D form stays safe even at 4 GiB. The tail-bound load mask
    # (base + flat < total_numel) is strictly equivalent to "row not out of
    # range", and the masked load's `other` is exactly the reduction
    # neutral element, so no extra where is needed. GCU's tl.max/tl.min
    # ignore NaN (measured: max([1, nan, -3]) == 1), so NaN semantics are
    # implemented via has_nan detection plus a NaN constant written back.
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = tl.arange(0, ROWS)
    flat = tl.arange(0, ROWS * SEGMENT)

    if IS_MAX:
        neutral = float("-inf")
    elif IS_MIN:
        neutral = float("inf")
    elif IS_PROD:
        neutral = 1.0
    else:
        neutral = 0.0

    for task in tl.range(tle.program_id(0), num_tasks, GRID, num_stages=NUM_STAGES):
        row0 = task.to(tl.int64) * rows_per_task
        rows_this = tl.minimum(total_rows - row0, rows_per_task)
        base = row0 * segment_length
        values = tl.load(
            data + base + flat, mask=base + flat < total_numel, other=neutral
        ).to(compute_dtype)
        values = tl.reshape(values, (ROWS, SEGMENT))

        if IS_SUM or IS_MEAN:
            result = tl.sum(values, axis=1)
            if IS_MEAN:
                result = result / segment_length
        elif IS_PROD:
            result = tl.reduce(values, axis=1, combine_fn=_mul_combine)
        else:
            if IS_MAX:
                chunk = tl.max(values, axis=1)
            else:
                chunk = tl.min(values, axis=1)
            has_nan = tl.max((values != values).to(tl.int8), axis=1) > 0
            zero = tl.zeros((ROWS,), dtype=compute_dtype)
            nan_value = zero / zero
            result = tl.where(has_nan, nan_value, chunk)

        tl.store(
            output + row0 + rows,
            result.to(output.dtype.element_ty),
            mask=rows < rows_this,
        )


@libentry()
@triton.jit
def _segment_reduce_uniform_inner_kernel(
    data,
    output,
    total_rows,
    segment_length,
    inner_size,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    GROUP: tl.constexpr,
    POS: tl.constexpr,
    INNER: tl.constexpr,
    EVEN: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Whole-row fast path for K > 1: the data of row r is the 1D contiguous
    # block [r*L*K, (r+1)*L*K). Each task covers GROUP whole rows with a
    # (GROUP, POS, INNER) 3D tile (the row-dimension base advances as an
    # i64 scalar and in-tile offsets span <= 512 KiB, far away from P17's
    # 2^31 32-bit address wrap), reducing along the position axis into
    # GROUP*INNER values. A/B measurements: choosing GROUP from the dtype
    # budget (2B -> 65536 elements / 4B -> 131072 elements) is 1.9-2.6x
    # faster than one task per row. The tail mask's `other` is the
    # reduction neutral element; GCU tl.max/min ignore NaN, so NaN
    # semantics use has_nan plus a NaN constant.
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = tl.arange(0, GROUP)
    pos = tl.arange(0, POS)
    cols = tl.arange(0, INNER)
    row_numel = segment_length * inner_size
    num_tasks = tl.cdiv(total_rows, GROUP)

    if IS_MAX:
        neutral = float("-inf")
    elif IS_MIN:
        neutral = float("inf")
    elif IS_PROD:
        neutral = 1.0
    else:
        neutral = 0.0

    for task in tl.range(tle.program_id(0), num_tasks, GRID, num_stages=NUM_STAGES):
        row0 = task * GROUP
        rows_this = tl.minimum(total_rows - row0, GROUP)
        base = row0.to(tl.int64) * row_numel
        row_mask = rows < rows_this
        offsets = (
            rows[:, None, None] * row_numel
            + pos[None, :, None] * inner_size
            + cols[None, None, :]
        )
        if EVEN:
            load_mask = row_mask[:, None, None]
        else:
            load_mask = row_mask[:, None, None] & (pos[None, :, None] < segment_length)
        values = tl.load(data + base + offsets, mask=load_mask, other=neutral).to(
            compute_dtype
        )

        if IS_SUM or IS_MEAN:
            result = tl.sum(values, axis=1)
            if IS_MEAN:
                result = result / segment_length
        elif IS_PROD:
            result = tl.reduce(values, axis=1, combine_fn=_mul_combine)
        else:
            if IS_MAX:
                chunk = tl.max(values, axis=1)
            else:
                chunk = tl.min(values, axis=1)
            has_nan = tl.max((values != values).to(tl.int8), axis=1) > 0
            zero = tl.zeros((GROUP, INNER), dtype=compute_dtype)
            nan_value = zero / zero
            result = tl.where(has_nan, nan_value, chunk)

        out_offsets = (row0 + rows).to(tl.int64)[:, None] * inner_size + cols[None, :]
        tl.store(
            output + out_offsets,
            result.to(output.dtype.element_ty),
            mask=row_mask[:, None],
        )


@libentry()
@triton.jit
def _segment_reduce_uniform_loop_kernel(
    data,
    output,
    total_rows,
    segment_length,
    inner_size,
    num_k_tiles,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Generic path for K > 1: task = (row, inner tile), looping along the
    # position axis, each iteration a 1D contiguous load of BLOCK_K inner
    # values. Taken when the inner width is not a power of two or L*K
    # exceeds the tile budget.
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32
    num_tasks = total_rows * num_k_tiles

    for task in tl.range(tle.program_id(0), num_tasks, GRID, num_stages=NUM_STAGES):
        row = task // num_k_tiles
        k_tile = task % num_k_tiles
        cols = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = cols < inner_size
        base = row.to(tl.int64) * segment_length * inner_size

        if IS_MAX:
            neutral = float("-inf")
        elif IS_MIN:
            neutral = float("inf")
        elif IS_PROD:
            neutral = 1.0
        else:
            neutral = 0.0
        acc = tl.full((BLOCK_K,), neutral, dtype=compute_dtype)
        has_nan = tl.zeros((BLOCK_K,), dtype=tl.int32)
        nan_value = tl.zeros((BLOCK_K,), dtype=compute_dtype)

        for seg_pos in tl.range(0, segment_length):
            offsets = base + seg_pos.to(tl.int64) * inner_size + cols
            values = tl.load(data + offsets, mask=k_mask, other=neutral).to(
                compute_dtype
            )
            if IS_SUM or IS_MEAN:
                acc += values
            elif IS_PROD:
                acc *= values
            else:
                nan_mask = (values != values) & k_mask
                has_nan = has_nan | nan_mask.to(tl.int32)
                nan_value = tl.where(nan_mask, values, nan_value)
                if IS_MAX:
                    acc = tl.maximum(acc, tl.where(k_mask & ~nan_mask, values, neutral))
                else:
                    acc = tl.minimum(acc, tl.where(k_mask & ~nan_mask, values, neutral))

        if IS_MEAN:
            acc = acc / segment_length
        if IS_MAX or IS_MIN:
            acc = tl.where(has_nan > 0, nan_value, acc)
        tl.store(
            output + row.to(tl.int64) * inner_size + cols,
            acc.to(output.dtype.element_ty),
            mask=k_mask,
        )


@libentry()
@triton.jit
def _segment_reduce_uniform_split_partial_kernel(
    data,
    partial,
    total_rows,
    segment_length,
    num_chunks,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    CHUNK: tl.constexpr,
    GRID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # K == 1 with a segment exceeding the single-tile budget (or a
    # non-power-of-two length): task = (row, chunk), each chunk is a 1D
    # contiguous load reduced into an fp32 partial, then the final kernel
    # merges the partials a second time. The masked load's `other` is the
    # reduction neutral element; GCU tl.max/min ignore NaN, so NaN
    # semantics use has_nan plus a NaN constant.
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32
    num_tasks = total_rows * num_chunks
    cols = tl.arange(0, CHUNK)

    if IS_MAX:
        neutral = float("-inf")
    elif IS_MIN:
        neutral = float("inf")
    elif IS_PROD:
        neutral = 1.0
    else:
        neutral = 0.0

    for task in tl.range(tle.program_id(0), num_tasks, GRID, num_stages=NUM_STAGES):
        row = task // num_chunks
        chunk = task % num_chunks
        chunk_len = tl.minimum(segment_length - chunk * CHUNK, CHUNK)
        base = row.to(tl.int64) * segment_length + chunk * CHUNK

        valid = cols < chunk_len
        values = tl.load(data + base + cols, mask=valid, other=neutral).to(
            compute_dtype
        )

        if IS_SUM or IS_MEAN:
            scalar = tl.sum(values, axis=0)
        elif IS_PROD:
            scalar = tl.reduce(values, axis=0, combine_fn=_mul_combine)
        else:
            if IS_MAX:
                scalar = tl.max(values, axis=0)
            else:
                scalar = tl.min(values, axis=0)
            has_nan = tl.max((values != values).to(tl.int8), axis=0) > 0
            zero = tl.full((), 0.0, dtype=compute_dtype)
            nan_value = zero / zero
            scalar = tl.where(has_nan, nan_value, scalar)

        tl.store(partial + task, scalar.to(partial.dtype.element_ty))


@libentry()
@triton.jit
def _segment_reduce_uniform_split_final_kernel(
    partial,
    output,
    total_rows,
    num_chunks,
    segment_length,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    C_BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    partial_dtype = partial.dtype.element_ty
    cols = tl.arange(0, C_BLOCK)

    if IS_MAX:
        neutral = float("-inf")
    elif IS_MIN:
        neutral = float("inf")
    elif IS_PROD:
        neutral = 1.0
    else:
        neutral = 0.0

    for row in tl.range(tle.program_id(0), total_rows, GRID):
        base = row.to(tl.int64) * num_chunks
        acc = tl.full((), neutral, dtype=partial_dtype)
        has_nan = tl.full((), 0, dtype=tl.int32)

        for chunk_start in tl.range(0, num_chunks, C_BLOCK):
            offsets = chunk_start + cols
            mask = offsets < num_chunks
            values = tl.load(partial + base + offsets, mask=mask, other=neutral)
            if IS_SUM or IS_MEAN:
                acc += tl.sum(values, axis=0)
            elif IS_PROD:
                acc *= tl.reduce(values, axis=0, combine_fn=_mul_combine)
            else:
                if IS_MAX:
                    acc = tl.maximum(acc, tl.max(values, axis=0))
                else:
                    acc = tl.minimum(acc, tl.min(values, axis=0))
                has_nan = has_nan | (
                    tl.sum((values != values).to(tl.int32), axis=0) > 0
                )

        if IS_MEAN:
            acc = acc / segment_length
        if IS_MAX or IS_MIN:
            zero = tl.full((), 0.0, dtype=partial_dtype)
            nan_value = zero / zero
            acc = tl.where(has_nan > 0, nan_value, acc)
        tl.store(output + row, acc.to(output.dtype.element_ty))


def _segment_reduce_uniform_lengths(data, reduce, lengths, axis):
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None

    output_shape = lengths.shape + data.shape[axis + 1 :]
    inner_size = _prod(data.shape[axis + 1 :])
    total_rows = _prod(lengths.shape)
    output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
    if output.numel() == 0:
        return output

    is_sum = reduce == "sum"
    is_mean = reduce == "mean"
    is_max = reduce == "max"
    is_min = reduce == "min"
    is_prod = reduce == "prod"

    # Under uniform lengths the flat layout of data is a perfect (rows, L, K)
    # blocking: the r-th segment occupies [r*L*K, (r+1)*L*K). Every path uses
    # only our own Triton kernels doing 1D contiguous memory access -- the
    # host must not fall back to torch.sum/amax or other aten compute entry
    # points (under use_gems dispatch those route back to the corresponding
    # FlagGems ops, and on 2026-11-18 their 2D row-major tiles were proven
    # to trigger the P17-family 32-bit address wrap device abort at 2 GiB
    # coverage).
    if inner_size == 1:
        if _is_pow2(segment_length) and segment_length <= _UNIFORM_TILE_ELEMS:
            elem_bytes = 8 if data.dtype == torch.float64 else 4
            # rows_per_task also has to keep the 48 CTAs busy: with only
            # total_rows // GRID_SIZE rows per task the strided task loop fills the
            # device (fp16 (4096,4096), L=64: 512 rows/task leaves grid=8 of 48 CTAs,
            # capping the whole kernel at 1/6 of the achievable bandwidth).
            rows_per_task = max(
                1,
                min(
                    _UNIFORM_TILE_ELEMS // segment_length,
                    _uniform_rows_per_task(segment_length, elem_bytes),
                    _prev_pow2(max(1, total_rows // _GRID_SIZE)),
                ),
            )
            num_tasks = triton.cdiv(total_rows, rows_per_task)
            grid = (min(_GRID_SIZE, num_tasks),)
            with torch_device_fn.device(data.device):
                _segment_reduce_uniform_rows_kernel[grid](
                    data,
                    output,
                    data.numel(),
                    total_rows,
                    segment_length,
                    rows_per_task,
                    num_tasks,
                    is_sum,
                    is_mean,
                    is_max,
                    is_min,
                    is_prod,
                    ROWS=rows_per_task,
                    SEGMENT=segment_length,
                    GRID=min(_GRID_SIZE, num_tasks),
                    NUM_STAGES=2,
                    num_warps=4,
                )
            return output

        num_chunks = triton.cdiv(segment_length, _UNIFORM_SPLIT_CHUNK)
        partial_dtype = torch.float64 if data.dtype == torch.float64 else torch.float32
        partial = torch.empty(
            (total_rows, num_chunks), dtype=partial_dtype, device=data.device
        )
        num_tasks = total_rows * num_chunks
        grid = (min(_GRID_SIZE, num_tasks),)
        with torch_device_fn.device(data.device):
            _segment_reduce_uniform_split_partial_kernel[grid](
                data,
                partial,
                total_rows,
                segment_length,
                num_chunks,
                is_sum,
                is_mean,
                is_max,
                is_min,
                is_prod,
                CHUNK=_UNIFORM_SPLIT_CHUNK,
                GRID=min(_GRID_SIZE, num_tasks),
                NUM_STAGES=2,
                num_warps=4,
            )
            c_block = min(triton.next_power_of_2(num_chunks), _UNIFORM_TILE_ELEMS)
            grid2 = (min(_GRID_SIZE, total_rows),)
            _segment_reduce_uniform_split_final_kernel[grid2](
                partial,
                output,
                total_rows,
                num_chunks,
                segment_length,
                is_sum,
                is_mean,
                is_max,
                is_min,
                is_prod,
                C_BLOCK=c_block,
                GRID=min(_GRID_SIZE, total_rows),
                num_warps=2,
            )
        return output

    inner_budget = (
        _UNIFORM_INNER_BUDGET_4B
        if data.element_size() >= 4
        else _UNIFORM_INNER_BUDGET_2B
    )
    if reduce in ("max", "min"):
        # The max/min path carries an extra int8 NaN mask and comparison
        # intermediates, so tighten the budget one tier; otherwise the
        # 131072-element tier exceeds the 917504-byte DSM hard limit.
        inner_budget = _UNIFORM_INNER_BUDGET_2B
    pos = triton.next_power_of_2(segment_length)
    if _is_pow2(inner_size) and pos * inner_size <= inner_budget:
        # Whole-row fast path: each task covers GROUP whole rows with a 3D
        # tile. GROUP is the smaller power of two between the dtype budget
        # and the parallelism target (tasks >= 48).
        group_budget = inner_budget // (pos * inner_size)
        group_par = _prev_pow2(max(1, total_rows // _GRID_SIZE))
        group = max(1, min(group_budget, group_par))
        num_tasks = triton.cdiv(total_rows, group)
        grid = (min(_GRID_SIZE, num_tasks),)
        with torch_device_fn.device(data.device):
            _segment_reduce_uniform_inner_kernel[grid](
                data,
                output,
                total_rows,
                segment_length,
                inner_size,
                is_sum,
                is_mean,
                is_max,
                is_min,
                is_prod,
                GROUP=group,
                POS=pos,
                INNER=inner_size,
                EVEN=pos == segment_length,
                GRID=min(_GRID_SIZE, num_tasks),
                NUM_STAGES=2,
                num_warps=4,
            )
        return output

    block_k = min(triton.next_power_of_2(inner_size), _UNIFORM_LOOP_BLOCK_K)
    num_k_tiles = triton.cdiv(inner_size, block_k)
    num_tasks = total_rows * num_k_tiles
    grid = (min(_GRID_SIZE, num_tasks),)
    with torch_device_fn.device(data.device):
        _segment_reduce_uniform_loop_kernel[grid](
            data,
            output,
            total_rows,
            segment_length,
            inner_size,
            num_k_tiles,
            is_sum,
            is_mean,
            is_max,
            is_min,
            is_prod,
            BLOCK_K=block_k,
            GRID=min(_GRID_SIZE, num_tasks),
            NUM_STAGES=2,
            num_warps=4,
        )
    return output


def _segment_reduce_uniform_sum_mean_backward(
    data, grad, reduce, lengths, axis, out_buffer=None
):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None

    grad_input = (
        out_buffer
        if out_buffer is not None
        else torch.empty_like(data, dtype=grad.dtype)
    )
    if grad_input.numel() == 0:
        return grad_input

    inner_size = _prod(data.shape[axis + 1 :])
    bcast_config = _uniform_bcast_tile_config(segment_length, inner_size)
    with torch_device_fn.device(data.device):
        if bcast_config is not None:
            # Region-aligned broadcast: one grad row is smeared over the L
            # positions of its region with a single contiguous store stream.
            tile, tiles_per_region = bcast_config
            num_tile = triton.cdiv(data.numel(), tile)
            grid = (min(_GRID_SIZE, num_tile), 1, 1)
            _segment_reduce_uniform_bcast_backward_kernel[grid](
                grad,
                grad_input,
                data.numel(),
                reduce == "mean",
                SEGMENT=segment_length,
                INNER=inner_size,
                TILE=tile,
                TILES_PER_REGION=tiles_per_region,
                SCALAR_SMALL=inner_size == 1,
                GRID=min(_GRID_SIZE, num_tile),
                NUM_STAGES=_UNIFORM_BACKWARD_GATHER_STAGES,
                num_warps=4,
            )
            return grad_input
        # Fallback for non power-of-two shapes: per-element gather.
        # Only GCU uses the measured best tile; other devices keep their original block size.
        block_size = (
            _UNIFORM_BACKWARD_GATHER_BLOCK
            if data.device.type != "npu"
            else _get_block_size(data.device)
        )
        num_tile = triton.cdiv(data.numel(), block_size)
        grid = (min(_GRID_SIZE, num_tile), 1, 1)
        _segment_reduce_uniform_sum_mean_backward_kernel[grid](
            grad,
            grad_input,
            data.numel(),
            segment_count,
            segment_length,
            inner_size,
            data.shape[axis],
            reduce == "mean",
            BLOCK_SIZE=block_size,
            GRID_SIZE=min(_GRID_SIZE, num_tile),
            NUM_STAGES=_UNIFORM_BACKWARD_GATHER_STAGES,
            num_warps=4,
        )
    return grad_input


def _segment_reduce_uniform_other_backward(
    data, output, grad, reduce, lengths, axis, initial, out_buffer=None
):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None

    inner_size = _prod(data.shape[axis + 1 :])
    total_rows = _prod(lengths.shape)
    bcast_config = _uniform_bcast_tile_config(segment_length, inner_size)
    if bcast_config is not None:
        grad_input = (
            out_buffer
            if out_buffer is not None
            else torch.empty_like(data, dtype=grad.dtype)
        )
        if grad_input.numel() == 0:
            return grad_input
        tile, tiles_per_region = bcast_config
        block_k, pos, num_k_tiles = _uniform_stats_config(segment_length, inner_size)
        num_task = total_rows * num_k_tiles
        stats_grid = (min(_GRID_SIZE, num_task), 1, 1)
        out_grid = (min(_GRID_SIZE, triton.cdiv(data.numel(), tile)), 1, 1)
        _, initial_prod_value = _make_initial("prod", initial)
        with torch_device_fn.device(data.device):
            if reduce in ("max", "min"):
                counts = torch.empty(
                    total_rows * inner_size, dtype=torch.float32, device=data.device
                )
                _segment_reduce_uniform_count_backward_kernel[stats_grid](
                    data,
                    output,
                    counts,
                    total_rows,
                    segment_length,
                    INNER=inner_size,
                    BLOCK_K=block_k,
                    POS=pos,
                    NUM_K_TILES=num_k_tiles,
                    GRID=min(_GRID_SIZE, num_task),
                    NUM_STAGES=2,
                    num_warps=4,
                )
                _segment_reduce_uniform_match_backward_kernel[out_grid](
                    data,
                    output,
                    grad,
                    counts,
                    grad_input,
                    data.numel(),
                    SEGMENT=segment_length,
                    INNER=inner_size,
                    TILE=tile,
                    TILES_PER_REGION=tiles_per_region,
                    SCALAR_SMALL=inner_size == 1,
                    GRID=min(_GRID_SIZE, triton.cdiv(data.numel(), tile)),
                    NUM_STAGES=_UNIFORM_BACKWARD_GATHER_STAGES,
                    num_warps=4,
                )
            else:
                products = torch.empty(
                    total_rows * inner_size, dtype=torch.float32, device=data.device
                )
                flags = torch.empty(
                    total_rows * inner_size, dtype=torch.int8, device=data.device
                )
                _segment_reduce_uniform_prod_stats_backward_kernel[stats_grid](
                    data,
                    products,
                    flags,
                    total_rows,
                    segment_length,
                    INITIAL_PROD=initial_prod_value,
                    INNER=inner_size,
                    BLOCK_K=block_k,
                    POS=pos,
                    NUM_K_TILES=num_k_tiles,
                    GRID=min(_GRID_SIZE, num_task),
                    NUM_STAGES=2,
                    num_warps=4,
                )
                _segment_reduce_uniform_prod_backward_kernel[out_grid](
                    data,
                    output,
                    grad,
                    products,
                    flags,
                    grad_input,
                    data.numel(),
                    SEGMENT=segment_length,
                    INNER=inner_size,
                    TILE=tile,
                    TILES_PER_REGION=tiles_per_region,
                    SCALAR_SMALL=inner_size == 1,
                    GRID=min(_GRID_SIZE, triton.cdiv(data.numel(), tile)),
                    NUM_STAGES=_UNIFORM_BACKWARD_GATHER_STAGES,
                    num_warps=4,
                )
        return grad_input

    if segment_length > _UNIFORM_KERNEL_MAX_SEGMENT_LENGTH:
        return None

    if reduce in ("max", "min"):
        grad_input = torch.zeros_like(data, dtype=grad.dtype)
    else:
        grad_input = torch.empty_like(data, dtype=grad.dtype)
    if grad_input.numel() == 0:
        return grad_input

    block_m, block_k = _get_uniform_backward_tile_config(
        data.device, inner_size, reduce, data.dtype
    )
    row_tiles = triton.cdiv(total_rows, block_m)
    grid_m = min(_GRID_SIZE, row_tiles)
    block_n = min(_get_block_size(data.device), triton.next_power_of_2(segment_length))
    # GCU400's DSM limit constrains 3D tile element count; 4096 elements measured safe,
    # 16384 elements report out of resource: dsm memory. BLOCK_N comes from segment_length,
    # so BLOCK_M is tightened dynamically by the actual BLOCK_N.
    while block_m > 1 and block_m * block_n * block_k > 4096:
        block_m //= 2
    row_tiles = triton.cdiv(total_rows, block_m)
    grid_m = min(_GRID_SIZE, row_tiles)
    _, initial_prod_value = _make_initial("prod", initial)
    grid = (grid_m, triton.cdiv(inner_size, block_k))
    with torch_device_fn.device(data.device):
        _segment_reduce_uniform_other_backward_kernel[grid](
            grad,
            output,
            data,
            grad_input,
            total_rows,
            segment_count,
            segment_length,
            inner_size,
            data.shape[axis],
            row_tiles,
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            initial_prod_value,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GRID_M=grid_m,
        )
    return grad_input


@libentry()
@triton.jit
def _lengths_to_offsets_kernel(
    lengths,
    offsets,
    outer_count,
    segment_count,
):
    pid = tle.program_id(0)
    acc = tl.full((), 0, dtype=tl.int64)
    base_lengths = pid * segment_count
    base_offsets = pid * (segment_count + 1)
    tl.store(offsets + base_offsets, acc)

    idx = 0
    while idx < segment_count:
        length = tl.load(lengths + base_lengths + idx)
        acc += length
        tl.store(offsets + base_offsets + idx + 1, acc)
        idx += 1


@libentry()
@triton.jit
def _segment_reduce_forward_kernel(
    data,
    offsets,
    output,
    num_outputs,
    segment_count,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    INITIAL_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    for pid in tl.range(
        tle.program_id(0), num_outputs, GRID_SIZE, num_stages=NUM_STAGES
    ):
        data_dtype = data.dtype.element_ty
        compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

        inner_idx = pid % inner_size
        row_idx = pid // inner_size
        dim_idx = row_idx % segment_count
        outer_idx = row_idx // segment_count

        offsets_base = outer_idx * (segment_count + 1) + dim_idx
        segment_start = tl.load(offsets + offsets_base)
        segment_end = tl.load(offsets + offsets_base + 1)
        segment_length = segment_end - segment_start

        acc = tl.full((), INITIAL_VALUE, dtype=compute_dtype)
        if IS_PROD:
            pos = segment_start
            while pos < segment_end:
                data_offset = (
                    outer_idx * data_size_axis * inner_size
                    + pos * inner_size
                    + inner_idx
                )
                value = tl.load(data + data_offset).to(compute_dtype)
                acc *= value
                pos += 1
        else:
            pos = segment_start
            while pos < segment_end:
                segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                mask = segment_offsets < segment_end
                data_offsets = (
                    outer_idx * data_size_axis * inner_size
                    + segment_offsets * inner_size
                    + inner_idx
                )

                if IS_SUM or IS_MEAN:
                    values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                        compute_dtype
                    )
                    acc += tl.sum(tl.where(mask, values, 0.0), axis=0)
                elif IS_MAX:
                    values = tl.load(
                        data + data_offsets, mask=mask, other=float("-inf")
                    ).to(compute_dtype)
                    nan_mask = (values != values) & mask
                    has_nan = tl.sum(nan_mask.to(tl.int32), axis=0) > 0
                    nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=0)
                    chunk = tl.max(
                        tl.where(mask & ~nan_mask, values, float("-inf")), axis=0
                    )
                    chunk = tl.where(has_nan, nan_value, chunk)
                    acc = tl.where(has_nan, chunk, tl.maximum(acc, chunk))
                elif IS_MIN:
                    values = tl.load(
                        data + data_offsets, mask=mask, other=float("inf")
                    ).to(compute_dtype)
                    nan_mask = (values != values) & mask
                    has_nan = tl.sum(nan_mask.to(tl.int32), axis=0) > 0
                    nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=0)
                    chunk = tl.min(
                        tl.where(mask & ~nan_mask, values, float("inf")), axis=0
                    )
                    chunk = tl.where(has_nan, nan_value, chunk)
                    acc = tl.where(has_nan, chunk, tl.minimum(acc, chunk))
                pos += BLOCK_SIZE

        if IS_MEAN:
            acc_is_nan = acc != acc
            nan_value = acc / acc
            if not HAS_INITIAL:
                acc = tl.where(segment_length == 0, nan_value, acc)
            acc = tl.where(
                (segment_length > 0) & ~acc_is_nan, acc / segment_length, acc
            )

        tl.store(output + pid, acc)


@libentry()
@triton.jit
def _segment_reduce_backward_kernel(
    grad,
    output,
    data,
    offsets,
    grad_input,
    num_outputs,
    segment_count,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    INITIAL_PROD_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    for pid in tl.range(
        tle.program_id(0), num_outputs, GRID_SIZE, num_stages=NUM_STAGES
    ):
        data_dtype = data.dtype.element_ty
        compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

        inner_idx = pid % inner_size
        row_idx = pid // inner_size
        dim_idx = row_idx % segment_count
        outer_idx = row_idx // segment_count

        offsets_base = outer_idx * (segment_count + 1) + dim_idx
        segment_start = tl.load(offsets + offsets_base)
        segment_end = tl.load(offsets + offsets_base + 1)
        segment_length = segment_end - segment_start

        if segment_length > 0:
            grad_value = tl.load(grad + pid).to(compute_dtype)
            output_value = tl.load(output + pid).to(compute_dtype)

            if IS_SUM or IS_MEAN:
                if IS_MEAN:
                    grad_value = grad_value / segment_length
                pos = segment_start
                while pos < segment_end:
                    segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                    mask = segment_offsets < segment_end
                    data_offsets = (
                        outer_idx * data_size_axis * inner_size
                        + segment_offsets * inner_size
                        + inner_idx
                    )
                    tl.store(grad_input + data_offsets, grad_value, mask=mask)
                    pos += BLOCK_SIZE
            elif IS_MAX or IS_MIN:
                counter = tl.full((), 0, dtype=tl.int64)
                pos = segment_start
                while pos < segment_end:
                    segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                    mask = segment_offsets < segment_end
                    data_offsets = (
                        outer_idx * data_size_axis * inner_size
                        + segment_offsets * inner_size
                        + inner_idx
                    )
                    values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                        compute_dtype
                    )
                    match = ((values != values) | (values == output_value)) & mask
                    counter += tl.sum(match.to(tl.int64), axis=0)
                    pos += BLOCK_SIZE

                store_value = tl.where(
                    (counter >= 2) & (grad_value > 0),
                    grad_value / counter,
                    grad_value,
                )
                pos = segment_start
                while pos < segment_end:
                    segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                    mask = segment_offsets < segment_end
                    data_offsets = (
                        outer_idx * data_size_axis * inner_size
                        + segment_offsets * inner_size
                        + inner_idx
                    )
                    values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                        compute_dtype
                    )
                    match = ((values != values) | (values == output_value)) & mask
                    tl.store(grad_input + data_offsets, store_value, mask=match)
                    pos += BLOCK_SIZE
            elif IS_PROD:
                zero_count = tl.full((), 0, dtype=tl.int64)
                nan_count = tl.full((), 0, dtype=tl.int64)
                product = tl.full((), INITIAL_PROD_VALUE, dtype=compute_dtype)
                pos = segment_start
                while pos < segment_end:
                    data_offset = (
                        outer_idx * data_size_axis * inner_size
                        + pos * inner_size
                        + inner_idx
                    )
                    value = tl.load(data + data_offset).to(compute_dtype)
                    if value != value:
                        nan_count += 1
                    elif value == 0:
                        zero_count += 1
                    else:
                        product *= value
                    pos += 1

                zero_scalar = tl.full((), 0.0, dtype=compute_dtype)
                nan_scalar = zero_scalar / zero_scalar
                normal_prefix = grad_value * output_value
                pos = segment_start
                while pos < segment_end:
                    segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                    mask = segment_offsets < segment_end
                    data_offsets = (
                        outer_idx * data_size_axis * inner_size
                        + segment_offsets * inner_size
                        + inner_idx
                    )
                    values = tl.load(data + data_offsets, mask=mask, other=1.0).to(
                        compute_dtype
                    )
                    nan_mask = (values != values) & mask
                    zero_mask = (values == 0) & mask & ~nan_mask
                    normal_grad = normal_prefix / values
                    zero_exclusive = tl.where(
                        nan_count > 0,
                        nan_scalar,
                        tl.where(zero_count > 1, zero_scalar, product),
                    )
                    nan_exclusive = tl.where(
                        nan_count > 1,
                        nan_scalar,
                        tl.where(zero_count > 0, zero_scalar, product),
                    )
                    exclusive = tl.where(nan_mask, nan_exclusive, zero_exclusive)
                    grad_result = tl.where(
                        nan_mask | zero_mask,
                        grad_value * exclusive,
                        normal_grad,
                    )
                    tl.store(grad_input + data_offsets, grad_result, mask=mask)
                    pos += BLOCK_SIZE


def _lengths_to_offsets(lengths):
    segment_count = lengths.shape[-1]
    offsets_shape = lengths.shape[:-1] + (segment_count + 1,)
    offsets = torch.empty(offsets_shape, dtype=lengths.dtype, device=lengths.device)
    outer_count = _prod(lengths.shape[:-1])
    if offsets.numel() > 0:
        with torch_device_fn.device(lengths.device):
            _lengths_to_offsets_kernel[(outer_count,)](
                lengths,
                offsets,
                outer_count,
                segment_count,
            )
    return offsets


def _prepare_common(data, reduce, lengths, offsets, indices, axis, unsafe):
    _check_reduce_and_dtype(data, reduce)
    axis = _wrap_axis(axis, data.dim())
    if indices is not None:
        raise RuntimeError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    if offsets is not None:
        _check_index_tensor(data, offsets, "offsets", axis)
        offsets_contig = offsets.contiguous()
        segment_count = offsets_contig.shape[-1] - 1
        output_shape = (
            offsets_contig.shape[:-1] + (segment_count,) + data.shape[axis + 1 :]
        )
        return axis, offsets_contig, output_shape, True

    if lengths is None:
        raise RuntimeError(
            "segment_reduce(): Either lengths or offsets must be defined."
        )

    _validate_lengths(data, lengths, axis, unsafe)
    lengths_contig = lengths.contiguous()
    offsets_contig = _lengths_to_offsets(lengths_contig)
    output_shape = lengths_contig.shape + data.shape[axis + 1 :]
    return axis, offsets_contig, output_shape, False


def segment_reduce(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
):
    logger.debug("GEMS_ENFLAME SEGMENT_REDUCE")
    _check_reduce_and_dtype(data, reduce)
    axis = _wrap_axis(axis, data.dim())
    if indices is not None:
        raise RuntimeError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    if initial is None and lengths is not None and offsets is None:
        _check_index_tensor(data, lengths, "lengths", axis)
        if _is_unit_lengths(data, lengths, axis):
            return data.contiguous()

        data_contig = data.contiguous()
        uniform_result = _segment_reduce_uniform_lengths(
            data_contig, reduce, lengths, axis
        )
        if uniform_result is not None:
            return uniform_result

    axis, offsets_contig, output_shape, _ = _prepare_common(
        data, reduce, lengths, offsets, indices, axis, unsafe
    )

    data_contig = data.contiguous()
    output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
    if output.numel() == 0:
        return output

    segment_count = output_shape[axis]
    inner_size = _prod(data_contig.shape[axis + 1 :])
    data_size_axis = data_contig.shape[axis]
    has_initial, initial_value = _make_initial(reduce, initial)
    num_outputs = output.numel()
    grid = (min(_GRID_SIZE, num_outputs), 1, 1)

    with torch_device_fn.device(data.device):
        _segment_reduce_forward_kernel[grid](
            data_contig,
            offsets_contig,
            output,
            num_outputs,
            segment_count,
            inner_size,
            data_size_axis,
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            has_initial,
            initial_value,
            BLOCK_SIZE=_get_block_size(data.device),
            GRID_SIZE=min(_GRID_SIZE, num_outputs),
            NUM_STAGES=4,
            num_warps=4,
        )
    return output


def segment_reduce_out(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
    out,
):
    logger.debug("GEMS_ENFLAME SEGMENT_REDUCE_OUT")
    result = segment_reduce(
        data,
        reduce,
        lengths=lengths,
        indices=indices,
        offsets=offsets,
        axis=axis,
        unsafe=unsafe,
        initial=initial,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out


def _segment_reduce_backward(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
    out_buffer=None,
):
    logger.debug("GEMS_ENFLAME _SEGMENT_REDUCE_BACKWARD")
    # out_buffer lets the *_out variants receive the kernel result in place
    # instead of computing a temporary plus a full copy_.
    reuse_buffer = (
        out_buffer
        if out_buffer is not None
        and out_buffer.dtype == grad.dtype
        and out_buffer.is_contiguous()
        else None
    )
    if (
        initial is None
        and lengths is not None
        and offsets is None
        and reduce in _SUPPORTED_REDUCES
    ):
        _check_reduce_and_dtype(data, reduce)
        axis = _wrap_axis(axis, data.dim())
        _check_index_tensor(data, lengths, "lengths", axis)
        if _is_unit_lengths(data, lengths, axis):
            return grad.contiguous()

    if lengths is not None and offsets is None and reduce in ("sum", "mean"):
        _check_reduce_and_dtype(data, reduce)
        axis = _wrap_axis(axis, data.dim())
        _check_index_tensor(data, lengths, "lengths", axis)
        data_contig = data.contiguous()
        grad_contig = grad.contiguous()
        buffer = (
            reuse_buffer.reshape(data_contig.shape)
            if reuse_buffer is not None
            and reuse_buffer.numel() == data_contig.numel()
            else None
        )
        uniform_result = _segment_reduce_uniform_sum_mean_backward(
            data_contig, grad_contig, reduce, lengths, axis, buffer
        )
        if uniform_result is not None:
            return uniform_result
    if lengths is not None and offsets is None and reduce in ("max", "min", "prod"):
        _check_reduce_and_dtype(data, reduce)
        axis = _wrap_axis(axis, data.dim())
        _check_index_tensor(data, lengths, "lengths", axis)
        data_contig = data.contiguous()
        grad_contig = grad.contiguous()
        output_contig = output.contiguous()
        buffer = (
            reuse_buffer.reshape(data_contig.shape)
            if reuse_buffer is not None
            and reuse_buffer.numel() == data_contig.numel()
            else None
        )
        uniform_result = _segment_reduce_uniform_other_backward(
            data_contig, output_contig, grad_contig, reduce, lengths, axis, initial, buffer
        )
        if uniform_result is not None:
            return uniform_result

    axis, offsets_contig, output_shape, _ = _prepare_common(
        data, reduce, lengths, offsets, None, axis, True
    )
    data_contig = data.contiguous()
    grad_contig = grad.contiguous()
    output_contig = output.contiguous()
    grad_input = torch.zeros(data_contig.shape, dtype=grad.dtype, device=grad.device)

    if output_contig.numel() == 0:
        return grad_input

    segment_count = output_shape[axis]
    inner_size = _prod(data_contig.shape[axis + 1 :])
    data_size_axis = data_contig.shape[axis]
    _, initial_prod_value = _make_initial("prod", initial)
    num_outputs = output_contig.numel()
    grid = (min(_GRID_SIZE, num_outputs), 1, 1)

    with torch_device_fn.device(data.device):
        _segment_reduce_backward_kernel[grid](
            grad_contig,
            output_contig,
            data_contig,
            offsets_contig,
            grad_input,
            num_outputs,
            segment_count,
            inner_size,
            data_size_axis,
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            initial_prod_value,
            BLOCK_SIZE=_get_block_size(data.device),
            GRID_SIZE=min(_GRID_SIZE, num_outputs),
            NUM_STAGES=4,
            num_warps=4,
        )
    return grad_input


def _segment_reduce_backward_out(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
    out,
):
    logger.debug("GEMS_ENFLAME _SEGMENT_REDUCE_BACKWARD_OUT")
    buffer = None
    if out.is_contiguous() and out.dtype == grad.dtype and out.numel() == data.numel():
        buffer = out.reshape(-1)
    result = _segment_reduce_backward(
        grad,
        output,
        data,
        reduce,
        lengths=lengths,
        offsets=offsets,
        axis=axis,
        initial=initial,
        out_buffer=buffer,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    if result.data_ptr() != out.data_ptr():
        out.copy_(result)
    return out
