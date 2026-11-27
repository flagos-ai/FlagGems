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

"""Ascend specialization for replay-safe scatter reductions."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter_reduce import (
    _scatter_reduce_high_rank,
    _scatter_reduce_prod_scan,
)
from flag_gems.ops.scatter_reduce import scatter_reduce as _generic_scatter_reduce
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Small products are cheaper to scan once per destination.
_SOURCE_PRODUCT_MIN_INDICES = 1 << 18

_TRITON_VERSION = tuple(int(part) for part in triton.__version__.split(".")[:2])
_REPLAY_ATOMICS_MIN_TRITON = (3, 5)
_USE_REPLAY_ATOMICS = _TRITON_VERSION >= _REPLAY_ATOMICS_MIN_TRITON

_MAX_SEGMENT_PROGRAMS = 1 << 15
_SEGMENT_BLOCK = 256
# The generic sum, mean and extrema kernels process 128 * 4 indices per
# program. Ascend accepts at most 65535 programs in one launch.
_MAX_ATOMIC_ELEMENTS = 65535 * 128 * 4


@triton.jit
def _segment_sum(left_key, left, right_key, right):
    return right_key, tl.where(left_key == right_key, left + right, right)


@triton.jit
def _segment_prod(left_key, left, right_key, right):
    return right_key, tl.where(left_key == right_key, left * right, right)


@triton.jit
def _segment_min(left_key, left, right_key, right):
    value = tl.minimum(left, right, propagate_nan=tl.PropagateNan.ALL)
    return right_key, tl.where(left_key == right_key, value, right)


@triton.jit
def _segment_max(left_key, left, right_key, right):
    value = tl.maximum(left, right, propagate_nan=tl.PropagateNan.ALL)
    return right_key, tl.where(left_key == right_key, value, right)


@libentry()
@triton.jit
def _scatter_reduce_pack_kernel(
    key_ptr,
    order_ptr,
    src_ptr,
    flat_key_ptr,
    value_ptr,
    num_values,
    index_width,
    line_width,
    src_row_stride,
    src_col_stride,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(num_values, BLOCK), tl.num_programs(0)):
        pos = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = pos < num_values
        row = pos // index_width
        col = tl.load(order_ptr + pos, mask=mask, other=0)
        key = tl.load(key_ptr + row * index_width + col, mask=mask, other=0)
        value = tl.load(
            src_ptr + row * src_row_stride + col * src_col_stride, mask=mask, other=0
        )
        tl.store(flat_key_ptr + pos, key + row * line_width, mask=mask)
        tl.store(value_ptr + pos, value, mask=mask)


@libentry()
@triton.jit
def _scatter_reduce_segment_kernel(
    key_ptr,
    value_ptr,
    input_count_ptr,
    inp_ptr,
    out_ptr,
    next_key_ptr,
    next_value_ptr,
    next_count_ptr,
    num_values,
    num_tiles,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        base = tile.to(tl.int64) * BLOCK
        size = tl.minimum(num_values - base, BLOCK).to(tl.int32)
        pos = base + tl.minimum(lane, size - 1)
        keys = tl.load(key_ptr + pos)
        values = tl.load(value_ptr + pos)
        if REDUCE == "prod":
            _, reduced = tl.associative_scan((keys, values), 0, _segment_prod)
        elif REDUCE == "amin":
            _, reduced = tl.associative_scan((keys, values), 0, _segment_min)
        elif REDUCE == "amax":
            _, reduced = tl.associative_scan((keys, values), 0, _segment_max)
        else:
            _, reduced = tl.associative_scan((keys, values), 0, _segment_sum)
        if REDUCE == "mean":
            if input_count_ptr is None:
                counts = tl.full((BLOCK,), 1, tl.int64)
            else:
                counts = tl.load(input_count_ptr + pos)
            _, lengths = tl.associative_scan((keys, counts), 0, _segment_sum)

        next_key = tl.load(key_ptr + base + tl.minimum(lane + 1, size - 1))
        end = (lane < size) & ((keys != next_key) | (lane == size - 1))
        if next_key_ptr is not None:
            # Interior runs are complete. Only the two edge runs can continue
            # in an adjacent tile, so reduce their partials in the next launch.
            first = tl.load(key_ptr + base)
            last = tl.load(key_ptr + base + size - 1)
            first_end = tl.sum(((lane < size) & (keys == first)).to(tl.int32), 0) - 1
            edge = tl.arange(0, 2)
            edge_key = tl.where(edge == 0, first, last)
            # Select the edge lanes by bit pattern to preserve signed zeros and NaNs.
            bits = reduced.to(tl.int32, bitcast=True)
            first_bits = tl.sum(tl.where(lane == first_end, bits, 0), 0)
            last_bits = tl.sum(tl.where(lane == size - 1, bits, 0), 0)
            edge_value = tl.where(edge == 0, first_bits, last_bits).to(
                tl.float32, bitcast=True
            )
            duplicate = (edge == 1) & (first == last)
            partial = tile.to(tl.int64) * 2 + edge
            if REDUCE == "prod":
                edge_value = tl.where(duplicate, 1.0, edge_value)
            elif REDUCE == "sum" or REDUCE == "mean":
                edge_value = tl.where(duplicate, 0.0, edge_value)
            tl.store(next_key_ptr + partial, edge_key)
            tl.store(next_value_ptr + partial, edge_value)
            if REDUCE == "mean":
                first_count = tl.sum(tl.where(lane == first_end, lengths, 0), 0)
                last_count = tl.sum(tl.where(lane == size - 1, lengths, 0), 0)
                edge_count = tl.where(edge == 0, first_count, last_count)
                tl.store(next_count_ptr + partial, tl.where(duplicate, 0, edge_count))
            end = end & (keys != first) & (keys != last)

        # A completed run owns its destination. Read self from the original
        # input so replaying this assignment cannot apply a reduction twice.
        if INCLUDE_SELF:
            initial = tl.load(inp_ptr + keys, mask=end, other=0).to(tl.float32)
            if REDUCE == "prod":
                reduced *= initial
            elif REDUCE == "amin":
                reduced = tl.minimum(
                    reduced, initial, propagate_nan=tl.PropagateNan.ALL
                )
            elif REDUCE == "amax":
                reduced = tl.maximum(
                    reduced, initial, propagate_nan=tl.PropagateNan.ALL
                )
            else:
                reduced += initial
        if REDUCE == "mean":
            reduced /= tl.maximum(lengths.to(tl.float32) + INCLUDE_SELF, 1.0)
        tl.store(out_ptr + keys, reduced, mask=end)


@libentry()
@triton.jit
def _scatter_merge_order_kernel(
    keys, prev, out, n, width, run, steps, BLOCK: tl.constexpr
):
    for tile in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        offset = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < n
        col = offset % width
        row = offset - col
        if prev is None:
            order = col
        else:
            order = tl.load(prev + offset, mask=valid, other=0)
        key = tl.load(keys + row + order, mask=valid, other=0)
        start = (col // (2 * run)) * (2 * run)
        right = col >= start + run
        other = tl.where(right, start, start + run)
        low = tl.minimum(other, width)
        high = tl.minimum(other + run, width)
        # Equal keys from the left run precede those from the right run.
        for _ in range(steps):
            mid = (low + high) // 2
            probe = valid & (low < high)
            if prev is None:
                peer = mid
            else:
                peer = tl.load(prev + row + mid, mask=probe, other=0)
            other_key = tl.load(keys + row + peer, mask=probe, other=0)
            before = (other_key < key) | (right & (other_key == key))
            low = tl.where(probe & before, mid + 1, low)
            high = tl.where(probe & ~before, mid, high)
        own = tl.where(right, start + run, start)
        rank = start + col - own + low - tl.minimum(other, width)
        tl.store(out + row + rank, order, mask=valid)


def _scatter_merge_order(keys):
    # Merge sorted runs using exact int64 keys. Triton 3.2 cannot index the
    # constexpr stride tuples used by the vector argsort in later runtimes.
    width = keys.size(1)
    order = None
    scratch = torch.empty_like(keys, dtype=torch.int64)
    alternate = torch.empty_like(scratch)
    grid = (min(triton.cdiv(keys.numel(), _SEGMENT_BLOCK), _MAX_SEGMENT_PROGRAMS),)
    with torch_device_fn.device(keys.device):
        for step in range(max(1, (width - 1).bit_length())):
            _scatter_merge_order_kernel[grid](
                keys,
                order,
                scratch,
                keys.numel(),
                width,
                1 << step,
                step + 1,
                BLOCK=_SEGMENT_BLOCK,
            )
            order = scratch
            scratch, alternate = alternate, scratch
    return order


@triton.jit
def _scatter_lower_bound(keys, dest, row, width, steps, valid):
    low = tl.full(dest.shape, 0, tl.int64)
    high = tl.full(dest.shape, width, tl.int64)
    for _ in range(steps):
        mid = (low + high) // 2
        probe = valid & (low < high)
        key = tl.load(keys + row * width + mid, mask=probe, other=0)
        low = tl.where(probe & (key < dest), mid + 1, low)
        high = tl.where(probe & (key >= dest), mid, high)
    return row * width + low


@triton.jit
def _scatter_product(a, b):
    return a * b


@libentry()
@triton.jit
def _scatter_reduce_sorted_output_kernel(
    keys,
    values,
    inp,
    out,
    n,
    width,
    line_width,
    steps,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Own consecutive destinations. Triton 3.2 can fault on the indirect
    # vector stores used to finish sparse segments on newer runtimes.
    for tile in range(tl.program_id(0), tl.cdiv(n, ROWS), tl.num_programs(0)):
        dest = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row = dest // line_width
        valid = dest < n
        start = _scatter_lower_bound(keys, dest, row, width, steps, valid)
        end = _scatter_lower_bound(keys, dest + 1, row, width, steps, valid)
        count = end - start
        rounds = tl.cdiv(tl.max(count, 0), BLOCK)
        identity: tl.constexpr = 1.0 if REDUCE == "prod" else 0.0
        acc = tl.full((ROWS,), identity, tl.float32)
        offs = start[:, None] + tl.arange(0, BLOCK)[None, :]
        for _ in range(rounds):
            block = tl.load(values + offs, mask=offs < end[:, None], other=identity)
            if REDUCE == "prod":
                acc *= tl.reduce(block, 1, _scatter_product)
            else:
                acc += tl.sum(block, 1)
            offs += BLOCK
        initial = tl.load(inp + dest, mask=valid, other=0).to(tl.float32)
        if INCLUDE_SELF:
            if REDUCE == "prod":
                acc *= initial
            else:
                acc += initial
        if REDUCE == "mean":
            acc /= tl.maximum(count.to(tl.float32) + INCLUDE_SELF, 1.0)
        tl.store(out + dest, tl.where(count > 0, acc, initial), mask=valid)


def _reduce_packed_segments(
    flat_keys, sorted_src, line_input, active_result, reduce, include_self
):
    active_result.copy_(line_input)
    input_counts = None
    while True:
        num_values = flat_keys.numel()
        tiles = triton.cdiv(num_values, _SEGMENT_BLOCK)
        next_keys = next_values = next_counts = None
        if tiles > 1:
            next_keys = flat_keys.new_empty((2 * tiles,))
            next_values = sorted_src.new_empty((2 * tiles,))
            if reduce == "mean":
                next_counts = flat_keys.new_empty((2 * tiles,))
        _scatter_reduce_segment_kernel[(min(tiles, _MAX_SEGMENT_PROGRAMS),)](
            flat_keys,
            sorted_src,
            input_counts,
            line_input,
            active_result,
            next_keys,
            next_values,
            next_counts,
            num_values,
            tiles,
            REDUCE=reduce,
            INCLUDE_SELF=include_self,
            BLOCK=_SEGMENT_BLOCK,
        )
        if tiles == 1:
            break
        flat_keys, sorted_src, input_counts = next_keys, next_values, next_counts


def _reduce_packed_rows(keys, values, line_input, result, reduce, include_self):
    width = keys.size(1)
    outputs_per_program = 8
    grid = (
        min(triton.cdiv(result.numel(), outputs_per_program), _MAX_SEGMENT_PROGRAMS),
    )
    _scatter_reduce_sorted_output_kernel[grid](
        keys,
        values,
        line_input,
        result,
        result.numel(),
        width,
        line_input.size(-1),
        width.bit_length(),
        REDUCE=reduce,
        INCLUDE_SELF=include_self,
        ROWS=outputs_per_program,
        BLOCK=32,
    )


if _USE_REPLAY_ATOMICS:
    from .argsort import argsort as _scatter_order

    _reduce_packed = _reduce_packed_segments
else:
    _scatter_order = _scatter_merge_order
    _reduce_packed = _reduce_packed_rows


def _scatter_reduce_sorted_segments(inp, dim, index, src, reduce, include_self):
    """Reduce equal destinations on device, including segments spanning tiles."""
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    dim %= inp.ndim

    index_shape = tuple(int(size) for size in index.shape)
    active_inp = inp
    active_src = src
    for axis, index_size in enumerate(index_shape):
        active_src = active_src.narrow(axis, 0, index_size)
        if axis != dim:
            active_inp = active_inp.narrow(axis, 0, index_size)

    # Destinations from different lines cannot collide. Sort each line along
    # the scatter dimension, then number its destinations in a disjoint range.
    line_size = inp.size(dim)
    line_layout = active_inp.movedim(dim, -1)
    destination_keys = index.movedim(dim, -1).reshape(-1, index.size(dim)).contiguous()
    src_rows = active_src.movedim(dim, -1).reshape_as(destination_keys)
    # Native int64 sort can use AiCPU. Both Triton routes keep the exact keys.
    order = _scatter_order(destination_keys)
    num_values = order.numel()
    flat_keys = torch.empty_like(order)
    sorted_src = src.new_empty((num_values,), dtype=torch.float32)
    line_input = line_layout.contiguous()
    active_result = torch.empty_like(line_input)

    with torch_device_fn.device(inp.device):
        pack_grid = (
            min(triton.cdiv(num_values, _SEGMENT_BLOCK), _MAX_SEGMENT_PROGRAMS),
        )
        _scatter_reduce_pack_kernel[pack_grid](
            destination_keys,
            order,
            src_rows,
            flat_keys,
            sorted_src,
            num_values,
            index.size(dim),
            line_size,
            *src_rows.stride(),
            BLOCK=_SEGMENT_BLOCK,
        )
        _reduce_packed(
            flat_keys, sorted_src, line_input, active_result, reduce, include_self
        )

    active_result = active_result.movedim(-1, dim).contiguous()

    active_domain_is_full = all(
        axis == dim or index_size == inp.size(axis)
        for axis, index_size in enumerate(index_shape)
    )
    if active_domain_is_full:
        return active_result

    result = torch.empty_like(inp, memory_format=torch.contiguous_format)
    result.copy_(inp)
    result_active = result
    for axis, index_size in enumerate(index_shape):
        if axis != dim:
            result_active = result_active.narrow(axis, 0, index_size)
    result_active.copy_(active_result)
    return result


def _scatter_reduce_prod(inp, dim, index, src, include_self):
    if inp.ndim > 5:
        return _scatter_reduce_high_rank(
            inp,
            dim,
            index,
            src,
            "prod",
            include_self,
            use_prod_scan=True,
            materialize_product=True,
        )
    return _scatter_reduce_prod_scan(
        inp,
        dim,
        index,
        src,
        include_self,
        materialize_product=True,
    )


def _scatter_reduce_with_segments(inp, dim, index, src, reduce, include_self):
    """Use scan/sorted-segment reductions for runtimes without replay atomics.

    CANN 8.5 replay is not monotonic in source size, so non-idempotent sum and
    mean updates must never fall back to the generic source-atomic kernels.
    """
    if index.numel() == 0:
        return _generic_scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
        )
    source_product = index.numel() >= _SOURCE_PRODUCT_MIN_INDICES or inp.numel() > 65535
    if reduce in ("sum", "mean") or (reduce == "prod" and source_product):
        return _scatter_reduce_sorted_segments(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self,
        )
    if reduce == "prod":
        return _scatter_reduce_prod(inp, dim, index, src, include_self)
    return _generic_scatter_reduce(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
    )


def _scatter_reduce_with_claims(inp, dim, index, src, reduce, include_self):
    """Claim sum/mean updates on runtimes that can replay source programs."""
    if (
        index.numel()
        and max(index.numel(), inp.numel()) > _MAX_ATOMIC_ELEMENTS
        and reduce in ("sum", "mean", "amin", "amax")
    ):
        return _scatter_reduce_sorted_segments(
            inp, dim, index, src, reduce, include_self
        )
    if reduce == "prod":
        # At large sizes the per-destination product lock can time out on 910B.
        return _scatter_reduce_with_segments(inp, dim, index, src, reduce, include_self)
    return _generic_scatter_reduce(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
        _deduplicate_programs=reduce in ("sum", "mean"),
    )


# Bind the runtime strategy once when this backend module is imported. Triton
# 3.2 reduces each sorted row by destination because it cannot safely lower
# claim atomics. Triton 3.5+ can claim sum/mean updates. Both
# paths use the same scan or sorted segments for products.
_scatter_reduce = (
    _scatter_reduce_with_claims
    if _USE_REPLAY_ATOMICS
    else _scatter_reduce_with_segments
)


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_ASCEND SCATTER_REDUCE_TWO")
    return _scatter_reduce(inp, dim, index, src, reduce, include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_ASCEND SCATTER_REDUCE_TWO_")
    result = _scatter_reduce(inp, dim, index, src, reduce, include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(
    inp,
    dim,
    index,
    src,
    reduce,
    *,
    include_self=True,
    out=None,
):
    logger.debug("GEMS_ASCEND SCATTER_REDUCE_TWO_OUT")
    if out is not None and out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    result = _scatter_reduce(inp, dim, index, src, reduce, include_self)
    if out is not None:
        out.copy_(result)
        return out
    return result
