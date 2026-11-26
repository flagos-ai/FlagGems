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

_SOURCE_PRODUCT_MIN_INDICES = 1 << 18
# Below this size, the output-centric product scan is cheaper; at and above it,
# the selected runtime uses sorted segments or claimed source programs. Both
# sides are replay-safe, so this threshold is only a product performance gate.

_TRITON_VERSION = tuple(int(part) for part in triton.__version__.split(".")[:2])
_REPLAY_ATOMICS_MIN_TRITON = (3, 5)
_USE_REPLAY_ATOMICS = _TRITON_VERSION >= _REPLAY_ATOMICS_MIN_TRITON
# CANN 8.5 ships Triton 3.2 and CANN 9 ships Triton 3.5. The older compiler
# cannot lower the replay-claim and product-lock kernels, so treat the two
# runtime generations as separate Ascend backends while sharing this module.
# Resolve this capability once at import time; public calls use the statically
# selected implementation below and do not compare versions on the hot path.

_MAX_SEGMENT_PROGRAMS = 1 << 15
_SEGMENT_BLOCK = 256


@triton.jit
def _segment_sum(left_key, left, right_key, right):
    return right_key, tl.where(left_key == right_key, left + right, right)


@triton.jit
def _segment_prod(left_key, left, right_key, right):
    return right_key, tl.where(left_key == right_key, left * right, right)


@libentry()
@triton.jit
def _scatter_reduce_segment_kernel(
    key_ptr,
    value_ptr,
    input_count_ptr,
    contribution_ptr,
    present_ptr,
    count_ptr,
    next_key_ptr,
    next_value_ptr,
    next_count_ptr,
    num_values,
    num_tiles,
    IS_PRODUCT: tl.constexpr,
    WRITE_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        base = tile.to(tl.int64) * BLOCK
        size = tl.minimum(num_values - base, BLOCK).to(tl.int32)
        pos = base + tl.minimum(lane, size - 1)
        keys = tl.load(key_ptr + pos)
        values = tl.load(value_ptr + pos)
        if IS_PRODUCT:
            _, reduced = tl.associative_scan((keys, values), 0, _segment_prod)
        else:
            _, reduced = tl.associative_scan((keys, values), 0, _segment_sum)
        if WRITE_COUNT:
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
            edge_lane = tl.where(edge == 0, first_end, size - 1)
            edge_key = tl.where(edge == 0, first, last)
            edge_value = tl.gather(reduced, edge_lane, 0)
            duplicate = (edge == 1) & (first == last)
            partial = tile.to(tl.int64) * 2 + edge
            tl.store(next_key_ptr + partial, edge_key)
            tl.store(
                next_value_ptr + partial,
                tl.where(duplicate, 1.0 if IS_PRODUCT else 0.0, edge_value),
            )
            if WRITE_COUNT:
                first_count = tl.sum(tl.where(lane == first_end, lengths, 0), 0)
                last_count = tl.sum(tl.where(lane == size - 1, lengths, 0), 0)
                edge_count = tl.where(edge == 0, first_count, last_count)
                tl.store(next_count_ptr + partial, tl.where(duplicate, 0, edge_count))
            end = end & (keys != first) & (keys != last)

        tl.store(contribution_ptr + keys, reduced, mask=end)
        tl.store(present_ptr + keys, 1, mask=end)
        if WRITE_COUNT:
            tl.store(count_ptr + keys, lengths.to(tl.float32), mask=end)


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
    destination_keys = index.movedim(dim, -1).reshape(-1, index.size(dim))
    src_rows = active_src.movedim(dim, -1).reshape_as(destination_keys)
    sorted_keys, order = torch.sort(destination_keys)
    rows = torch.arange(
        destination_keys.size(0), dtype=torch.int64, device=index.device
    )
    sorted_keys = (sorted_keys + rows[:, None] * line_size).reshape(-1)
    sorted_src = src_rows[rows[:, None], order].to(torch.float32).reshape(-1)

    inp_f32 = line_layout.contiguous().to(torch.float32)
    if reduce == "prod":
        contribution = torch.ones_like(inp_f32).reshape(-1)
    else:
        contribution = torch.zeros_like(inp_f32).reshape(-1)
    present = torch.zeros(
        contribution.shape,
        dtype=torch.int32,
        device=inp.device,
    )
    count = (
        torch.zeros_like(contribution)
        if reduce == "mean"
        else torch.empty(1, dtype=torch.float32, device=inp.device)
    )

    input_counts = None
    with torch_device_fn.device(inp.device):
        while True:
            num_values = sorted_keys.numel()
            tiles = triton.cdiv(num_values, _SEGMENT_BLOCK)
            next_keys = next_values = next_counts = None
            if tiles > 1:
                next_keys = sorted_keys.new_empty((2 * tiles,))
                next_values = sorted_src.new_empty((2 * tiles,))
                if reduce == "mean":
                    next_counts = sorted_keys.new_empty((2 * tiles,))
            _scatter_reduce_segment_kernel[(min(tiles, _MAX_SEGMENT_PROGRAMS),)](
                sorted_keys,
                sorted_src,
                input_counts,
                contribution,
                present,
                count,
                next_keys,
                next_values,
                next_counts,
                num_values,
                tiles,
                IS_PRODUCT=reduce == "prod",
                WRITE_COUNT=reduce == "mean",
                BLOCK=_SEGMENT_BLOCK,
            )
            if tiles == 1:
                break
            sorted_keys, sorted_src, input_counts = next_keys, next_values, next_counts

    inp_flat = inp_f32.reshape(-1)
    has_contribution = present != 0
    if reduce == "sum":
        active_result = (
            inp_flat + contribution
            if include_self
            else torch.where(has_contribution, contribution, inp_flat)
        )
    elif reduce == "prod":
        active_result = (
            inp_flat * contribution
            if include_self
            else torch.where(has_contribution, contribution, inp_flat)
        )
    else:
        if include_self:
            active_result = (inp_flat + contribution) / (count + 1.0)
        else:
            active_result = torch.where(
                has_contribution,
                contribution / torch.clamp(count, min=1.0),
                inp_flat,
            )
    active_result = (
        active_result.to(inp.dtype)
        .reshape(line_layout.shape)
        .movedim(-1, dim)
        .contiguous()
    )

    active_domain_is_full = all(
        axis == dim or index_size == inp.size(axis)
        for axis, index_size in enumerate(index_shape)
    )
    if active_domain_is_full:
        return active_result

    result = inp.contiguous().clone()
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
    """Use program claims and a product lock on replay-safe newer runtimes.

    CANN 9 can also replay below the product performance boundary. Claim every
    non-idempotent source program instead of using tensor size as a safety gate.
    """
    source_product = index.numel() >= _SOURCE_PRODUCT_MIN_INDICES or inp.numel() > 65535
    if reduce == "prod" and not source_product:
        return _scatter_reduce_prod(inp, dim, index, src, include_self)
    replay_sensitive = reduce in ("sum", "prod", "mean")
    return _generic_scatter_reduce(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
        _deduplicate_programs=replay_sensitive,
        _use_product_lock=reduce == "prod",
    )


# Bind the runtime strategy once when this backend module is imported. Triton
# 3.2 uses sorted, replay-idempotent segment assignments because it cannot
# safely lower the claim/lock atomics; Triton 3.5+ keeps the faster
# source-driven kernels and prevents replay with a program claim and product
# lock. Static binding avoids a version branch on every operator invocation.
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
