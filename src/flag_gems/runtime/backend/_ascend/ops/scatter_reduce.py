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
import math

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
# The sorted-segment kernel bounds its launch grid and lets each program scan
# multiple candidate boundaries. Each segment writes one complete output
# contribution, so a runtime replay repeats an idempotent assignment.


@libentry()
@triton.jit(do_not_specialize=["num_segments"])
def _scatter_reduce_segment_kernel(
    sorted_src_ptr,
    sorted_key_ptr,
    contribution_ptr,
    present_ptr,
    count_ptr,
    num_values,
    IS_PRODUCT: tl.constexpr,
    WRITE_COUNT: tl.constexpr,
):
    """Identify and reduce sorted segments with idempotent output stores.

    This path is selected for Triton 3.2, where the source-driven replay claim
    and product lock used by newer Ascend runtimes cannot be lowered safely.
    """
    position = tl.program_id(axis=0).to(tl.int64)
    position_stride = tl.num_programs(axis=0).to(tl.int64)

    while position < num_values:
        output_offset = tl.load(sorted_key_ptr + position).to(tl.int64)
        previous_key = tl.load(
            sorted_key_ptr + position - 1,
            mask=position > 0,
            other=-1,
        ).to(tl.int64)
        is_segment_start = (position == 0) | (output_offset != previous_key)
        cursor = position
        segment_active = is_segment_start
        length = position * 0
        if IS_PRODUCT:
            accumulator = 1.0
        else:
            accumulator = 0.0

        while segment_active:
            value = tl.load(sorted_src_ptr + cursor).to(tl.float32)
            if IS_PRODUCT:
                accumulator *= value
            else:
                accumulator += value
            length += 1
            cursor += 1
            next_key = tl.load(
                sorted_key_ptr + cursor,
                mask=cursor < num_values,
                other=-1,
            ).to(tl.int64)
            segment_active = (cursor < num_values) & (next_key == output_offset)

        tl.store(
            contribution_ptr + output_offset,
            accumulator,
            mask=is_segment_start,
        )
        tl.store(present_ptr + output_offset, 1, mask=is_segment_start)
        if WRITE_COUNT:
            tl.store(
                count_ptr + output_offset,
                length.to(tl.float32),
                mask=is_segment_start,
            )
        position += position_stride


def _scatter_reduce_sorted_segments(inp, dim, index, src, reduce, include_self):
    """Run a large legacy-runtime reduction through sorted segments.

    Triton 3.2 cannot compile the newer replay-claim and lock paths. Sorting
    destination offsets groups each output's sources, and the kernel assigns a
    complete contribution once, without atomics or CPU fallback.
    """
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

    outer = math.prod(index_shape[:dim])
    inner = math.prod(index_shape[dim + 1 :])
    active_shape = tuple(active_inp.shape)
    inp_3d = active_inp.contiguous().reshape(outer, inp.size(dim), inner)
    index_3d = index.contiguous().reshape(outer, index_shape[dim], inner)
    src_3d = active_src.contiguous().reshape(outer, index_shape[dim], inner)

    source_positions = torch.arange(
        index_3d.numel(),
        dtype=torch.int64,
        device=index.device,
    )
    outer_offsets = source_positions // (index_shape[dim] * inner)
    inner_offsets = source_positions % inner
    destination_keys = (
        outer_offsets * (inp.size(dim) * inner)
        + index_3d.reshape(-1) * inner
        + inner_offsets
    )
    sorted_keys, order = torch.sort(destination_keys)
    sorted_src = src_3d.to(torch.float32).reshape(-1)[order]

    inp_f32 = inp_3d.to(torch.float32)
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

    num_values = sorted_keys.numel()
    grid = (min(num_values, _MAX_SEGMENT_PROGRAMS),)
    with torch_device_fn.device(inp.device):
        _scatter_reduce_segment_kernel[grid](
            sorted_src,
            sorted_keys,
            contribution,
            present,
            count,
            num_values,
            reduce == "prod",
            reduce == "mean",
        )

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
    active_result = active_result.to(inp.dtype).reshape(active_shape)

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
    source_product = index.numel() >= _SOURCE_PRODUCT_MIN_INDICES
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
    source_product = index.numel() >= _SOURCE_PRODUCT_MIN_INDICES
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
