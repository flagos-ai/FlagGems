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

"""Source-order FP32 index_add for contiguous MetaX tensors."""

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

# Conservative compiler/resource bounds inherited from the validated prototype,
# not performance crossovers. The caller retains its legacy path outside them.
_MAX_SORT_EXTENT = 8192
_MAX_SCAN_BLOCKS = 8192
_SCAN_BLOCK = 512
_MAX_REDUCTION_PROGRAMS = 2**31 - 1


@libentry()
@triton.jit
def _clear_counts(counts, info, N: tl.constexpr, B: tl.constexpr):
    r = ext.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(counts + r, 0, r < N)
    if ext.program_id(0) == 0:
        tl.store(info + 2, 0)


@libentry()
@triton.jit
def _count_receivers(
    index,
    counts,
    info,
    N: tl.constexpr,
    OUT_DIM: tl.constexpr,
    STRIDE: tl.constexpr,
    B: tl.constexpr,
):
    edge = ext.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    receiver = tl.load(index + edge * STRIDE, edge < N, 0).to(tl.int64)
    valid = (receiver >= 0) & (receiver < OUT_DIM)
    # Invalid values never form an out-of-range histogram pointer, even for
    # extreme int64 inputs. No floating destination is touched in this phase.
    safe_receiver = tl.where(valid, receiver, 0)
    tl.atomic_add(counts + safe_receiver, 1, (edge < N) & valid, sem="relaxed")
    invalid = tl.max(((edge < N) & ~valid).to(tl.int32))
    tl.atomic_max(info + 2, invalid, sem="relaxed")


@libentry()
@triton.jit
def _scan_counts(
    counts, offsets, sums, maxima, active, N: tl.constexpr, B: tl.constexpr
):
    block = ext.program_id(0)
    r = block.to(tl.int64) * B + tl.arange(0, B)
    count = tl.load(counts + r, r < N, 0)
    tl.store(offsets + r, tl.cumsum(count) - count, r < N)
    tl.store(sums + block, tl.sum(count))
    tl.store(maxima + block, tl.max(count))
    tl.store(active + block, tl.sum((count > 0).to(tl.int32)))


@libentry()
@triton.jit
def _scan_blocks(sums, maxima, active, info, N: tl.constexpr, B: tl.constexpr):
    block = tl.arange(0, B)
    total = tl.load(sums + block, block < N, 0)
    used = tl.load(active + block, block < N, 0)
    maximum = tl.load(maxima + block, block < N, 0)
    tl.store(sums + block, tl.cumsum(total) - total, block < N)
    tl.store(active + block, tl.cumsum(used) - used, block < N)
    tl.store(info, tl.max(maximum))
    tl.store(info + 1, tl.sum(used))


@libentry()
@triton.jit
def _finish_groups(
    counts,
    offsets,
    sums,
    active,
    receivers,
    grouped_offsets,
    info,
    N: tl.constexpr,
    EDGES: tl.constexpr,
    B: tl.constexpr,
):
    block = ext.program_id(0)
    r = block.to(tl.int64) * B + tl.arange(0, B)
    count = tl.load(counts + r, r < N, 0)
    offset = tl.load(offsets + r, r < N, 0) + tl.load(sums + block)
    tl.store(offsets + r, offset, r < N)
    used = (count > 0) & (r < N)
    group = tl.cumsum(used.to(tl.int32)) - 1 + tl.load(active + block)
    tl.store(receivers + group, r.to(tl.int32), used)
    tl.store(grouped_offsets + group, offset, used)
    # All local scan inputs have been consumed; reuse counts as cursors.
    tl.store(counts + r, 0, r < N)
    if block == 0:
        tl.store(grouped_offsets + tl.load(info + 1), EDGES)


@libentry()
@triton.jit
def _scatter_edges(
    index,
    cursors,
    offsets,
    edges,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    B: tl.constexpr,
):
    edge = ext.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    receiver = tl.load(index + edge * STRIDE, edge < N, 0).to(tl.int64)
    slot = tl.atomic_add(cursors + receiver, 1, edge < N, sem="relaxed")
    base = tl.load(offsets + receiver, edge < N, 0).to(tl.int64)
    tl.store(edges + base + slot.to(tl.int64), edge.to(tl.int32), edge < N)


@libentry()
@triton.jit
def _sort_edges(offsets, edges, DEGREE: tl.constexpr):
    group = ext.program_id(0)
    start = tl.load(offsets + group).to(tl.int64)
    stop = tl.load(offsets + group + 1).to(tl.int64)
    positions = start + tl.arange(0, DEGREE)
    values = tl.load(edges + positions, positions < stop, 2147483647)
    ordered = tl.sort(values, descending=False)
    tl.store(edges + positions, ordered, positions < stop)


def _sort_widths(max_degree):
    if max_degree <= 1:
        return ()
    width = triton.next_power_of_2(max_degree)
    return tuple(bound for bound in (256, 1024) if bound < width) + (width,)


def _prefer_bucketed_sort(group_count, max_degree, edge_count):
    """Select an adaptive sort only where there is spare padding work."""
    if max_degree <= 4:
        # Packing won at 8192 receivers in the C550 degree-three sweep.
        return 1 < max_degree and group_count >= 8192
    # The C550 sweep measured width 4096. Skip the extra launches on small
    # grids, other widths, and distributions with uncertain padding savings.
    widths = _sort_widths(max_degree)
    if group_count < 384 or widths != (256, 1024, 4096):
        return False
    extra_edges = edge_count - group_count
    slots = widths[0] * group_count
    for lower, upper in zip(widths, widths[1:]):
        # A group wider than lower must use at least lower extra edges.
        slots += (upper - lower) * min(group_count, extra_edges // lower)
    return 2 * slots <= widths[-1] * group_count


@libentry()
@triton.jit
def _sort_small_groups(offsets, edges, group_count, BLOCK: tl.constexpr):
    # Pack many degree-at-most-four receivers in a single program.
    groups = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group_mask = groups < group_count
    start = tl.load(offsets + groups, group_mask, 0).to(tl.int64)
    stop = tl.load(offsets + groups + 1, group_mask, 0).to(tl.int64)
    a = tl.load(edges + start, start < stop, 2147483647)
    b = tl.load(edges + start + 1, start + 1 < stop, 2147483647)
    c = tl.load(edges + start + 2, start + 2 < stop, 2147483647)
    d = tl.load(edges + start + 3, start + 3 < stop, 2147483647)
    a, b = tl.minimum(a, b), tl.maximum(a, b)
    c, d = tl.minimum(c, d), tl.maximum(c, d)
    a, c = tl.minimum(a, c), tl.maximum(a, c)
    b, d = tl.minimum(b, d), tl.maximum(b, d)
    b, c = tl.minimum(b, c), tl.maximum(b, c)
    tl.store(edges + start, a, start < stop)
    tl.store(edges + start + 1, b, start + 1 < stop)
    tl.store(edges + start + 2, c, start + 2 < stop)
    tl.store(edges + start + 3, d, start + 3 < stop)


@libentry()
@triton.jit
def _sort_degree_bucket(offsets, edges, LOWER: tl.constexpr, WIDTH: tl.constexpr):
    group = ext.program_id(0)
    start = tl.load(offsets + group).to(tl.int64)
    stop = tl.load(offsets + group + 1).to(tl.int64)
    degree = stop - start
    if (degree > LOWER) & (degree <= WIDTH):
        positions = start + tl.arange(0, WIDTH)
        values = tl.load(edges + positions, positions < stop, 2147483647)
        ordered = tl.sort(values, descending=False)
        tl.store(edges + positions, ordered, positions < stop)


def _sort_groups_bucketed(offsets, edges, group_count, max_degree):
    """Sort each nontrivial receiver by original source position in place."""
    if max_degree <= 1:
        return
    if max_degree <= 4:
        block = 128
        _sort_small_groups[(triton.cdiv(group_count, block),)](
            offsets, edges, group_count, block
        )
        return
    lower = 1
    for width in _sort_widths(max_degree):
        _sort_degree_bucket[(group_count,)](offsets, edges, lower, width)
        lower = width


@libentry()
@triton.jit
def _reduce_groups(
    out,
    src,
    offsets,
    edges,
    receivers,
    GROUPS: tl.constexpr,
    OUT_DIM: tl.constexpr,
    INDEX_LEN: tl.constexpr,
    SUFFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flatten features as well as receivers: do not rely on a backend's
    # secondary grid-dimension limit for a wide contiguous suffix.
    column_tiles = tl.cdiv(SUFFIX, BLOCK)
    program = ext.program_id(0).to(tl.int64)
    group_row = program // column_tiles
    group = group_row % GROUPS
    receiver = tl.load(receivers + group).to(tl.int64)
    prefix = group_row // GROUPS
    row = prefix * OUT_DIM + receiver
    cols = (program % column_tiles) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < SUFFIX
    # Widen before pointer arithmetic, not after loading the edge value.
    start = tl.load(offsets + group).to(tl.int64)
    stop = tl.load(offsets + group + 1).to(tl.int64)
    total = tl.load(out + row * SUFFIX + cols, mask, 0).to(tl.float32)
    for position in range(start, stop):
        edge = tl.load(edges + position).to(tl.int64)
        value = tl.load(src + (prefix * INDEX_LEN + edge) * SUFFIX + cols, mask, 0)
        total = total + value.to(tl.float32)
    tl.store(out + row * SUFFIX + cols, total, mask)


class _Grouping(NamedTuple):
    counts: torch.Tensor
    offsets: torch.Tensor
    sums: torch.Tensor
    active: torch.Tensor
    info: torch.Tensor
    max_degree: int
    active_count: int
    programs: int


def _can_prepare_groups(length, out_dim):
    return (
        length < 2**31
        and out_dim < 2**31
        and triton.cdiv(out_dim, _SCAN_BLOCK) <= _MAX_SCAN_BLOCKS
    )


def _prepare_grouped_suffix(index, out_dim, prefix, suffix):
    """Validate indices and build statistics with one host-result transfer.

    Call only inside the device context and the supported resource region.
    A None result means valid indices but an unsupported degree/grid: the
    caller may safely use its legacy reduction. Invalid indices always raise
    before a floating destination is allocated or written. No implicit cache.
    """
    length = index.numel()
    # Still launch the validity protocol when the indexed output axis is empty.
    blocks = max(1, triton.cdiv(out_dim, _SCAN_BLOCK))
    workspace = torch.empty(
        2 * out_dim + 3 * blocks + 3, dtype=torch.int32, device=index.device
    )
    counts = workspace[:out_dim]
    offsets = workspace[out_dim : 2 * out_dim]
    stats = workspace[2 * out_dim : -3].view(3, blocks)
    sums, maxima, active = stats.unbind(0)
    info = workspace[-3:]
    _clear_counts[(blocks,)](counts, info, out_dim, _SCAN_BLOCK)
    _count_receivers[(triton.cdiv(length, 256),)](
        index, counts, info, length, out_dim, index.stride(0), 256
    )
    _scan_counts[(blocks,)](counts, offsets, sums, maxima, active, out_dim, _SCAN_BLOCK)
    _scan_blocks[(1,)](
        sums, maxima, active, info, blocks, triton.next_power_of_2(blocks)
    )
    max_degree, active_count, invalid = info.cpu().tolist()
    if invalid:
        raise AssertionError("0 <= index < self.size(dim)")
    programs = prefix * active_count * triton.cdiv(suffix, 128)
    if max_degree > _MAX_SORT_EXTENT or programs > _MAX_REDUCTION_PROGRAMS:
        return None
    return _Grouping(
        counts, offsets, sums, active, info, max_degree, active_count, programs
    )


def _run_grouped_suffix(out, dim, index, src, alpha, grouping):
    """Finish per-call source-position grouping, then reduce in source order."""
    counts, offsets, sums, active, info, max_degree, active_count, programs = grouping
    length, out_dim = index.numel(), out.size(dim)
    blocks = max(1, triton.cdiv(out_dim, _SCAN_BLOCK))
    # Only active receivers need compact group storage. All these views share
    # one allocation; preparation's counts are reused as scatter cursors.
    workspace = torch.empty(
        2 * active_count + 1 + length, dtype=torch.int32, device=out.device
    )
    receivers = workspace[:active_count]
    grouped_offsets = workspace[active_count : 2 * active_count + 1]
    edges = workspace[2 * active_count + 1 :]
    _finish_groups[(blocks,)](
        counts,
        offsets,
        sums,
        active,
        receivers,
        grouped_offsets,
        info,
        out_dim,
        length,
        _SCAN_BLOCK,
    )
    _scatter_edges[(triton.cdiv(length, 256),)](
        index, counts, offsets, edges, length, index.stride(0), 256
    )
    if _prefer_bucketed_sort(active_count, max_degree, length):
        _sort_groups_bucketed(grouped_offsets, edges, active_count, max_degree)
    elif max_degree > 1:
        _sort_edges[(active_count,)](
            grouped_offsets, edges, triton.next_power_of_2(max_degree)
        )
    # Native rounds the scaled value before adding it. Keep the multiply
    # separate because MetaX can fuse it into the loop even with FMA disabled.
    if alpha != 1:
        src = src.mul(alpha)
    _reduce_groups[(programs,)](
        out,
        src,
        grouped_offsets,
        edges,
        receivers,
        active_count,
        out_dim,
        length,
        math.prod(src.shape[dim + 1 :]),
        BLOCK=128,
        enable_fp_fusion=False,
    )
