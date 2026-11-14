# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin override of flag_gems._nested_tensor_from_mask.
#
# Why the generic (public) implementation fails on XPU
# ----------------------------------------------------
# The generic lengths kernel (_nested_tensor_from_mask_lengths_kernel) detects
# left-alignment gaps with a second, **shifted-by-one** masked load:
#
#     m_next  = (offs + 1) < L
#     vals_next = tl.load(mask + n * L + offs + 1, mask=m_next, other=0)
#     gap_acc += ((vals == 0) & (vals_next == 1)).to(tl.int32)
#
# On the Kunlunxin (P800/XPU3) Triton backend the misaligned (offset+1) gather
# is scalarized and, for rows that are entirely True (count == L), the degraded
# path intermittently loses one lane: 300-trial A/B on shape (16, 64) shows the
# generic kernel wrong in 61/300 runs (always 64 -> 63 on a full-True row),
# while the same kernel with the shifted load removed is 0/300 wrong. The wrong
# `lengths` then also corrupts offsets and the compacted values buffer.
#
# Fix: never use a shifted load for the gap check. A row is left-aligned
# (True prefix, no gap) iff  count_trues == index_of_first_false  (L if all
# True). The kernel below computes both from a single aligned load per tile and
# stores gaps[n] = (count != first_false), which is exactly the public gap
# flag. Everything else (offsets scan, compaction, strided-layout nested view)
# matches the generic implementation and is verified correct on XPU.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _nested_tensor_from_mask_lengths_kernel(
    mask,
    lengths,
    gaps,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Number of valid (True) entries per batch row, plus a per-row flag for a
    left-alignment gap (a False followed by a True), computed from a single
    aligned load per tile (no shifted load; see module docstring).
    """
    n = tle.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    first_false = tl.full((BLOCK,), L, dtype=tl.int32)
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        m = offs < L
        vals = tl.load(mask + n * L + offs, mask=m, other=0).to(tl.int32)
        acc += vals
        first_false = tl.minimum(first_false, tl.where(vals == 0, offs, L))
    count = tl.sum(acc, axis=0).to(tl.int64)
    gap = tl.min(first_false, axis=0)
    tl.store(lengths + n, count)
    tl.store(gaps + n, (count != gap).to(tl.int8))


@libentry()
@triton.jit
def _nested_tensor_from_mask_offsets_kernel(
    lengths,
    gaps,
    offsets,
    has_gap,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-block exclusive scan of `lengths` to produce offsets, together
    with the total number of valid rows and a scalar flag reporting whether any
    row of the mask contains a gap.
    """
    idx = tl.arange(0, BLOCK)
    m = idx < N
    vals = tl.load(lengths + idx, mask=m, other=0).to(tl.int64)
    gap_vals = tl.load(gaps + idx, mask=m, other=0).to(tl.int64)
    inclusive = tl.cumsum(vals, axis=0)
    exclusive = inclusive - vals
    tl.store(offsets + idx, exclusive, mask=m)
    tl.store(offsets + N, tl.sum(vals, axis=0))
    tl.store(has_gap, (tl.sum(gap_vals, axis=0) > 0).to(tl.int8))


@libentry()
@triton.jit
def _nested_tensor_from_mask_compact_kernel(
    t,
    lengths,
    offsets,
    values,
    D: tl.constexpr,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compact the valid rows of each batch into a contiguous value buffer.

    For batch `n`, the first `lengths[n]` rows of `t[n]` are copied into
    `values` starting at the cumulative offset `offsets[n] * D`.
    """
    n = tle.program_id(0)
    length_n = tl.load(lengths + n).to(tl.int64)
    offset_n = tl.load(offsets + n).to(tl.int64)
    count = length_n * D
    src_base = n * L * D
    dst_base = offset_n * D
    for i in range(0, L * D, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        m = idx < count
        v = tl.load(t + src_base + idx, mask=m)
        tl.store(values + dst_base + idx, v, mask=m)


def _nested_tensor_from_mask(t, mask, mask_check=True):
    logger.debug("GEMS_KUNLUNXIN _NESTED_TENSOR_FROM_MASK")
    assert t.dim() == 3, "Input should be a 3D tensor, N * L * D"
    assert mask.dim() == 2, "Padding mask should be 2D"
    assert mask.dtype == torch.bool, "Expected mask to be a bool tensor"
    N, L, D = t.shape
    assert mask.shape[0] == N and mask.shape[1] == L, "Mask shape should match input"

    t = t.contiguous()
    mask = mask.contiguous()

    # 1. Compute per-batch lengths and left-alignment flags via a Triton
    #    reduction kernel (no shifted load; see module docstring).
    lengths = torch.empty((N,), dtype=torch.int64, device=t.device)
    gaps = torch.empty((N,), dtype=torch.int8, device=t.device)
    block_len = max(1, min(triton.next_power_of_2(L), 1024))
    _nested_tensor_from_mask_lengths_kernel[(N,)](
        mask, lengths, gaps, L, BLOCK=block_len
    )

    # 2. Compute the exclusive prefix sum of the lengths to get offsets.
    offsets = torch.empty((N + 1,), dtype=torch.int64, device=t.device)
    has_gap = torch.empty((1,), dtype=torch.int8, device=t.device)
    if N <= 4096:
        block_scan = max(2, triton.next_power_of_2(N))
        _nested_tensor_from_mask_offsets_kernel[(1,)](
            lengths, gaps, offsets, has_gap, N, BLOCK=block_scan
        )
    else:
        offsets.copy_(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=t.device),
                    torch.cumsum(lengths, 0),
                ]
            )
        )
        has_gap.copy_(gaps.any().to(torch.int8))

    if mask_check:
        # A mask is left-aligned iff there is no False followed by a True
        # within the same row.
        if bool(has_gap.item()):
            raise RuntimeError("Mask must be left-aligned without gaps")

    total_valid = int(offsets[N].item())

    # 3. Compact the valid rows into a flat buffer via a Triton compaction
    #    kernel. Each batch contributes ``lengths[n] * D`` contiguous elements,
    #    so a single 1D buffer holds all packed components (matching the
    #    reference aten implementation's flattened ``values``).
    values = torch.empty((total_valid * D,), dtype=t.dtype, device=t.device)
    if total_valid > 0:
        block_compact = max(1, min(triton.next_power_of_2(L * D), 1024))
        _nested_tensor_from_mask_compact_kernel[(N,)](
            t, lengths, offsets, values, D, L, BLOCK=block_compact
        )

    # 4. Reconstruct a legacy (``torch.strided`` layout) NestedTensor directly
    #    from the flat values buffer, matching the reference aten implementation.
    #    Component ``n`` is the contiguous ``[lengths[n], D]`` slice whose storage
    #    offset is the cumulative ``lengths * D``. ``_nested_view_from_buffer``
    #    requires the size/strides/offsets metadata on CPU, so build it from the
    #    already-synced ``lengths``/``offsets`` on CPU.
    lengths_cpu = lengths.to("cpu")
    offsets_cpu = offsets.to("cpu")

    nested_size = torch.stack(
        [lengths_cpu, torch.full((N,), D, dtype=torch.int64)], dim=1
    )
    nested_strides = torch.empty_like(nested_size)
    nested_strides[:, 0] = D
    nested_strides[:, 1] = 1

    storage_offsets = offsets_cpu[:-1] * D

    return torch.ops.aten._nested_view_from_buffer.default(
        values, nested_size, nested_strides, storage_offsets
    )


__all__ = ["_nested_tensor_from_mask"]