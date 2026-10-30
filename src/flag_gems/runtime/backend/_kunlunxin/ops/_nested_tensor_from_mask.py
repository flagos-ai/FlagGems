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

from flag_gems.ops._nested_tensor_from_mask import (
    _nested_tensor_from_mask_compact_kernel,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Rows per program of the two-level scan; also the threshold below which the
# single-block scan is used.
_SCAN_CHUNK = 1024
# Upper bound on the number of chunks the single-block chunk scan can hold.
_MAX_CHUNKS = 4096


@libentry()
@triton.jit
def _nested_mask_lengths_kernel(
    mask,
    lengths,
    gaps,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per mask row: number of True entries + "not left-aligned" flag.

    One program per row, a single affine masked load per iteration.
    """
    n = tle.program_id(0)
    row = mask + n * L
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    pos = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        m = offs < L
        vals = tl.load(row + offs, mask=m, other=0).to(tl.int32)
        vals = tl.where(m, vals, 0)
        acc += vals
        pos = tl.maximum(pos, vals * (offs.to(tl.int32) + 1))
    count = tl.sum(acc, axis=0)
    last = tl.max(pos, axis=0)
    tl.store(lengths + n, count.to(tl.int64))
    tl.store(gaps + n, (last != count).to(tl.int8))


@libentry()
@triton.jit
def _nested_mask_scan_kernel(
    lengths,
    gaps,
    offsets,
    storage_offsets,
    has_gap,
    N,
    D,
    BLOCK: tl.constexpr,
):
    """Single-block exclusive scan of ``lengths`` (requires ``N <= BLOCK``).

    ``offsets[0:N]`` gets the exclusive scan and ``offsets[1:N + 1]`` the
    inclusive one, so ``offsets[N]`` ends up holding the total without a
    separate reduction. ``storage_offsets`` is the exclusive scan scaled by
    ``D`` (the per-component storage offset of the nested view) so that no
    host-side tensor arithmetic is needed. ``has_gap`` is the any-reduction of
    the per-row gap flags.
    """
    idx = tl.arange(0, BLOCK)
    m = idx < N
    vals = tl.load(lengths + idx, mask=m, other=0).to(tl.int64)
    vals = tl.where(m, vals, 0)
    gap_vals = tl.load(gaps + idx, mask=m, other=0).to(tl.int32)
    gap_vals = tl.where(m, gap_vals, 0)
    inclusive = tl.cumsum(vals, axis=0)
    exclusive = inclusive - vals
    tl.store(offsets + idx, exclusive, mask=m)
    tl.store(offsets + 1 + idx, inclusive, mask=m)
    tl.store(storage_offsets + idx, exclusive * D, mask=m)
    tl.store(has_gap, (tl.max(gap_vals, axis=0) > 0).to(tl.int8))


@libentry()
@triton.jit
def _nested_mask_chunk_sums_kernel(
    lengths,
    gaps,
    chunk_sums,
    chunk_gaps,
    N,
    CHUNK: tl.constexpr,
):
    """Per-chunk sum of ``lengths`` and per-chunk any of ``gaps``.

    The tail chunk is handled with a clamped (unmasked) load plus a register
    ``tl.where``; a masked tail reduction is miscompiled on large grids.
    """
    c = tle.program_id(0)
    idx = c * CHUNK + tl.arange(0, CHUNK)
    safe = tl.minimum(idx, N - 1)
    keep = idx < N
    vals = tl.load(lengths + safe).to(tl.int64)
    vals = tl.where(keep, vals, 0)
    gap_vals = tl.load(gaps + safe).to(tl.int32)
    gap_vals = tl.where(keep, gap_vals, 0)
    tl.store(chunk_sums + c, tl.sum(vals, axis=0))
    tl.store(chunk_gaps + c, tl.max(gap_vals, axis=0))


@libentry()
@triton.jit
def _nested_mask_chunk_scan_kernel(
    chunk_sums,
    chunk_gaps,
    chunk_offsets,
    has_gap,
    NC,
    BLOCK: tl.constexpr,
):
    """Single-block exclusive scan of the chunk sums + global gap any."""
    idx = tl.arange(0, BLOCK)
    m = idx < NC
    vals = tl.load(chunk_sums + idx, mask=m, other=0).to(tl.int64)
    vals = tl.where(m, vals, 0)
    gap_vals = tl.load(chunk_gaps + idx, mask=m, other=0).to(tl.int32)
    gap_vals = tl.where(m, gap_vals, 0)
    inclusive = tl.cumsum(vals, axis=0)
    tl.store(chunk_offsets + idx, inclusive - vals, mask=m)
    tl.store(has_gap, (tl.max(gap_vals, axis=0) > 0).to(tl.int8))


@libentry()
@triton.jit
def _nested_mask_scan_apply_kernel(
    lengths,
    chunk_offsets,
    offsets,
    storage_offsets,
    N,
    D,
    CHUNK: tl.constexpr,
):
    """Per-chunk scan of ``lengths`` offset by the scanned chunk base."""
    c = tle.program_id(0)
    base = tl.load(chunk_offsets + c).to(tl.int64)
    idx = c * CHUNK + tl.arange(0, CHUNK)
    safe = tl.minimum(idx, N - 1)
    keep = idx < N
    vals = tl.load(lengths + safe).to(tl.int64)
    vals = tl.where(keep, vals, 0)
    inclusive = base + tl.cumsum(vals, axis=0)
    exclusive = inclusive - vals
    tl.store(offsets + idx, exclusive, mask=keep)
    tl.store(offsets + 1 + idx, inclusive, mask=keep)
    tl.store(storage_offsets + idx, exclusive * D, mask=keep)


def _nested_tensor_from_mask(t, mask, mask_check=True):
    logger.debug("GEMS_KUNLUNXIN _NESTED_TENSOR_FROM_MASK")
    logging.getLogger("flag_gems.ops._nested_tensor_from_mask").debug(
        "GEMS _NESTED_TENSOR_FROM_MASK"
    )
    assert t.dim() == 3, "Input should be a 3D tensor, N * L * D"
    assert mask.dim() == 2, "Padding mask should be 2D"
    assert mask.dtype == torch.bool, "Expected mask to be a bool tensor"
    N, L, D = t.shape
    assert mask.shape[0] == N and mask.shape[1] == L, "Mask shape should match input"

    t = t.contiguous()
    mask = mask.contiguous()

    # 1. Per-row valid length + left-alignment flag.
    lengths = torch.empty((N,), dtype=torch.int64, device=t.device)
    gaps = torch.empty((N,), dtype=torch.int8, device=t.device)
    block_len = max(1, min(triton.next_power_of_2(L), 1024))
    _nested_mask_lengths_kernel[(N,)](mask, lengths, gaps, L, BLOCK=block_len)

    # 2. Exclusive prefix sum of the lengths (``offsets[N]`` is the total) plus
    #    the any-reduction of the per-row gap flags.
    offsets = torch.empty((N + 1,), dtype=torch.int64, device=t.device)
    storage_offsets_dev = torch.empty((N,), dtype=torch.int64, device=t.device)
    has_gap = torch.empty((1,), dtype=torch.int8, device=t.device)
    if N <= _SCAN_CHUNK:
        _nested_mask_scan_kernel[(1,)](
            lengths,
            gaps,
            offsets,
            storage_offsets_dev,
            has_gap,
            N,
            D,
            BLOCK=max(2, triton.next_power_of_2(N)),
        )
    else:
        n_chunks = triton.cdiv(N, _SCAN_CHUNK)
        assert n_chunks <= _MAX_CHUNKS, f"batch size {N} exceeds the scan capacity"
        chunk_sums = torch.empty((n_chunks,), dtype=torch.int64, device=t.device)
        chunk_gaps = torch.empty((n_chunks,), dtype=torch.int32, device=t.device)
        chunk_offsets = torch.empty((n_chunks,), dtype=torch.int64, device=t.device)
        _nested_mask_chunk_sums_kernel[(n_chunks,)](
            lengths, gaps, chunk_sums, chunk_gaps, N, CHUNK=_SCAN_CHUNK
        )
        _nested_mask_chunk_scan_kernel[(1,)](
            chunk_sums,
            chunk_gaps,
            chunk_offsets,
            has_gap,
            n_chunks,
            BLOCK=max(2, triton.next_power_of_2(n_chunks)),
        )
        _nested_mask_scan_apply_kernel[(n_chunks,)](
            lengths,
            chunk_offsets,
            offsets,
            storage_offsets_dev,
            N,
            D,
            CHUNK=_SCAN_CHUNK,
        )

    if mask_check and bool(has_gap.item()):
        raise RuntimeError("Mask must be left-aligned without gaps")

    total_valid = int(offsets[N].item())

    # 3. Compact the valid rows into one flat buffer (generic kernel: a plain
    #    affine masked 1D copy, probed correct on this backend).
    values = torch.empty((total_valid * D,), dtype=t.dtype, device=t.device)
    if total_valid > 0:
        block_compact = max(1, min(triton.next_power_of_2(L * D), 1024))
        _nested_tensor_from_mask_compact_kernel[(N,)](
            t, lengths, offsets, values, D, L, BLOCK=block_compact
        )

    # 4. Rebuild a legacy (``torch.strided``) NestedTensor view over the flat
    #    buffer. ``_nested_view_from_buffer`` is a pure metadata view
    #    constructor (no compute) and requires its size / stride / storage
    #    offset metadata on CPU.
    nested_size = torch.empty((N, 2), dtype=torch.int64)
    nested_size[:, 0] = lengths.to("cpu")
    nested_size[:, 1] = D
    nested_strides = torch.empty((N, 2), dtype=torch.int64)
    nested_strides[:, 0] = D
    nested_strides[:, 1] = 1
    storage_offsets = storage_offsets_dev.to("cpu")

    return torch.ops.aten._nested_view_from_buffer.default(
        values, nested_size, nested_strides, storage_offsets
    )
