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

"""Kunlunxin (TritonXPU) specialization of ``aten::unique_dim``.

Why this file exists
--------------------
The generic implementation (``src/flag_gems/ops/unique_dim.py``) relies on
several Triton constructs that are *not* honoured by the TritonXPU backend:

1. ``tl.load(..., mask=m, other=V)`` -- ``other=`` is silently ignored on this
   backend (both for affine and for discrete addressing).  The generic
   ``_unique_dim_counts_kernel`` depends on ``other=num_rows`` to synthesize the
   sentinel "one past the last group", so whenever the whole tile is masked off
   (``num_unique == 1``) every lane reads real out-of-range memory instead and
   ``counts`` comes out as garbage.  This is the single root cause of the
   93 accuracy failures observed on HEAD 9f10aff78 (all of them
   ``res_counts``, all of them exactly 1/1 element, all of them
   ``num_unique == 1``).
2. Masked *discrete* stores are not honoured either -- the inactive lanes really
   do write.  The generic inverse kernels do
   ``tl.store(inverse_ptr + sorted_indices, offsets, mask=mask)``, so the
   inactive lanes scribble over ``inverse[garbage]``.
3. Reduction / scan tiles narrower than 64 lanes mis-lower.

The strategy used throughout this file is the one the backend notes endorse:
**over-allocate every 1D helper buffer to a whole number of tiles and then use
no masks and no ``other=`` at all**.  Where a value genuinely has to differ for
the tail lanes we redirect with ``tl.where`` on the *value* (never rely on the
mask), and where a discrete store would need masking we redirect the inactive
lanes to a per-lane-unique scratch slot past the live region.

Additionally ``input.movedim(dim, 0).contiguous()`` is replaced by
``aten::_copy_from`` into a freshly allocated contiguous buffer: inside
``use_gems()`` ``Tensor.contiguous()`` is itself a FlagGems op and on this
backend a strided gems copy is both orders of magnitude slower than the native
strided engine and a documented card-wedging hazard.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# Tile used by every elementwise per-row kernel.  All per-row 1D helper buffers
# are padded up to a multiple of this so the kernels can run entirely unmasked.
_ROW_BLOCK = 1024
# Tile used when comparing rows column-wise.
_COMPARE_BLOCK = 1024
# Largest row count handled by the single-launch group-id scan kernel.
_GROUP_SCAN_MAX_ROWS = 4096
# Largest key count sorted by the single-launch rank-sort kernel.
_RANK_SORT_MAX_KEYS = 2048
# Narrowest legal reduction / scan tile on TritonXPU.
_MIN_TILE = 64

_INT_DTYPE_BITS = {
    torch.bool: 1,
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
}

_REMAP_INT = 0
_REMAP_FP16 = 1
_REMAP_FP32 = 2


def _num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 8
    if block_size >= 2048:
        return 4
    return 1


def _pad_len(n: int, block: int = _ROW_BLOCK) -> int:
    """Length of ``n`` rounded up to a whole number of ``block``-wide tiles."""
    return max(1, triton.cdiv(n, block)) * block


def _row_pad_len(num_rows: int) -> int:
    """Padded length used for every per-row helper buffer.

    Always a multiple of ``_ROW_BLOCK`` *and* at least as wide as the widest
    single-launch tile we may use over these buffers, so that both the
    ``_ROW_BLOCK``-tiled elementwise kernels and the single-tile scan / rank
    kernels can address the buffer without any mask.

    The ``next_power_of_2`` term matters: the single-launch group-id scan uses a
    ``next_pow2(num_rows)`` tile, which for ``num_rows`` in ``(2048, 3072]`` is
    wider than ``cdiv(num_rows, 1024) * 1024``.  Getting this wrong silently
    corrupts the scan (observed: 2049 and 3000 rows read past the key buffer).
    """
    pad = _pad_len(num_rows, _ROW_BLOCK)
    if num_rows <= _GROUP_SCAN_MAX_ROWS:
        pad = max(pad, triton.next_power_of_2(max(num_rows, 1)))
    return pad


def _empty_padded(num_rows: int, device, dtype=torch.int64):
    return torch.empty(_row_pad_len(num_rows), dtype=dtype, device=device)


def _to_padded(src: torch.Tensor, block: int = _ROW_BLOCK, extra: int = 0):
    n = src.numel()
    buf = torch.empty(_pad_len(n, block) + extra, dtype=src.dtype, device=src.device)
    if n:
        with torch_device_fn.device(src.device.index):
            torch.ops.aten._copy_from(src, buf[:n], False)
    return buf


def _to_row_padded(src: torch.Tensor, num_rows: int):
    """Like ``_to_padded`` but sized with ``_row_pad_len`` (scan-tile safe)."""
    buf = torch.empty(_row_pad_len(num_rows), dtype=src.dtype, device=src.device)
    if num_rows:
        with torch_device_fn.device(src.device.index):
            torch.ops.aten._copy_from(src, buf[:num_rows], False)
    return buf


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _rank_sort_kernel(
    keys_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    sorted_keys_ptr: tl.tensor,
    num_keys: int,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per key; counts how many keys sort before it.

    ``keys_ptr`` is padded to at least ``BLOCK_SIZE`` entries so the tile load
    needs no mask.  The pad entries are uninitialised, hence the comparison is
    gated arithmetically by ``candidates < num_keys`` -- never by a load mask.
    ``rank`` is a reduction result, so the store is a genuine 0-d store (the one
    store shape this backend does not widen to 64 elements).
    """
    row = ext.program_id(0)
    candidates = tl.arange(0, BLOCK_SIZE)
    live = candidates < num_keys

    cur = tl.load(keys_ptr + row)
    vals = tl.load(keys_ptr + candidates)
    before = ((vals < cur) | ((vals == cur) & (candidates < row))) & live
    rank = tl.sum(before.to(tl.int32), axis=0)
    tl.store(indices_ptr + rank, row)
    tl.store(sorted_keys_ptr + rank, cur)


@libentry()
@triton.jit
def _group_id_scan_kernel(
    keys_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    last_group_id_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Dense lexicographic group ids for one tile of ascending keys.

    Fully unmasked: ``keys_ptr`` / ``group_id_ptr`` are padded to at least
    ``BLOCK_SIZE``.  ``prev`` is fetched through a ``tl.where``-redirected offset
    instead of ``mask=offsets > 0, other=cur`` because ``other=`` is ignored on
    this backend.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    live = offsets < num_rows
    cur = tl.load(keys_ptr + offsets)
    prev = tl.load(keys_ptr + tl.where(offsets == 0, 0, offsets - 1))
    diff = ((cur - prev) != 0) & live & (offsets != 0)
    group_id = tl.cumsum(diff.to(tl.int64), axis=0)
    tl.store(group_id_ptr + offsets, group_id)
    last = tl.sum(tl.where(offsets == num_rows - 1, group_id, 0), axis=0)
    tl.store(last_group_id_ptr, last)


@libentry()
@triton.jit
def _adjacent_diff_kernel(
    keys_ptr: tl.tensor,
    diff_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    """``diff[i] = keys[i] != keys[i-1]`` (``diff[0] = 0``), int64, unmasked.

    Replaces the generic ``(keys[1:] - keys[:-1]) != 0`` + ``torch.cat``: those
    dispatch to gems ``sub``/``cat`` on int64 composite keys whose magnitude is
    ``num_rows << key_bits`` (up to ~4e13 for int32 input), i.e. far above the
    2**24 point where any accidental fp32 promotion in a shared vendor op starts
    losing bits.  Doing it here removes that whole dependency.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cur = tl.load(keys_ptr + offsets)
    prev = tl.load(keys_ptr + tl.where(offsets == 0, 0, offsets - 1))
    live = (offsets < num_rows) & (offsets != 0)
    diff = ((cur - prev) != 0) & live
    tl.store(diff_ptr + offsets, diff.to(tl.int64))


@libentry()
@triton.jit
def _gather_1d_kernel(
    values_ptr: tl.tensor,
    index_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_elements: int,
    EXACT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """``output[i] = values[index[i]]``, fully unmasked.

    ``index_ptr`` and ``output_ptr`` are padded to whole tiles.  When the padding
    is non-empty the index of a pad lane is forced to 0 with ``tl.where`` so the
    (discrete) value gather can never leave the ``values`` allocation -- a masked
    discrete gather would not be honoured here.  ``EXACT`` (padding empty) skips
    that ``tl.where`` because a runtime scalar inside an address expression is a
    documented 15-170x slowdown on this backend.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    raw = tl.load(index_ptr + offsets)
    if EXACT:
        idx = raw
    else:
        idx = tl.where(offsets < num_elements, raw, 0)
    values = tl.load(values_ptr + idx)
    tl.store(output_ptr + offsets, values)


@libentry()
@triton.jit
def _row_chunk_diff_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    row_chunk_diff_ptr: tl.tensor,
    row_len: int,
    num_chunks: int,
    EXACT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-(sorted row, column chunk) "differs from previous sorted row" flag.

    Two lowering hazards are avoided here, both established by the four-way
    single-variable probe ``probes/p05_rowdiff_variants.py`` (row_len 1025,
    BLOCK 1024, nc 2, one process per variant):

    * **A 1D ``tl.sum`` sitting in one arm of a runtime ``if/else`` does not
      lower** -- ``PassManager::run failed`` in ``make_llir``.  Variants v1
      (``tl.where(chunk == 0, 1, 0)`` in the other arm, i.e. verbatim what
      ``src/flag_gems/ops/unique_dim.py`` does), v2 (cast instead of
      ``tl.where``) and v4 (``tl.int1`` accumulator) all fail; only v3, which
      runs the reduction *unconditionally* and folds the ``row == 0`` case into
      arithmetic, compiles.  So the reduction is unconditional below.
    * The generic file's hash pre-filtered twin
      (``_unique_dim_row_chunk_diff_hash_kernel``) nests a second runtime ``if``
      around the same reduction and is dropped entirely: it costs a whole extra
      pass over ``flat`` to build the hashes, and it only ever runs on inputs
      that ``all_unique`` early-termination did *not* skip.
    """
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    prev_idx = tl.where(row == 0, 0, row - 1)
    cur_row = tl.load(sorted_indices_ptr + row)
    prev_row = tl.load(sorted_indices_ptr + prev_idx)
    if EXACT:
        # ``num_chunks * BLOCK_SIZE == row_len``: no tail, so keep the tile a
        # provably stride-1 block DMA (no runtime scalar in the address).
        cur = tl.load(flat_ptr + cur_row * row_len + offsets)
        prev = tl.load(flat_ptr + prev_row * row_len + offsets)
        neq = cur != prev
    else:
        live = offsets < row_len
        safe = tl.where(live, offsets, 0)
        cur = tl.load(flat_ptr + cur_row * row_len + safe)
        prev = tl.load(flat_ptr + prev_row * row_len + safe)
        neq = (cur != prev) & live
    diff = (tl.sum(neq.to(tl.int32), axis=0) != 0).to(tl.int32)
    # Row 0 always starts a group; put the flag on chunk 0 only so the
    # per-row OR-reduce below sees exactly one set bit.
    out = tl.where(row == 0, (chunk == 0).to(tl.int32), diff)
    tl.store(row_chunk_diff_ptr + row * num_chunks + chunk, out)


@libentry()
@triton.jit
def _row_diff_reduce_kernel(
    row_chunk_diff_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_chunks: int,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    live = chunks < num_chunks
    safe = tl.where(live, chunks, 0)
    raw = tl.load(row_chunk_diff_ptr + row * num_chunks + safe)
    vals = tl.where(live, raw, 0)
    tl.store(is_first_ptr + row, tl.sum(vals, axis=0) != 0)


@libentry()
@triton.jit
def _row_single_chunk_first_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    row_len: int,
    EXACT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    prev_idx = tl.where(row == 0, 0, row - 1)
    cur_row = tl.load(sorted_indices_ptr + row)
    prev_row = tl.load(sorted_indices_ptr + prev_idx)
    if EXACT:
        cur = tl.load(flat_ptr + cur_row * row_len + offsets)
        prev = tl.load(flat_ptr + prev_row * row_len + offsets)
        neq = cur != prev
    else:
        live = offsets < row_len
        safe = tl.where(live, offsets, 0)
        cur = tl.load(flat_ptr + cur_row * row_len + safe)
        prev = tl.load(flat_ptr + prev_row * row_len + safe)
        neq = (cur != prev) & live
    diff = tl.sum(neq.to(tl.int32), axis=0) != 0
    tl.store(is_first_ptr + row, (row == 0) | diff)


@libentry()
@triton.jit
def _gather_moved_kernel(
    flat_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    row_len: int,
    EXACT: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy ``flat[unique_indices[row], :]`` into ``output[row, :]``.

    ``output`` is over-allocated to ``num_unique * N_CHUNKS * BLOCK_SIZE`` and
    laid out with that padded row pitch, so the tail tile of every row spills
    into its *own* padding instead of into the next row -- no store mask.

    ``EXACT`` means ``N_CHUNKS * BLOCK_SIZE == row_len`` (the case for every
    power-of-two row length, hence for the whole official benchmark matrix).  It
    is worth a separate code path: clamping ``col`` with ``tl.where(col <
    row_len, ...)`` puts the runtime scalar ``row_len`` into a per-lane address
    and degrades the block DMA into a discrete gather.  Measured on this very
    kernel, ``(10000, 65536) int32 dim=0``: 379.9 ms with the clamp vs the
    generic implementation's 63.5 ms; ``(4096, 4096)`` 12.0 vs 3.7 ms.
    """
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    col = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    src_row = tl.load(unique_indices_ptr + row)
    if EXACT:
        values = tl.load(flat_ptr + src_row * row_len + col)
    else:
        values = tl.load(flat_ptr + src_row * row_len + tl.where(col < row_len, col, 0))
    tl.store(output_ptr + row * (N_CHUNKS * BLOCK_SIZE) + col, values)


@libentry()
@triton.jit
def _scatter_inverse_kernel(
    sorted_indices_ptr: tl.tensor,
    values_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_rows: int,
    IDENTITY: tl.constexpr,
    EXACT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """``inverse[sorted_indices[i]] = values[i]`` (or ``= i`` when ``IDENTITY``).

    A masked discrete store is not honoured on this backend, so instead of
    masking we give every dead lane a *unique, harmless* target past the live
    region (``num_rows + offsets``) exactly like the ``nonzero_numpy`` fix.
    ``inverse_ptr`` is therefore allocated with ``num_rows + n_tiles*BLOCK``
    slots and only the first ``num_rows`` are handed back.  Keeping the scratch
    targets unique also avoids the 8-144x address-aliasing penalty of a
    scatter that funnels many lanes into one slot.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tgt_raw = tl.load(sorted_indices_ptr + offsets)
    if EXACT:
        tgt = tgt_raw
    else:
        tgt = tl.where(offsets < num_rows, tgt_raw, num_rows + offsets)
    if IDENTITY:
        val = offsets.to(tl.int64)
    else:
        # ``values`` is the inclusive prefix sum of ``is_first``; the dense group
        # id is that minus one.  Folding the ``-1`` in here removes a gems
        # ``sub.Tensor`` launch from the chain.
        val = tl.load(values_ptr + offsets) - 1
    tl.store(inverse_ptr + tgt, val)


@libentry()
@triton.jit
def _counts_kernel(
    positions_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    num_rows: int,
    num_unique: int,
    BLOCK_SIZE: tl.constexpr,
):
    """``counts[g] = positions[g + 1] - positions[g]`` with a synthetic sentinel.

    This is the kernel that was actually broken on HEAD: the generic version
    synthesizes ``positions[num_unique]`` with
    ``tl.load(..., mask=(offsets + 1) < num_unique, other=num_rows)`` and
    ``other=`` is ignored on TritonXPU, so for ``num_unique == 1`` (every lane
    masked off) lane 0 read whatever followed the ``nonzero`` output in memory.
    Here the sentinel is applied to the loaded *value* with ``tl.where`` -- never
    through a load mask -- and ``positions_ptr`` is padded by one whole tile plus
    one element so both loads are plain in-range unmasked loads.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    positions = tl.load(positions_ptr + offsets)
    nxt_raw = tl.load(positions_ptr + offsets + 1)
    nxt = tl.where(offsets + 1 < num_unique, nxt_raw, num_rows)
    tl.store(counts_ptr + offsets, nxt - positions)


@libentry()
@triton.jit
def _build_key_kernel(
    flat_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    out_ptr: tl.tensor,
    num_rows: int,
    row_stride: int,
    col: int,
    KEY_OFFSET: tl.constexpr,
    KEY_SCALE: tl.constexpr,
    REMAP_KIND: tl.constexpr,
    FIRST: tl.constexpr,
    EXACT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One fused launch per cascade pass: ``group_id * KEY_SCALE + monotonic(v)``.

    Runs fully unmasked; ``indices_ptr`` / ``group_id_ptr`` / ``out_ptr`` are all
    padded to whole tiles, and the source row index of a pad lane is forced to 0
    so the (discrete, when ``not FIRST``) value load stays in range.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if FIRST:
        if EXACT:
            row = offsets.to(tl.int64)
        else:
            row = tl.where(offsets < num_rows, offsets, 0).to(tl.int64)
    else:
        raw = tl.load(indices_ptr + offsets)
        if EXACT:
            row = raw
        else:
            row = tl.where(offsets < num_rows, raw, 0)
    base = row * row_stride + col

    if REMAP_KIND == 0:  # _REMAP_INT
        x = tl.load(flat_ptr + base).to(tl.int64)
        val = x + KEY_OFFSET
    elif REMAP_KIND == 1:  # _REMAP_FP16
        bits = tl.load(flat_ptr + base).to(tl.int64) & 0xFFFF
        sign = (bits & 0x8000) != 0
        val = tl.where(sign, bits ^ 0xFFFF, bits ^ 0x8000)
    else:  # _REMAP_FP32
        bits = tl.load(flat_ptr + base).to(tl.int64) & 0xFFFFFFFF
        sign = (bits & 0x80000000) != 0
        val = tl.where(sign, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)

    if FIRST:
        out = val
    else:
        gid = tl.load(group_id_ptr + offsets)
        out = gid * KEY_SCALE + val
    tl.store(out_ptr + offsets, out)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _remap_info(flat: torch.Tensor):
    dt = flat.dtype
    if dt == torch.bool:
        return flat.view(torch.uint8), _REMAP_INT, 0
    if dt == torch.uint8:
        return flat, _REMAP_INT, 0
    if dt == torch.int8:
        return flat, _REMAP_INT, 1 << 7
    if dt == torch.int16:
        return flat, _REMAP_INT, 1 << 15
    if dt == torch.int32:
        return flat, _REMAP_INT, 1 << 31
    if dt in (torch.float16, torch.bfloat16):
        return flat.view(torch.int16), _REMAP_FP16, 0
    if dt == torch.float32:
        return flat.view(torch.int32), _REMAP_FP32, 0
    raise NotImplementedError(dt)


def _build_composite_key(
    flat_view: torch.Tensor,
    col: int,
    indices: torch.Tensor | None,
    group_id: torch.Tensor | None,
    num_rows: int,
    row_stride: int,
    key_offset: int,
    key_scale: int,
    remap_kind: int,
) -> torch.Tensor:
    out = _empty_padded(num_rows, flat_view.device)
    first = indices is None
    indices_arg = out if first else indices
    group_id_arg = out if first else group_id
    pad = out.numel()
    grid = (pad // _ROW_BLOCK, 1, 1)
    with torch_device_fn.device(flat_view.device.index):
        _build_key_kernel[grid](
            flat_view,
            indices_arg,
            group_id_arg,
            out,
            num_rows,
            row_stride,
            col,
            KEY_OFFSET=key_offset,
            KEY_SCALE=key_scale,
            REMAP_KIND=remap_kind,
            FIRST=first,
            EXACT=(pad == num_rows),
            BLOCK_SIZE=_ROW_BLOCK,
            num_warps=4,
        )
    return out


def _gather_1d(values: torch.Tensor, index_pad: torch.Tensor, n: int):
    """``values[index_pad[:n]]`` returned in a buffer padded like ``index_pad``."""
    output = torch.empty_like(index_pad, dtype=values.dtype)
    if n == 0:
        return output
    pad = index_pad.numel()
    grid = (triton.cdiv(pad, _ROW_BLOCK), 1, 1)
    with torch_device_fn.device(values.device.index):
        _gather_1d_kernel[grid](
            values,
            index_pad,
            output,
            n,
            EXACT=(pad == n),
            BLOCK_SIZE=_ROW_BLOCK,
            num_warps=4,
        )
    return output


def _argsort_keys(keys_pad: torch.Tensor, num_keys: int):
    """Ascending argsort of ``keys_pad[:num_keys]``.

    Returns ``(perm_pad, sorted_keys_pad)``, both padded like ``keys_pad``.
    """
    device = keys_pad.device
    if num_keys == 0:
        return keys_pad, keys_pad
    if num_keys <= _RANK_SORT_MAX_KEYS:
        perm = torch.empty_like(keys_pad)
        sorted_keys = torch.empty_like(keys_pad)
        block = max(_MIN_TILE, triton.next_power_of_2(num_keys))
        with torch_device_fn.device(device.index):
            _rank_sort_kernel[(num_keys, 1, 1)](
                keys_pad,
                perm,
                sorted_keys,
                num_keys,
                BLOCK_SIZE=block,
                num_warps=_num_warps(block),
            )
        return perm, sorted_keys
    sorted_keys, perm = torch.sort(keys_pad[:num_keys])
    return _to_row_padded(perm, num_keys), _to_row_padded(sorted_keys, num_keys)


def _group_id_from_sorted(sorted_keys_pad: torch.Tensor, num_rows: int):
    """Dense lexicographic group ids for ascending keys.

    Returns ``(group_id_pad, last_group_id)``.
    """
    device = sorted_keys_pad.device
    if num_rows == 0:
        return sorted_keys_pad, -1
    if num_rows <= _GROUP_SCAN_MAX_ROWS:
        group_id = torch.empty_like(sorted_keys_pad)
        last = torch.empty((), dtype=torch.int64, device=device)
        block = max(_MIN_TILE, triton.next_power_of_2(num_rows))
        with torch_device_fn.device(device.index):
            _group_id_scan_kernel[(1, 1, 1)](
                sorted_keys_pad,
                group_id,
                last,
                num_rows,
                BLOCK_SIZE=block,
                num_warps=_num_warps(block),
            )
        return group_id, int(last.item())

    diff = torch.empty_like(sorted_keys_pad)
    grid = (triton.cdiv(diff.numel(), _ROW_BLOCK), 1, 1)
    with torch_device_fn.device(device.index):
        _adjacent_diff_kernel[grid](
            sorted_keys_pad,
            diff,
            num_rows,
            BLOCK_SIZE=_ROW_BLOCK,
            num_warps=4,
        )
    group_id = torch.cumsum(diff, dim=0)
    return group_id, int(group_id[num_rows - 1].item())


def _lex_argsort_rows(flat: torch.Tensor, num_rows: int, num_cols: int):
    """Lex-sort the rows of a 2D tensor.  Returns ``(indices_pad, all_unique)``."""
    device = flat.device
    key_bits = _INT_DTYPE_BITS.get(flat.dtype)
    if key_bits is None:
        return _lex_argsort_rows_cascade(flat, num_rows, num_cols), False
    if num_cols == 0 or num_rows <= 1:
        idx = _to_row_padded(
            torch.arange(num_rows, dtype=torch.int64, device=device), num_rows
        )
        return idx, num_rows <= 1 and num_cols != 0

    key_scale = 1 << key_bits
    flat_view, remap_kind, key_offset = _remap_info(flat)
    indices = None
    group_id = None
    all_unique = False
    for col in range(num_cols):
        keys = _build_composite_key(
            flat_view,
            col,
            indices,
            group_id,
            num_rows,
            num_cols,
            key_offset,
            key_scale,
            remap_kind,
        )
        perm, sorted_keys = _argsort_keys(keys, num_rows)
        indices = perm if col == 0 else _gather_1d(indices, perm, num_rows)
        group_id, last_group_id = _group_id_from_sorted(sorted_keys, num_rows)
        if last_group_id == num_rows - 1:
            all_unique = True
            break
    return indices, all_unique


def _lex_argsort_rows_cascade(flat: torch.Tensor, num_rows: int, num_cols: int):
    """Generic-dtype fallback (e.g. int64): LSD cascade of stable argsorts."""
    device = flat.device
    indices = _to_row_padded(
        torch.arange(num_rows, dtype=torch.int64, device=device), num_rows
    )
    if num_rows <= 1 or num_cols == 0:
        return indices
    col_major = torch.empty((num_cols, num_rows), dtype=flat.dtype, device=device)
    with torch_device_fn.device(device.index):
        torch.ops.aten._copy_from(flat.t(), col_major, False)
    for col in range(num_cols - 1, -1, -1):
        keys = _gather_1d(_to_row_padded(col_major[col], num_rows), indices, num_rows)
        _, perm = torch.sort(keys[:num_rows], stable=True)
        indices = _gather_1d(indices, _to_row_padded(perm, num_rows), num_rows)
    return indices


def _first_mask(
    flat: torch.Tensor,
    sorted_indices_pad: torch.Tensor,
    num_rows: int,
    row_len: int,
):
    """Padded bool mask marking the first row of every sorted lex group."""
    device = flat.device
    is_first = torch.zeros(_row_pad_len(num_rows), dtype=torch.bool, device=device)
    if num_rows == 1 or row_len == 0:
        with torch_device_fn.device(device.index):
            torch.ops.aten._copy_from(
                torch.ones(1, dtype=torch.bool, device=device), is_first[:1], False
            )
        return is_first

    block = min(_COMPARE_BLOCK, max(_MIN_TILE, triton.next_power_of_2(row_len)))
    nc = triton.cdiv(row_len, block)
    if nc == 1:
        with torch_device_fn.device(device.index):
            _row_single_chunk_first_kernel[(num_rows, 1, 1)](
                flat,
                sorted_indices_pad,
                is_first,
                row_len,
                EXACT=(block == row_len),
                BLOCK_SIZE=block,
                num_warps=_num_warps(block),
            )
        return is_first

    row_chunk_diff = torch.empty((num_rows, nc), dtype=torch.int32, device=device)
    bc = max(_MIN_TILE, triton.next_power_of_2(nc))
    with torch_device_fn.device(device.index):
        _row_chunk_diff_kernel[(num_rows, nc, 1)](
            flat,
            sorted_indices_pad,
            row_chunk_diff,
            row_len,
            nc,
            EXACT=(nc * block == row_len),
            BLOCK_SIZE=block,
            num_warps=_num_warps(block),
        )
        _row_diff_reduce_kernel[(num_rows, 1, 1)](
            row_chunk_diff,
            is_first,
            nc,
            BLOCK_CHUNKS=bc,
            num_warps=_num_warps(bc),
        )
    return is_first


def _gather_output(
    moved: torch.Tensor,
    unique_indices_pad: torch.Tensor,
    num_unique: int,
    dim: int,
    input_shape: torch.Size,
):
    device = moved.device
    output_shape = (
        tuple(input_shape[:dim]) + (num_unique,) + tuple(input_shape[dim + 1 :])
    )
    if num_unique == 0:
        return torch.empty(output_shape, dtype=moved.dtype, device=device)

    row_len = moved[0].numel()
    flat = moved.reshape(moved.shape[0], row_len)
    out_shape = (num_unique,) + tuple(moved.shape[1:])
    if row_len == 0:
        return torch.empty(out_shape, dtype=moved.dtype, device=device).movedim(0, dim)
    if row_len == 1:
        # Degenerate row: this is exactly a 1D gather.  Going through the 2D
        # (row, chunk) grid would launch one 64-lane program per output element;
        # ``_gather_1d`` covers 1024 elements per program instead.
        out = _gather_1d(moved.reshape(-1), unique_indices_pad, num_unique)
        return out[:num_unique].reshape(out_shape).movedim(0, dim)

    block = min(_COMPARE_BLOCK, max(_MIN_TILE, triton.next_power_of_2(row_len)))
    nc = triton.cdiv(row_len, block)
    pitch = nc * block
    # ``pitch == row_len`` lets the kernel write straight into the result with no
    # store mask at all; otherwise we give every row its own padding so the tail
    # tile can never spill into the next row, then extract the prefix view with
    # the native strided engine.
    if pitch == row_len:
        moved_output = torch.empty(out_shape, dtype=moved.dtype, device=device)
        dest = moved_output
    else:
        padded = torch.empty(num_unique * pitch, dtype=moved.dtype, device=device)
        dest = padded
    with torch_device_fn.device(device.index):
        _gather_moved_kernel[(num_unique, nc, 1)](
            flat,
            unique_indices_pad,
            dest,
            row_len,
            EXACT=(pitch == row_len),
            N_CHUNKS=nc,
            BLOCK_SIZE=block,
            num_warps=4,
        )
    if pitch != row_len:
        moved_output = torch.empty(out_shape, dtype=moved.dtype, device=device)
        with torch_device_fn.device(device.index):
            torch.ops.aten._copy_from(
                padded.view(num_unique, pitch)[:, :row_len],
                moved_output.reshape(num_unique, row_len),
                False,
            )
    return moved_output.movedim(0, dim)


def _inverse(
    sorted_indices_pad: torch.Tensor,
    is_first_pad: torch.Tensor | None,
    num_rows: int,
):
    """``inverse`` in original index space.

    ``is_first_pad is None`` means "every row is unique", i.e. the value written
    is the sorted position itself.
    """
    device = sorted_indices_pad.device
    pad = _row_pad_len(num_rows)
    if num_rows == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    # ``num_rows`` live slots plus, when the grid over-runs ``num_rows``, one
    # unique scratch slot per dead lane (a masked scatter is not honoured here).
    exact = pad == num_rows
    buf = torch.empty(
        num_rows if exact else num_rows + pad, dtype=torch.int64, device=device
    )
    if is_first_pad is None:
        values = sorted_indices_pad
        identity = True
    else:
        values = torch.cumsum(is_first_pad.to(torch.int64), dim=0)
        identity = False
    with torch_device_fn.device(device.index):
        _scatter_inverse_kernel[(pad // _ROW_BLOCK, 1, 1)](
            sorted_indices_pad,
            values,
            buf,
            num_rows,
            IDENTITY=identity,
            EXACT=exact,
            BLOCK_SIZE=_ROW_BLOCK,
            num_warps=4,
        )
    return buf[:num_rows]


def _counts_from_positions(
    first_positions: torch.Tensor,
    num_rows: int,
):
    num_unique = first_positions.numel()
    device = first_positions.device
    if num_unique == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    pos = _to_padded(first_positions, extra=1)
    counts = torch.empty(_pad_len(num_unique), dtype=torch.int64, device=device)
    with torch_device_fn.device(device.index):
        _counts_kernel[(counts.numel() // _ROW_BLOCK, 1, 1)](
            pos,
            counts,
            num_rows,
            num_unique,
            BLOCK_SIZE=_ROW_BLOCK,
            num_warps=4,
        )
    return counts[:num_unique]


def _moved_contiguous(input: torch.Tensor, dim: int) -> torch.Tensor:
    """``input.movedim(dim, 0)`` made contiguous *without* gems ``contiguous()``.

    Inside ``use_gems()`` ``Tensor.contiguous()`` is a FlagGems op, and a strided
    gems copy on this backend is both far slower than the native strided engine
    (documented 234x on a transpose) and a known card-wedge hazard on a strided
    *source*.  ``aten::_copy_from`` is the repo's native out-writeback primitive.
    """
    view = input.movedim(dim, 0)
    if view.is_contiguous():
        return view
    dest = torch.empty(list(view.shape), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device.index):
        torch.ops.aten._copy_from(view, dest, False)
    return dest


def unique_dim(
    input: torch.Tensor,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN UNIQUE_DIM")

    ndim = input.ndim if input.ndim > 0 else 1
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= max(input.ndim, 1):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    device = input.device
    size_dim = input.size(dim) if input.ndim > 0 else input.numel()

    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    if size_dim == 0:
        return (
            input.clone(),
            empty_i64,
            torch.empty(0, dtype=torch.int64, device=device),
        )

    moved = _moved_contiguous(input, dim)
    row_len = moved.numel() // size_dim
    flat = moved.reshape(size_dim, row_len)

    sorted_indices, all_unique = _lex_argsort_rows(flat, size_dim, row_len)

    inverse_indices = empty_i64
    counts = torch.empty(0, dtype=torch.int64, device=device)

    if all_unique:
        if return_counts:
            counts = torch.ones(size_dim, dtype=torch.int64, device=device)
        if return_inverse:
            inverse_indices = _inverse(sorted_indices, None, size_dim)
        output = _gather_output(moved, sorted_indices, size_dim, dim, input.shape)
        return output, inverse_indices, counts

    is_first = _first_mask(flat, sorted_indices, size_dim, row_len)
    first_positions = torch.nonzero(is_first[:size_dim], as_tuple=False).flatten()
    num_unique = first_positions.numel()
    unique_in_orig = _gather_1d(sorted_indices, _to_padded(first_positions), num_unique)

    if return_inverse:
        inverse_indices = _inverse(sorted_indices, is_first, size_dim)
    if return_counts:
        counts = _counts_from_positions(first_positions, size_dim)

    output = _gather_output(moved, unique_in_orig, num_unique, dim, input.shape)
    return output, inverse_indices, counts
