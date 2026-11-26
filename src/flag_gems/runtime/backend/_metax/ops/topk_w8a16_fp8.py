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

"""MetaX W8A16 TopK.

Ported from the THead / PPU implementation. Input is grouped FP8 E5M2 plus
per-group scale (group_size=128). The kernel dequantizes on the fly and
returns BF16 values and int64 indices along the last dimension.

Portable Triton kernels cover the small and generic cases. Parallel radix
selection and sampled candidate filtering handle the wide rows common in
indexer workloads. Candidate filtering falls back to exact radix selection
on any row whose candidates are insufficient or exceed scratch capacity.
"""

import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _MAX_INT32_VAL, _MIN_INT32_VAL, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

# CHUNKED_MERGE_MIN_K / RADIX_MIN_N / RADIX_MAX_ROWS are runtime routing knobs:
# the benchmark harness overrides them via environment variables for A/B
# measurement, defaults come from C550 complete-call-latency measurements.
# Radix wins on large N (>= 16384) with the histogram's fixed work intact,
# but the per-row 256KB histogram zero-init scales linearly with batch and
# overturns the win once the row count gets large (~1M elements total).
CHUNKED_MERGE_MIN_K = int(os.environ.get("FLAGGEMS_TOPK_FP8_CHUNKED_MIN_K", "16"))
RADIX_THRESHOLD_N = int(os.environ.get("FLAGGEMS_TOPK_FP8_RADIX_MIN_N", "16384"))
RADIX_MAX_TOTAL = int(os.environ.get("FLAGGEMS_TOPK_FP8_RADIX_MAX_TOTAL", "1048576"))
CHUNK_MAX_K = 2048
# Chosen so per-program candidate blocks stay within the probed tl.sort
# width limit (2048) on this backend's triton.
CHUNK_SIZE = 1024

logger = logging.getLogger(__name__)


@triton.jit
def _fp8_e5_to_f32(x):
    return x.to(tl.float8e5, bitcast=True).to(tl.float32)


@triton.jit
def _fp8_bits_to_ordered_key(bits):
    sign = bits & 0x80
    flip_mask = tl.where(sign != 0, 0xFF, 0x80).to(tl.uint8)
    return bits ^ flip_mask


@libentry()
@triton.jit
def topk_fp8_single_stage_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    # Single-kernel small-N path, packed-key edition. The whole row is keyed
    # once (bf16-rounded value key << 16 | column payload), sorted as one
    # unsigned tensor, and only the k selected elements are dequantized from
    # the original FP8 payload. One u32 tl.sort replaces the dual-tensor f32
    # argsort (which sorted values and indices together) and the full-row
    # dequantize-to-output; the shared u16-key domain and pad conventions
    # match the chunked path.
    row = tle.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    valid = cols < N

    bits = tl.load(x_ptr + row * N + cols, mask=valid, other=0).to(tl.uint8)
    x_q = _fp8_e5_to_f32(bits)
    x_scale = tl.load(
        scale_ptr + row * NUM_GROUPS + cols // GROUP_SIZE, mask=valid, other=0.0
    ).to(tl.float32)
    value = (x_q * x_scale).to(tl.bfloat16)

    v_bits = value.to(tl.uint16, bitcast=True)
    v_flip = tl.where((v_bits & 0x8000) != 0, 0xFFFF, 0x8000).to(tl.uint16)
    key = tl.where(value != value, 0xFFFF, v_bits ^ v_flip).to(tl.uint16)
    # Real elements never produce sk == 0: the payload (0xFFFE - col) is
    # nonzero for every in-range column, so 0 stays a pure pad sentinel.
    sk = (key.to(tl.uint32) << 16) | (0xFFFE - cols).to(tl.uint32)

    keep = cols < k
    if DESCENDING:
        top = tl.sort(tl.where(valid, sk, 0), dim=0, descending=True)
    else:
        # ~sk flips the domain so one descending sort yields the smallest
        # values first; pads (0) are never complemented (see the chunk merge
        # kernel) and real keys never encode to 0.
        top = tl.sort(tl.where(valid, ~sk, 0), dim=0, descending=True)
        top = tl.where(keep & (top != 0), ~top, 0)

    index = (0xFFFE - (top & 0xFFFF)).to(tl.int64)
    sel = tl.where(keep & (top != 0), index, 0)
    bits_k = tl.load(x_ptr + row * N + sel, mask=keep, other=0).to(tl.uint8)
    x_q_k = _fp8_e5_to_f32(bits_k)
    scale_k = tl.load(
        scale_ptr + row * NUM_GROUPS + (sel // GROUP_SIZE), mask=keep, other=0.0
    ).to(tl.float32)
    value_k = (x_q_k * scale_k).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * k + cols, value_k, mask=keep)
    tl.store(index_ptr + row * k + cols, sel, mask=keep)


@triton.jit
def _merge_sorted_topk(
    best_val,
    best_idx,
    tile_val,
    tile_idx,
    BLOCK: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    sval, sidx = argsort(tile_val, tile_idx, 0, DESCENDING)
    row0 = tl.arange(0, 2)[:, None] == 0
    mval = tl.where(row0, best_val.reshape(1, BLOCK), sval.reshape(1, BLOCK)).reshape(
        2 * BLOCK
    )
    midx = tl.where(row0, best_idx.reshape(1, BLOCK), sidx.reshape(1, BLOCK)).reshape(
        2 * BLOCK
    )
    mval, midx = argsort(mval, midx, 0, DESCENDING)
    mval2 = mval.reshape(2, BLOCK)
    midx2 = midx.reshape(2, BLOCK)
    out_val = tl.sum(tl.where(row0, mval2, 0.0), axis=0)
    out_idx = tl.sum(tl.where(row0, midx2.to(tl.int64), 0), axis=0).to(tl.int32)
    return out_val, out_idx


@libentry()
@triton.jit
def topk_fp8_running_merge_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    N,
    k: tl.constexpr,
    BLOCK: tl.constexpr,
    DESCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    x_ptr += pid * N
    scale_ptr += pid * tl.cdiv(N, GROUP_SIZE)
    y_ptr += pid * k
    index_ptr += pid * k

    pad_val = float("-inf") if DESCENDING else float("inf")
    pad_idx = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL
    offs = tl.arange(0, BLOCK)
    best_val = tl.full([BLOCK], pad_val, dtype=tl.float32)
    best_idx = tl.full([BLOCK], pad_idx, dtype=tl.int32)

    n_tiles = tl.cdiv(N, BLOCK)
    for t in tl.range(0, n_tiles):
        cols = t * BLOCK + offs
        mask = cols < N
        x_q = _fp8_e5_to_f32(tl.load(x_ptr + cols, mask=mask, other=0))
        x_scale = tl.load(scale_ptr + cols // GROUP_SIZE, mask=mask, other=0.0).to(
            tl.float32
        )
        tile_val = tl.where(mask, x_q * x_scale, pad_val)
        tile_idx = tl.where(mask, cols, pad_idx).to(tl.int32)
        best_val, best_idx = _merge_sorted_topk(
            best_val, best_idx, tile_val, tile_idx, BLOCK, DESCENDING
        )

    out_mask = offs < k
    tl.store(y_ptr + offs, best_val, mask=out_mask)
    tl.store(index_ptr + offs, best_idx.to(tl.int64), mask=out_mask)


# ---------------------------------------------------------------------------
# Chunked path: parallel per-chunk local top-k, then hierarchical merge.
#
# Candidates are u32 sort keys: (u16 monotone key of the BF16-rounded A16
# value) << 16 | (0xFFFE - column). A single descending tl.sort serves both
# orders: descending sorts the key itself, ascending sorts its complement.
# Invalid slots are forced to 0, which is strictly below every valid key, so
# they always sort last; slots that survive a truncation to k but hold no
# real element are re-zeroed so pads never leak downstream. The 0xFFFE
# offset keeps real payloads clear of the pad value. Candidate buffers are
# int32 tensors; values cross the boundary bitcast to uint32.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def topk_fp8_chunk_stage1_kernel(
    cand_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    N: tl.constexpr,
    CHUNK: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    ASCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tle.program_id(0)
    chunk = tle.program_id(1)
    cols = chunk * CHUNK + tl.arange(0, CHUNK)
    valid = cols < N

    bits = tl.load(x_ptr + row * N + cols, mask=valid, other=0).to(tl.uint8)
    x_q = _fp8_e5_to_f32(bits)
    x_scale = tl.load(
        scale_ptr + row * tl.cdiv(N, GROUP_SIZE) + cols // GROUP_SIZE,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    value = (x_q * x_scale).to(tl.bfloat16)

    v_bits = value.to(tl.uint16, bitcast=True)
    v_flip = tl.where((v_bits & 0x8000) != 0, 0xFFFF, 0x8000).to(tl.uint16)
    key = tl.where(value != value, 0xFFFF, v_bits ^ v_flip).to(tl.uint16)
    # No real element can produce sk == 0: the low payload (0xFFFE - col) is
    # nonzero for every in-range column, so 0 is a safe pad sentinel.

    # sk is never 0 for a real element: cols < N <= 0xFFFE (routing guard),
    # so the payload (0xFFFE - cols) is >= 1 and 0 stays a pure pad sentinel.
    sk = (key.to(tl.uint32) << 16) | (0xFFFE - cols).to(tl.uint32)
    keep = tl.arange(0, CHUNK) < k
    if ASCENDING:
        dom = tl.where(valid, ~sk, 0)
        top = tl.sort(dom, dim=0, descending=True)
        # Top sort slots holding the pad (0) must stay pads: ~0 would poison
        # the candidate list. Real keys never encode to 0.
        out_sk = tl.where(keep & (top != 0), ~top, 0)
    else:
        dom = tl.where(valid, sk, 0)
        out_sk = tl.where(keep, tl.sort(dom, dim=0, descending=True), 0)

    dst = row * (N_CHUNKS * k) + chunk * k + tl.arange(0, CHUNK)
    tl.store(cand_ptr + dst, out_sk.to(tl.int32, bitcast=True), mask=keep)


@libentry()
@triton.jit
def topk_fp8_chunk_merge_kernel(
    dst_ptr,
    src_ptr,
    SRC_LISTS: tl.constexpr,
    G: tl.constexpr,
    k: tl.constexpr,
    DST_LISTS: tl.constexpr,
    ASCENDING: tl.constexpr,
    MBLOCK: tl.constexpr,
):
    # Merge G sorted candidate lists (k entries each) of one row into one.
    # Pads (0) sort below every real candidate, and slots kept past the real
    # prefix are re-zeroed so they stay pads downstream.
    row = tle.program_id(0)
    g = tle.program_id(1)
    cols = tl.arange(0, MBLOCK)
    n_in = tl.minimum(G, SRC_LISTS - g * G)
    valid = cols < n_in * k

    src_list = g * G + cols // k
    src_i = cols % k
    # Reloaded candidates are stored as int32; bitcast back to the unsigned
    # sort domain before comparing — a signed sort would rank every key with
    # the top bit set (all positive values) below the 0 pads.
    sk = tl.load(
        src_ptr + row * (SRC_LISTS * k) + src_list * k + src_i,
        mask=valid,
        other=0,
    ).to(tl.uint32, bitcast=True)
    if ASCENDING:
        # A source entry may itself be a pad (tail chunks emit fewer than k
        # real keys); ~pad would become 0xFFFFFFFF and break the sort. Kill
        # pads before they enter the domain, and again on the sorted output.
        dom = tl.where(valid & (sk != 0), ~sk, 0)
        top = tl.sort(dom, dim=0, descending=True)
        out_sk = tl.where((cols < k) & (top != 0), ~top, 0)
    else:
        dom = tl.where(valid, sk, 0)
        top = tl.sort(dom, dim=0, descending=True)
        out_sk = tl.where(cols < k, top, 0)

    keep = cols < k
    dst = row * (DST_LISTS * k) + g * k + cols
    tl.store(dst_ptr + dst, out_sk.to(tl.int32, bitcast=True), mask=keep)


@libentry()
@triton.jit
def topk_fp8_chunk_decode_kernel(
    y_ptr,
    index_ptr,
    cand_ptr,
    x_ptr,
    scale_ptr,
    N: tl.constexpr,
    SRC_LISTS: tl.constexpr,
    k: tl.constexpr,
    ASCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Final stage of every multi-kernel path: sort the row's merged
    # candidates (pads are 0 and sort last) and decode the top k. Values are
    # recomputed from the original FP8 payload by the decoded index rather
    # than unpacked from the sort key, so NaN payloads and the output dtype
    # stay exact.
    row = tle.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < SRC_LISTS * k
    # Unsigned sort domain (see merge kernel): int32 storage bitcast back.
    sk = tl.load(cand_ptr + row * (SRC_LISTS * k) + cols, mask=valid, other=0).to(
        tl.uint32, bitcast=True
    )
    if ASCENDING:
        dom = tl.where(valid & (sk != 0), ~sk, 0)
        top = tl.sort(dom, dim=0, descending=True)
        sk_top = tl.where(top != 0, ~top, 0)
    else:
        dom = tl.where(valid, sk, 0)
        sk_top = tl.sort(dom, dim=0, descending=True)

    keep = cols < k
    index = (0xFFFE - (sk_top & 0xFFFF)).to(tl.int64)
    sel = tl.where(keep & (sk_top != 0), index, 0)
    bits = tl.load(x_ptr + row * N + sel, mask=keep, other=0).to(tl.uint8)
    x_q = _fp8_e5_to_f32(bits)
    scale = tl.load(
        scale_ptr + row * tl.cdiv(N, GROUP_SIZE) + (sel // GROUP_SIZE),
        mask=keep,
        other=0.0,
    ).to(tl.float32)
    value = (x_q * scale).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * k + cols, value, mask=keep)
    tl.store(index_ptr + row * k + cols, index, mask=keep)


# ---------------------------------------------------------------------------
# Radix-select path: one parallel 65536-bin histogram pass over the u16 keys,
# a tiny per-row scan that pins the k-th largest key, and two collect passes
# (keys above the threshold, then ties truncated at k). Shares the A16 u16-key
# domain and the candidate format with the chunked path, so the chunked decode
# kernel serves as its final stage. Descending only (ascending rows stay on
# the chunked path). The earlier 4-level per-row radix scan was serial per row
# and dominated the route's latency; the histogram moves that work into the
# already-parallel element pass.
# ---------------------------------------------------------------------------

HIST_BINS = 65536  # full u16 key domain, one bin per key


@triton.jit
def _fp8_row_col_key(
    x_ptr, scale_ptr, row, cols, m, N: tl.constexpr, GROUP_SIZE: tl.constexpr
):
    # Monotone u16 key of the dequantized A16 value at (row, cols); the
    # bit-level recipe matches the chunked path exactly.
    bits = tl.load(x_ptr + row.to(tl.int64) * N + cols, mask=m, other=0).to(tl.uint8)
    q = _fp8_e5_to_f32(bits)
    s = tl.load(
        scale_ptr + row.to(tl.int64) * tl.cdiv(N, GROUP_SIZE) + cols // GROUP_SIZE,
        mask=m,
        other=0.0,
    ).to(tl.float32)
    v = (q * s).to(tl.bfloat16)
    vb = v.to(tl.uint16, bitcast=True)
    vf = tl.where((vb & 0x8000) != 0, 0xFFFF, 0x8000).to(tl.uint16)
    key = tl.where(v != v, 0xFFFF, vb ^ vf).to(tl.uint16)
    return key


@libentry()
@triton.jit
def topk_fp8_radix_hist_kernel(
    hist_ptr,
    x_ptr,
    scale_ptr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NBINS: tl.constexpr,
):
    # One program per (row, tile): dequantize, map to the u16 key, and bump
    # its histogram bin. Duplicate keys contend on one atomic, but ordinary
    # quantized data spreads over thousands of bins.
    row = tle.program_id(0)
    t = tle.program_id(1)
    cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
    m = cols < N
    key = _fp8_row_col_key(x_ptr, scale_ptr, row, cols, m, N, GROUP_SIZE)
    tl.atomic_add(hist_ptr + row.to(tl.int64) * NBINS + key.to(tl.int32), 1, mask=m)


@libentry()
@triton.jit
def topk_fp8_radix_thr16_kernel(
    hist_ptr,
    thr_ptr,
    K: tl.constexpr,
    NBINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Scan one row's bins ascending and store the k-th largest key. With
    # E[b] = #keys < b (nondecreasing) and D = total - K, the k-th largest
    # key is the largest b with E[b] <= D, i.e. one below the first bin
    # where E crosses D. No crossing inside the table means every key is
    # the same top value, so the threshold is the last bin.
    row = tle.program_id(0)
    bins = tl.arange(0, BLOCK)
    hrow = hist_ptr + row.to(tl.int64) * NBINS

    total = tl.full((), 0, dtype=tl.int32)
    for c in tl.range(0, NBINS // BLOCK):
        counts = tl.load(hrow + c * BLOCK + bins)
        total += tl.sum(counts, 0)
    D = total - K

    E = tl.full((), 0, dtype=tl.int32)
    done = tl.full((), 0, dtype=tl.int32)
    thr_val = tl.full((), 0, dtype=tl.int32)
    for c in tl.range(0, NBINS // BLOCK):
        counts = tl.load(hrow + c * BLOCK + bins)
        incl = tl.cumsum(counts, 0)
        E_in = E + incl - counts
        mask = E_in > D
        if tl.max(mask.to(tl.int32), 0) != 0:
            if done == 0:
                first = tl.min(tl.where(mask, bins, BLOCK), 0)
                thr_val = c * BLOCK + first - 1
                done = 1
        E += tl.sum(counts, 0)
    if done == 0:
        thr_val = NBINS - 1
    tl.store(thr_ptr + row, thr_val.to(tl.int16))


@libentry()
@triton.jit
def topk_fp8_radix_collect_kernel(
    cand_ptr,
    ctr_ptr,
    x_ptr,
    scale_ptr,
    thr_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TAKE_EQUAL: tl.constexpr,
):
    # Compact qualifying keys into the dense candidate list. Keys are
    # recomputed inline from the FP8 payload (no materialized key buffer).
    # Positions are claimed with a per-row atomic; the GT pass fills
    # [0, n_gt) and the EQ pass appends the ties, truncated at K.
    row = tle.program_id(0)
    tile = tle.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    m = cols < N
    key = _fp8_row_col_key(x_ptr, scale_ptr, row, cols, m, N, GROUP_SIZE)
    thr = tl.load(thr_ptr + row).to(tl.uint16, bitcast=True)
    if TAKE_EQUAL:
        take = m & (key == thr)
    else:
        take = m & (key > thr)

    n = tl.sum(take.to(tl.int32), axis=0)
    base = tl.atomic_add(ctr_ptr + row, n)
    csum = tl.cumsum(take.to(tl.int32), axis=0)
    pos = base + csum - 1
    keep = take & (pos < K)

    sk = (key.to(tl.uint32) << 16) | (0xFFFE - cols).to(tl.uint32)
    tl.store(cand_ptr + row * K + pos, sk.to(tl.int32, bitcast=True), mask=keep)


@libentry()
@triton.jit
def topk_fp8_one_group_packed_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    bits = tl.load(x_ptr + pid * N + cols, mask=mask, other=0).to(tl.uint8)
    ordered_key = _fp8_bits_to_ordered_key(bits)
    # E5M2 NaN is an all-ones exponent with a nonzero mantissa
    # ((bits & 0x7F) > 0x7C); testing the bits directly avoids converting the
    # whole row to f32 just to compare value != value.
    is_nan = (bits & 0x7F) > 0x7C
    ordered_key = tl.where(is_nan, 0xFF, ordered_key).to(tl.uint8)
    index_key = (0xFFFF - cols).to(tl.uint32)
    packed = (ordered_key.to(tl.uint32) << 16) | index_key
    packed = tl.where(mask, packed, 0)
    packed = tl.sort(packed, dim=0, descending=True)

    out_mask = cols < k
    selected_key = (packed >> 16).to(tl.uint8)
    selected_flip = tl.where((selected_key & 0x80) != 0, 0x80, 0xFF).to(tl.uint8)
    selected_bits = selected_key ^ selected_flip
    selected_values = _fp8_e5_to_f32(selected_bits)
    scale = tl.load(scale_ptr + pid).to(tl.float32)
    selected_indices = (0xFFFF - (packed & 0xFFFF)).to(tl.int64)
    tl.store(
        y_ptr + pid * k + cols,
        selected_values * scale,
        mask=out_mask,
    )
    tl.store(index_ptr + pid * k + cols, selected_indices, mask=out_mask)


@libentry()
@triton.jit
def topk_fp8_two_group_packed_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    local = tl.arange(0, GROUP_SIZE)
    row_start = pid * 2 * GROUP_SIZE

    bits_0 = tl.load(x_ptr + row_start + local).to(tl.uint8)
    bits_1 = tl.load(x_ptr + row_start + GROUP_SIZE + local).to(tl.uint8)
    key_0 = _fp8_bits_to_ordered_key(bits_0)
    key_1 = _fp8_bits_to_ordered_key(bits_1)
    # E5M2 NaN is an all-ones exponent with a nonzero mantissa
    # ((bits & 0x7F) > 0x7C); test bits instead of an f32 NaN compare.
    nan_0 = (bits_0 & 0x7F) > 0x7C
    nan_1 = (bits_1 & 0x7F) > 0x7C
    key_0 = tl.where(nan_0, 0xFF, key_0).to(tl.uint8)
    key_1 = tl.where(nan_1, 0xFF, key_1).to(tl.uint8)
    packed_0 = (key_0.to(tl.uint32) << 16) | (0xFFFF - local).to(tl.uint32)
    packed_1 = (key_1.to(tl.uint32) << 16) | (0xFFFF - GROUP_SIZE - local).to(tl.uint32)
    packed_0 = tl.sort(packed_0, dim=0, descending=True)
    packed_1 = tl.sort(packed_1, dim=0, descending=True)

    candidate_offsets = tl.arange(0, 2 * k)
    group_ranks = candidate_offsets % k
    selected_0 = tl.gather(packed_0, group_ranks, axis=0)
    selected_1 = tl.gather(packed_1, group_ranks, axis=0)
    selected = tl.where(candidate_offsets < k, selected_0, selected_1)

    raw_key = (selected >> 16).to(tl.uint8)
    raw_flip = tl.where((raw_key & 0x80) != 0, 0x80, 0xFF).to(tl.uint8)
    raw_bits = raw_key ^ raw_flip
    raw_values = _fp8_e5_to_f32(raw_bits)
    group_idx = (candidate_offsets >= k).to(tl.int32)
    scale = tl.load(scale_ptr + pid * 2 + group_idx).to(tl.float32)
    values = (raw_values * scale).to(tl.bfloat16)
    indices = (0xFFFF - (selected & 0xFFFF)).to(tl.int32)

    value_bits = values.to(tl.uint16, bitcast=True)
    value_flip = tl.where((value_bits & 0x8000) != 0, 0xFFFF, 0x8000).to(tl.uint16)
    value_key = value_bits ^ value_flip
    value_key = tl.where(values != values, 0xFFFF, value_key).to(tl.uint16)
    merged = (value_key.to(tl.uint32) << 16) | (0xFFFF - indices).to(tl.uint32)
    merged = tl.sort(merged, dim=0, descending=True)

    out_mask = candidate_offsets < k
    out_key = (merged >> 16).to(tl.uint16)
    out_flip = tl.where((out_key & 0x8000) != 0, 0x8000, 0xFFFF).to(tl.uint16)
    out_bits = out_key ^ out_flip
    final_values = out_bits.to(tl.bfloat16, bitcast=True)
    final_indices = (0xFFFF - (merged & 0xFFFF)).to(tl.int64)
    tl.store(
        y_ptr + pid * k + candidate_offsets,
        final_values,
        mask=out_mask,
    )
    tl.store(
        index_ptr + pid * k + candidate_offsets,
        final_indices,
        mask=out_mask,
    )


def _launch_chunked_topk(
    y_vals_2d,
    y_idx_2d,
    x_2d,
    scale_2d,
    batch_size,
    n,
    k,
    descending,
    group_size,
):
    # Stage 1: each program reduces one CHUNK of one row to its k best
    # candidates. Stage 2: hierarchically merge G lists at a time until one
    # list per row remains. Stage 3: decode the merged list.
    n_chunks = triton.cdiv(n, CHUNK_SIZE)
    # A chunk of CHUNK elements can emit at most CHUNK candidates, but the
    # lists are k-strided: when k > CHUNK each list leaves k - CHUNK slots
    # unwritten. Those gaps must read as pads (0), never garbage.
    cand = (
        torch.zeros((batch_size, n_chunks * k), device=x_2d.device, dtype=torch.int32)
        if k > CHUNK_SIZE
        else torch.empty(
            (batch_size, n_chunks * k), device=x_2d.device, dtype=torch.int32
        )
    )
    with torch_device_fn.device(x_2d.device):
        topk_fp8_chunk_stage1_kernel[(batch_size, n_chunks)](
            cand,
            x_2d,
            scale_2d,
            k,
            n,
            CHUNK_SIZE,
            n_chunks,
            not descending,
            group_size,
            num_warps=4,
            num_stages=1,
        )
        src = cand
        src_lists = n_chunks
        while src_lists > 1:
            g = 2 if src_lists > 2 else src_lists
            # Cap the merge fan-in so MBLOCK = G * k stays within the probed
            # tl.sort width (2048); the loop naturally handles any remainder.
            while g > 2 and g * k > 2048:
                g //= 2
            dst_lists = triton.cdiv(src_lists, g)
            mblock = triton.next_power_of_2(max(g * k, 32))
            dst = torch.empty(
                (batch_size, dst_lists * k), device=x_2d.device, dtype=torch.int32
            )
            topk_fp8_chunk_merge_kernel[(batch_size, dst_lists)](
                dst,
                src,
                src_lists,
                g,
                k,
                dst_lists,
                not descending,
                mblock,
                num_warps=4,
                num_stages=1,
            )
            src = dst
            src_lists = dst_lists
        block = triton.next_power_of_2(max(src_lists * k, k))
        topk_fp8_chunk_decode_kernel[(batch_size,)](
            y_vals_2d,
            y_idx_2d,
            src,
            x_2d,
            scale_2d,
            n,
            src_lists,
            k,
            not descending,
            group_size,
            block,
            num_warps=4,
            num_stages=1,
        )
    return y_vals_2d, y_idx_2d


def _launch_radix_topk(
    y_vals_2d,
    y_idx_2d,
    x_2d,
    scale_2d,
    batch_size,
    n,
    k,
    group_size,
):
    # Pass 0: 65536-bin histogram per row. Pass 1: per-row scan for the k-th
    # largest key. Pass 2a/2b: compact keys > thr, then == thr truncated at
    # k. Pass 3: shared decode kernel.
    block_n = 1024
    n_tiles = triton.cdiv(n, block_n)
    hist = torch.zeros((batch_size, HIST_BINS), device=x_2d.device, dtype=torch.int32)
    thr = torch.empty((batch_size,), device=x_2d.device, dtype=torch.int16)
    ctr = torch.zeros((batch_size,), device=x_2d.device, dtype=torch.int32)
    # EQ-pass overflow leaves tail slots unwritten; zeros keep them pads.
    cand = torch.zeros((batch_size, k), device=x_2d.device, dtype=torch.int32)
    with torch_device_fn.device(x_2d.device):
        topk_fp8_radix_hist_kernel[(batch_size, n_tiles)](
            hist,
            x_2d,
            scale_2d,
            n,
            group_size,
            block_n,
            HIST_BINS,
            num_warps=4,
            num_stages=1,
        )
        topk_fp8_radix_thr16_kernel[(batch_size,)](
            hist, thr, k, HIST_BINS, 8192, num_warps=8, num_stages=1
        )
        topk_fp8_radix_collect_kernel[(batch_size, n_tiles)](
            cand,
            ctr,
            x_2d,
            scale_2d,
            thr,
            k,
            n,
            group_size,
            block_n,
            False,
            num_warps=4,
            num_stages=1,
        )
        topk_fp8_radix_collect_kernel[(batch_size, n_tiles)](
            cand,
            ctr,
            x_2d,
            scale_2d,
            thr,
            k,
            n,
            group_size,
            block_n,
            True,
            num_warps=4,
            num_stages=1,
        )
        topk_fp8_chunk_decode_kernel[(batch_size,)](
            y_vals_2d,
            y_idx_2d,
            cand,
            x_2d,
            scale_2d,
            n,
            1,
            k,
            False,
            group_size,
            triton.next_power_of_2(max(k, 32)),
            num_warps=4,
            num_stages=1,
        )
    return y_vals_2d, y_idx_2d


@triton.jit
def _topk_needs_radix(state, row, K: tl.constexpr, CAP: tl.constexpr):
    if CAP == 0:
        needs_radix = tl.full((), True, tl.int1)
    else:
        greater = tl.load(state + row * 3 + 1)
        equal = tl.load(state + row * 3 + 2)
        needs_radix = (greater > CAP) | (greater + equal < K)
    return needs_radix


@libentry()
@triton.jit
def _clear_failed_row_histograms(
    histogram,
    counters,
    state,
    K: tl.constexpr,
    FILTER_CAP: tl.constexpr,
    ROWS: tl.constexpr,
    SHARDS: tl.constexpr,
):
    row = tl.program_id(0)
    if _topk_needs_radix(state, row, K, FILTER_CAP):
        bins = tl.arange(0, SHARDS * 256)
        address = histogram + row * SHARDS * 256 + bins
        tl.store(address, 0)
        tl.store(address + ROWS * SHARDS * 256, 0)
        tl.store(counters + row * 2 + tl.arange(0, 2), 0)


@libentry()
@triton.jit
def _partitioned_radix_histogram(
    hist_ptr,
    prefix_ptr,
    x_ptr,
    scale_ptr,
    N: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    SHARDS: tl.constexpr,
    LOW_BYTE: tl.constexpr,
    ASCENDING: tl.constexpr,
    K: tl.constexpr,
    filter_state,
    FILTER_CAP: tl.constexpr,
):
    row = tl.program_id(0)
    if _topk_needs_radix(filter_state, row, K, FILTER_CAP):
        tile = tl.program_id(1)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        valid = cols < N
        key = _partitioned_radix_key(
            x_ptr,
            scale_ptr,
            row,
            cols,
            valid,
            ROW_STRIDE,
            SCALE_STRIDE,
            GROUP_SIZE,
            ASCENDING,
        )
        if LOW_BYTE:
            prefix = tl.load(prefix_ptr + row * 2)
            valid &= (key >> 8) == prefix
            digit = key & 255
        else:
            digit = key >> 8
        counts = tl.histogram(tl.where(valid, digit, 256).to(tl.int32), 256)
        bins = tl.arange(0, 256)
        shard = tile % SHARDS
        tl.atomic_add(
            hist_ptr + (row * SHARDS + shard) * 256 + bins,
            counts,
            mask=counts != 0,
            sem="relaxed",
        )


@triton.jit
def _partitioned_radix_key(
    x_ptr,
    scale_ptr,
    row,
    cols,
    valid,
    ROW_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ASCENDING: tl.constexpr,
):
    bits = tl.load(x_ptr + row * ROW_STRIDE + cols, valid, 0)
    scale = tl.load(scale_ptr + row * SCALE_STRIDE + cols // GROUP_SIZE, valid, 0)
    value = (_fp8_e5_to_f32(bits) * scale.to(tl.float32)).to(tl.bfloat16)
    value_bits = value.to(tl.uint16, bitcast=True)
    flip = tl.where((value_bits & 0x8000) != 0, 0xFFFF, 0x8000).to(tl.uint16)
    key = tl.where(value != value, 0xFFFF, value_bits ^ flip).to(tl.uint16)
    if ASCENDING:
        key = ~key
    return key.to(tl.int32)


@libentry()
@triton.jit
def _partitioned_radix_threshold(
    prefix_ptr,
    hist_ptr,
    K: tl.constexpr,
    SHARDS: tl.constexpr,
    LOW_BYTE: tl.constexpr,
    filter_state,
    FILTER_CAP: tl.constexpr,
):
    row = tl.program_id(0)
    if _topk_needs_radix(filter_state, row, K, FILTER_CAP):
        shards = tl.arange(0, SHARDS)
        bins = tl.arange(0, 256)
        counts = tl.load(
            hist_ptr + (row * SHARDS + shards[:, None]) * 256 + bins[None, :]
        )
        counts = tl.sum(counts, axis=0)
        suffix = tl.cumsum(counts, 0, reverse=True)
        if LOW_BYTE:
            rank = tl.load(prefix_ptr + row * 2 + 1)
        else:
            rank = K
        bucket = tl.max(tl.where(suffix >= rank, bins, 0), 0)
        if LOW_BYTE:
            high = tl.load(prefix_ptr + row * 2)
            tl.store(prefix_ptr + row * 2, (high << 8) | bucket)
        else:
            greater = tl.sum(tl.where(bins > bucket, counts, 0), 0)
            tl.store(prefix_ptr + row * 2, bucket)
            tl.store(prefix_ptr + row * 2 + 1, K - greater)


@libentry()
@triton.jit
def _partitioned_radix_collect(
    candidates,
    counters,
    prefix_ptr,
    x_ptr,
    scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ASCENDING: tl.constexpr,
    filter_state,
    FILTER_CAP: tl.constexpr,
):
    row = tl.program_id(0)
    if _topk_needs_radix(filter_state, row, K, FILTER_CAP):
        tile = tl.program_id(1)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        valid = cols < N
        cutoff = tl.load(prefix_ptr + row * 2)
        key = _partitioned_radix_key(
            x_ptr,
            scale_ptr,
            row,
            cols,
            valid,
            ROW_STRIDE,
            SCALE_STRIDE,
            GROUP_SIZE,
            ASCENDING,
        )
        # Separate counters allow GT and EQ to compact concurrently. The final
        # sort keeps all GT entries, then fills the remaining ranks with ties.
        for bucket in tl.static_range(2):
            take = valid & (key > cutoff if bucket == 0 else key == cutoff)
            count = tl.sum(take.to(tl.int32), 0)
            if count > 0:
                base = tl.atomic_add(counters + row * 2 + bucket, count, sem="relaxed")
                ranks = base + tl.cumsum(take.to(tl.int32), 0) - 1
                tl.store(
                    candidates + (row * 2 + bucket) * K + ranks,
                    cols,
                    take & (ranks < K),
                )


@libentry()
@triton.jit
def _partitioned_radix_finish(
    y_ptr,
    index_ptr,
    candidates,
    counters,
    x_ptr,
    scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    ASCENDING: tl.constexpr,
    filter_state,
    FILTER_CAP: tl.constexpr,
):
    row = tl.program_id(0)
    if _topk_needs_radix(filter_state, row, K, FILTER_CAP):
        offsets = tl.arange(0, WIDTH)
        bucket = offsets // K
        rank = offsets % K
        count = tl.load(counters + row * 2 + bucket, bucket < 2, 0)
        valid = (bucket < 2) & (rank < count)
        indices = tl.load(candidates + row * 2 * K + offsets, valid, 0)
        key = _partitioned_radix_key(
            x_ptr,
            scale_ptr,
            row,
            indices,
            valid,
            ROW_STRIDE,
            SCALE_STRIDE,
            GROUP_SIZE,
            ASCENDING,
        )
        if N <= 65535:
            packed = (key.to(tl.uint32) << 16) | (65535 - indices).to(tl.uint32)
            best = tl.sort(tl.where(valid, packed, 0), descending=True)
            index = (65535 - (best & 65535)).to(tl.int32)
        else:
            packed = (key.to(tl.uint64) << 32) | (
                0xFFFFFFFF - indices.to(tl.uint32)
            ).to(tl.uint64)
            best = tl.sort(tl.where(valid, packed, 0), descending=True)
            index = (0xFFFFFFFF - best.to(tl.uint32)).to(tl.int32)
        keep = offsets < K
        raw = tl.load(x_ptr + row * ROW_STRIDE + index, keep, 0)
        scale = tl.load(scale_ptr + row * SCALE_STRIDE + index // GROUP_SIZE, keep, 0)
        tl.store(
            y_ptr + row * K + offsets, _fp8_e5_to_f32(raw) * scale.to(tl.float32), keep
        )
        tl.store(index_ptr + row * K + offsets, index.to(tl.int64), keep)


@libentry()
@triton.jit
def _single_cta_topk(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    K_PAD: tl.constexpr,
    ASCENDING: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    key = _partitioned_radix_key(
        x_ptr,
        scale_ptr,
        row,
        cols,
        valid,
        ROW_STRIDE,
        SCALE_STRIDE,
        GROUP_SIZE,
        ASCENDING,
    )
    packed = (key.to(tl.uint32) << 16) | (65535 - cols).to(tl.uint32)
    selected = tl.topk(tl.where(valid, packed, 0), K_PAD, dim=0)
    index = (65535 - (selected & 65535)).to(tl.int32)
    rank = tl.arange(0, K_PAD)
    keep = rank < K
    raw = tl.load(x_ptr + row * ROW_STRIDE + index, keep, 0)
    scale = tl.load(scale_ptr + row * SCALE_STRIDE + index // GROUP_SIZE, keep, 0)
    tl.store(y_ptr + row * K + rank, _fp8_e5_to_f32(raw) * scale.to(tl.float32), keep)
    tl.store(index_ptr + row * K + rank, index.to(tl.int64), keep)


@libentry()
@triton.jit
def _sample_topk_cutoff(
    state,
    x,
    scale,
    N: tl.constexpr,
    RS: tl.constexpr,
    SS: tl.constexpr,
    G: tl.constexpr,
    SAMPLES: tl.constexpr,
    SAMPLE_K: tl.constexpr,
    RUN: tl.constexpr,
    ASC: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, SAMPLES)
    cols = (lanes // RUN) * (N * RUN // SAMPLES) + lanes % RUN
    keys = _partitioned_radix_key(x, scale, row, cols, cols < N, RS, SS, G, ASC)
    best = tl.topk(keys, SAMPLE_K, dim=0)
    tl.store(state + row * 3, tl.min(best, 0))
    tl.store(state + row * 3 + 1, 0)
    tl.store(state + row * 3 + 2, 0)


@libentry()
@triton.jit
def _collect_sampled_topk(
    greater_ptr,
    equal_ptr,
    state,
    x,
    scale,
    N: tl.constexpr,
    RS: tl.constexpr,
    SS: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    ASC: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    idx = tile * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    cutoff = tl.load(state + row * 3)
    key = _partitioned_radix_key(x, scale, row, idx, valid, RS, SS, G, ASC)
    gt = valid & (key > cutoff)
    eq = valid & (key == cutoff)
    votes = gt.to(tl.int32) + (eq.to(tl.int32) << 16)
    # Each tile has fewer than 2**16 elements, so the two counts cannot carry.
    counts = tl.sum(votes, 0)
    gt_count = counts & 65535
    eq_count = counts >> 16
    if counts > 0:
        prefix = tl.cumsum(votes, 0)
        if gt_count > 0:
            base = tl.atomic_add(state + row * 3 + 1, gt_count, sem="relaxed")
            offset = base + (prefix & 65535) - 1
            packed = (key.to(tl.uint64) << 32) | (0xFFFFFFFF - idx.to(tl.uint32)).to(
                tl.uint64
            )
            tl.store(greater_ptr + row * CAP + offset, packed, gt & (offset < CAP))
        if eq_count > 0:
            base = tl.atomic_add(state + row * 3 + 2, eq_count, sem="relaxed")
            offset = base + (prefix >> 16) - 1
            tl.store(equal_ptr + row * K + offset, idx, eq & (offset < K))


@libentry()
@triton.jit
def _finish_sampled_topk(
    values,
    indices,
    greater_ptr,
    equal_ptr,
    state,
    x,
    scale,
    N: tl.constexpr,
    RS: tl.constexpr,
    SS: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    CAP: tl.constexpr,
    K_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    gt_count = tl.load(state + row * 3 + 1)
    if not _topk_needs_radix(state, row, K, CAP):
        r = tl.arange(0, CAP)
        packed = tl.load(greater_ptr + row * CAP + r, r < gt_count, 0)
        best = tl.topk(packed, K_PAD, dim=0)
        rank = tl.arange(0, K_PAD)
        idx = (0xFFFFFFFF - best.to(tl.uint32)).to(tl.int32)
        tail = tl.load(
            equal_ptr + row * K + rank - gt_count, (rank >= gt_count) & (rank < K), 0
        )
        idx = tl.where(rank < gt_count, idx, tail)
        q = tl.load(x + row * RS + idx, rank < K, 0)
        s = tl.load(scale + row * SS + idx // G, rank < K, 0)
        tl.store(
            values + row * K + rank, _fp8_e5_to_f32(q) * s.to(tl.float32), rank < K
        )
        tl.store(indices + row * K + rank, idx.to(tl.int64), rank < K)


def _launch_partitioned_radix(
    values, indices, x, scale, k, group_size, ascending, filter_state=None, filter_cap=0
):
    rows, n = x.shape
    if n <= 65534:
        block, max_shards = 4096, 8
    elif rows <= 16:
        block, max_shards = 1024, 16
    else:
        block, max_shards = 2048, 8
    tiles = triton.cdiv(n, block)
    shards = min(max_shards, triton.next_power_of_2(tiles))
    hist_shape = (2, rows, shards, 256)
    if filter_cap:
        histogram = torch.empty(hist_shape, device=x.device, dtype=torch.int32)
        counters = torch.empty((rows, 2), device=x.device, dtype=torch.int32)
    else:
        histogram = torch.zeros(hist_shape, device=x.device, dtype=torch.int32)
        counters = torch.zeros((rows, 2), device=x.device, dtype=torch.int32)
    prefix = torch.empty((rows, 2), device=x.device, dtype=torch.int32)
    candidates = torch.empty((rows, 2 * k), device=x.device, dtype=torch.int32)
    stride, scale_stride = x.stride(0), scale.stride(0)
    with torch_device_fn.device(x.device):
        if filter_cap:
            _clear_failed_row_histograms[(rows,)](
                histogram,
                counters,
                filter_state,
                k,
                filter_cap,
                rows,
                shards,
                num_warps=4,
            )
        for low in (False, True):
            hist = histogram[int(low)]
            _partitioned_radix_histogram[(rows, tiles)](
                hist,
                prefix,
                x,
                scale,
                n,
                stride,
                scale_stride,
                group_size,
                block,
                shards,
                low,
                ascending,
                k,
                filter_state,
                filter_cap,
                num_warps=4,
            )
            _partitioned_radix_threshold[(rows,)](
                prefix,
                hist,
                k,
                shards,
                low,
                filter_state,
                filter_cap,
                num_warps=4,
            )
        _partitioned_radix_collect[(rows, tiles)](
            candidates,
            counters,
            prefix,
            x,
            scale,
            n,
            k,
            stride,
            scale_stride,
            group_size,
            block,
            ascending,
            filter_state,
            filter_cap,
            num_warps=4,
        )
        _partitioned_radix_finish[(rows,)](
            values,
            indices,
            candidates,
            counters,
            x,
            scale,
            n,
            k,
            stride,
            scale_stride,
            group_size,
            triton.next_power_of_2(2 * k),
            ascending,
            filter_state,
            filter_cap,
            num_warps=4,
        )


def _launch_sampled_topk(values, indices, x, scale, k, group_size, ascending):
    rows, n = x.shape
    # The sample only predicts a cutoff. The full scan counts every value
    # above/equal to it before committing output or handing off to radix.
    sample_size, sample_k = 8192, 32
    capacity = 2048
    block = 4096
    greater = torch.empty((rows, capacity), device=x.device, dtype=torch.uint64)
    equal = torch.empty((rows, k), device=x.device, dtype=torch.int32)
    state = torch.empty((rows, 3), device=x.device, dtype=torch.int32)
    shared = (n, x.stride(0), scale.stride(0), group_size)
    with torch_device_fn.device(x.device):
        _sample_topk_cutoff[(rows,)](
            state, x, scale, *shared, sample_size, sample_k, 32, ascending, num_warps=4
        )
        _collect_sampled_topk[(rows, triton.cdiv(n, block))](
            greater,
            equal,
            state,
            x,
            scale,
            *shared,
            k,
            capacity,
            block,
            ascending,
            num_warps=4,
        )
        _finish_sampled_topk[(rows,)](
            values,
            indices,
            greater,
            equal,
            state,
            x,
            scale,
            *shared,
            k,
            capacity,
            triton.next_power_of_2(k),
            num_warps=4,
        )
    _launch_partitioned_radix(
        values, indices, x, scale, k, group_size, ascending, state, capacity
    )


@triton.jit
def _row_radix_cutoff(key, valid, rank):
    bins = tl.arange(0, 256)
    cutoff = tl.full((), 0, tl.int32)
    for shift in tl.static_range(8, -1, -8):
        matching = valid & ((key >> (shift + 8)) == cutoff)
        digit = (key >> shift) & 255
        counts = tl.histogram(tl.where(matching, digit, 256), 256)
        suffix = tl.cumsum(counts, 0, reverse=True)
        bucket = tl.max(tl.where(suffix >= rank, bins, 0), 0)
        rank -= tl.sum(tl.where(bins > bucket, counts, 0), 0)
        cutoff = (cutoff << 8) | bucket
    return cutoff


@libentry()
@triton.jit
def _compact_row_radix_keys(
    scratch,
    x,
    scale,
    N: tl.constexpr,
    RS,
    SS,
    G: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ASC: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    key = _partitioned_radix_key(x, scale, row, cols, valid, RS, SS, G, ASC)
    cutoff = _row_radix_cutoff(key, valid, K)
    greater = valid & (key > cutoff)
    equal = valid & (key == cutoff)
    gt_total = tl.sum(greater.to(tl.int32), 0)
    # The high half counts ties. With at most 16384 elements both counts fit.
    votes = greater.to(tl.int32) + (equal.to(tl.int32) << 16)
    ranks = tl.cumsum(votes, 0)
    offset = tl.where(greater, ranks & 0xFFFF, gt_total + (ranks >> 16)) - 1
    selected = (greater | equal) & (offset < K)
    packed = (key.to(tl.uint32) << 16) | (65535 - cols).to(tl.uint32)
    tl.store(scratch + row * K + offset, packed, selected)


@libentry()
@triton.jit
def _sort_row_radix_keys(
    output,
    indices,
    scratch,
    K: tl.constexpr,
    KP: tl.constexpr,
    ASC: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, KP)
    packed = tl.load(scratch + row * K + rank, rank < K, 0)
    best = tl.sort(packed, descending=True)
    col = (65535 - (best & 65535)).to(tl.int32)
    key = (best >> 16).to(tl.uint16)
    if ASC:
        key = ~key
    flip = tl.where((key & 0x8000) != 0, 0x8000, 0xFFFF).to(tl.uint16)
    value = (key ^ flip).to(tl.bfloat16, bitcast=True)
    tl.store(output + row * K + rank, value, rank < K)
    tl.store(indices + row * K + rank, col.to(tl.int64), rank < K)


def _launch_row_radix_topk(values, indices, x, scale, k, group_size, ascending):
    rows, n = x.shape
    candidates = torch.empty((rows, k), device=x.device, dtype=torch.uint32)
    with torch_device_fn.device(x.device):
        _compact_row_radix_keys[(rows,)](
            candidates,
            x,
            scale,
            n,
            x.stride(0),
            scale.stride(0),
            group_size,
            k,
            triton.next_power_of_2(n),
            ascending,
            num_warps=4,
        )
        _sort_row_radix_keys[(rows,)](
            values,
            indices,
            candidates,
            k,
            triton.next_power_of_2(k),
            ascending,
            num_warps=4,
        )


def _parallel_topk_route(rows, n, k):
    if n == 262144 and k == 512 and rows >= 20:
        return "sampled"
    if rows >= 4096 and 2048 < n <= 16384 and k == 512:
        return "row_radix"
    if 65534 < n <= 262144 and 64 <= k <= 1024:
        return "radix"
    if 256 <= k <= 512:
        if rows >= 1024 and 1024 < n <= 2048:
            return "single"
        if (rows >= 128 and 8192 < n <= 65534) or (rows >= 1024 and 4096 < n <= 8192):
            return "radix"
    return None


def topk_w8a16_fp8(
    x_fp8,
    x_scale,
    k,
    dim=-1,
    largest=True,
    sorted=True,
    group_size=128,
    out_dtype=torch.bfloat16,
):
    logger.debug("GEMS_METAX TOPK_FP8E5_W8A16")
    if dim < 0:
        dim = dim + x_fp8.ndim

    assert dim == x_fp8.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"
    if x_fp8.dtype != torch.float8_e5m2:
        if x_fp8.itemsize != 1:
            raise TypeError(
                f"topk_w8a16_fp8 expects float8_e5m2 (or 8-bit) storage, got {x_fp8.dtype}"
            )
        x_fp8 = x_fp8.view(torch.float8_e5m2)

    if k == 0:
        out_shape = list(x_fp8.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype),
            torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64),
        )

    topk_elem_cnt = x_fp8.shape[dim]
    batch_size = math.prod(x_fp8.shape) // topk_elem_cnt
    num_groups = triton.cdiv(topk_elem_cnt, group_size)
    expected_scale_shape = x_fp8.shape[:-1] + (num_groups,)
    assert (
        x_scale.shape == expected_scale_shape
    ), f"x_scale shape should be {expected_scale_shape}, got {x_scale.shape}"

    x_2d = x_fp8.view(torch.uint8).reshape(batch_size, topk_elem_cnt)
    scale_2d = x_scale.reshape(batch_size, num_groups)
    descending = True if largest else False

    out_shape = x_fp8.shape[:-1] + (k,)
    y_vals = torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype)
    y_idx = torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64)
    y_vals_2d = y_vals.reshape(batch_size, k)
    y_idx_2d = y_idx.reshape(batch_size, k)

    if out_dtype == torch.bfloat16 and x_2d.stride(1) == scale_2d.stride(1) == 1:
        route = _parallel_topk_route(batch_size, topk_elem_cnt, k)
        if route == "sampled":
            _launch_sampled_topk(
                y_vals_2d, y_idx_2d, x_2d, scale_2d, k, group_size, not descending
            )
            return y_vals, y_idx
        if route == "row_radix":
            _launch_row_radix_topk(
                y_vals_2d, y_idx_2d, x_2d, scale_2d, k, group_size, not descending
            )
            return y_vals, y_idx
        if route == "radix":
            _launch_partitioned_radix(
                y_vals_2d, y_idx_2d, x_2d, scale_2d, k, group_size, not descending
            )
            return y_vals, y_idx
        if route == "single":
            with torch_device_fn.device(x_fp8.device):
                _single_cta_topk[(batch_size,)](
                    y_vals_2d,
                    y_idx_2d,
                    x_2d,
                    scale_2d,
                    topk_elem_cnt,
                    k,
                    x_2d.stride(0),
                    scale_2d.stride(0),
                    group_size,
                    triton.next_power_of_2(topk_elem_cnt),
                    triton.next_power_of_2(k),
                    not descending,
                    num_warps=4,
                )
            return y_vals, y_idx

    # A single non-negative quantization scale preserves the raw FP8 order.
    # Sort packed 8-bit keys and only dequantize the selected values.
    if descending and sorted and num_groups == 1 and topk_elem_cnt <= 128:
        block_size = triton.next_power_of_2(topk_elem_cnt)
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_one_group_packed_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                x_2d,
                scale_2d,
                k,
                topk_elem_cnt,
                block_size,
                num_warps=8,
                num_stages=1,
            )
        return (y_vals, y_idx)

    # For two 128-element groups, keep only each group's K best raw FP8
    # candidates, dequantize 2K values, then merge those candidates in BF16.
    if (
        descending
        and sorted
        and out_dtype == torch.bfloat16
        and group_size == 128
        and topk_elem_cnt == 2 * group_size
        and k == triton.next_power_of_2(k)
    ):
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_two_group_packed_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                x_2d,
                scale_2d,
                k,
                group_size,
                num_warps=2,
                num_stages=1,
            )
        return (y_vals, y_idx)

    if topk_elem_cnt <= 512:
        block_size = triton.next_power_of_2(topk_elem_cnt)
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_single_stage_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                x_2d,
                scale_2d,
                k,
                topk_elem_cnt,
                block_size,
                descending,
                group_size,
                num_groups,
            )
        return (y_vals, y_idx)

    # N > 512: chunked local top-k + hierarchical merge (both directions) for
    # moderate k; the histogram radix select competes on large N (descending
    # only). The u16 column payload caps these paths at N <= 0xFFFE; beyond
    # that (or when k exceeds the tl.sort width limit) the sequential-ish
    # running-merge path stays as the generic fallback. Thresholds are C550
    # complete-call-latency measurements; FLAGGEMS_TOPK_FP8_* env overrides.
    if k <= CHUNK_MAX_K and topk_elem_cnt <= 0xFFFE:
        if (
            descending
            and topk_elem_cnt >= RADIX_THRESHOLD_N
            and batch_size * topk_elem_cnt <= RADIX_MAX_TOTAL
            and k >= CHUNKED_MERGE_MIN_K
            and out_dtype == torch.bfloat16
        ):
            with torch_device_fn.device(x_fp8.device):
                _launch_radix_topk(
                    y_vals_2d,
                    y_idx_2d,
                    x_2d,
                    scale_2d,
                    batch_size,
                    topk_elem_cnt,
                    k,
                    group_size,
                )
            return (y_vals, y_idx)
        _launch_chunked_topk(
            y_vals_2d,
            y_idx_2d,
            x_2d,
            scale_2d,
            batch_size,
            topk_elem_cnt,
            k,
            descending,
            group_size,
        )
        return (y_vals, y_idx)

    # The buffer must hold at least K candidates, otherwise the tail of the
    # output stays unwritten. Without the THead TLE fast path there is no
    # upper dispatch branch that would keep K small here.
    k_pad = triton.next_power_of_2(max(k, 1))
    block = max(k_pad, 128)
    with torch_device_fn.device(x_fp8.device):
        topk_fp8_running_merge_kernel[(batch_size,)](
            y_vals_2d,
            y_idx_2d,
            x_2d,
            scale_2d,
            topk_elem_cnt,
            k,
            block,
            descending,
            group_size,
            num_warps=8,
        )
    return (y_vals, y_idx)
