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
from collections import namedtuple

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

from .sort import convert_to_uint_preverse_order

logger = logging.getLogger(__name__)

MedianResult = namedtuple("median", ["values", "indices"])
MAX_BLOCK_N = 128
BOOL_BLOCK_N = 1024
MAX_NDIM = 8
KEY_BLOCK_LIMIT = 32768


@triton.jit
def _median_is_nan(vals):
    vals_fp32 = vals.to(tl.float32)
    return vals_fp32 != vals_fp32


@triton.jit
def _median_keys(vals, KEY_BITS: tl.constexpr):
    if KEY_BITS == 64:
        if not vals.dtype.is_floating() and vals.dtype.primitive_bitwidth < 64:
            w = vals.to(tl.int64)
        else:
            w = vals
        return convert_to_uint_preverse_order(w, False)
    if vals.dtype.is_floating():
        w = vals
    elif vals.dtype.primitive_bitwidth < 32:
        w = vals.to(tl.int32)
    else:
        w = vals
    k = convert_to_uint_preverse_order(w, False)
    return k.to(tl.uint32)


@libentry()
@triton.jit
def median_direct_select_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    STRIDE_DIM: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    S6: tl.constexpr,
    S7: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    T4: tl.constexpr,
    T5: tl.constexpr,
    T6: tl.constexpr,
    T7: tl.constexpr,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    dtype = inp.dtype.element_ty
    max_value = get_dtype_max(dtype)
    fallback_value = get_dtype_min(dtype)

    idx = pid
    base = tl.full((), 0, dtype=tl.int64)
    if NDIM >= 8:
        if DIM != 7:
            coord = idx % S7
            idx = idx // S7
            base += coord * T7
    if NDIM >= 7:
        if DIM != 6:
            coord = idx % S6
            idx = idx // S6
            base += coord * T6
    if NDIM >= 6:
        if DIM != 5:
            coord = idx % S5
            idx = idx // S5
            base += coord * T5
    if NDIM >= 5:
        if DIM != 4:
            coord = idx % S4
            idx = idx // S4
            base += coord * T4
    if NDIM >= 4:
        if DIM != 3:
            coord = idx % S3
            idx = idx // S3
            base += coord * T3
    if NDIM >= 3:
        if DIM != 2:
            coord = idx % S2
            idx = idx // S2
            base += coord * T2
    if NDIM >= 2:
        if DIM != 1:
            coord = idx % S1
            idx = idx // S1
            base += coord * T1
    if NDIM >= 1:
        if DIM != 0:
            coord = idx % S0
            base += coord * T0
    data = tl.load(inp + base + offsets * STRIDE_DIM, mask=mask, other=max_value)

    if data.dtype.is_floating():
        nan_mask = mask & _median_is_nan(data)
        has_nan = tl.max(nan_mask.to(tl.int32), axis=0) != 0
        first_nan_idx = tl.min(tl.where(nan_mask, offsets, BLOCK_N), axis=0)
    else:
        has_nan = False
        first_nan_idx = tl.full((), 0, dtype=tl.int32)

    median_rank = (N - 1) // 2

    active = mask
    median_val = tl.full((), fallback_value, dtype=data.dtype)
    median_idx = tl.full((), 0, dtype=tl.int32)
    for select_iter in tl.static_range(0, BLOCK_N):
        select_vals = tl.where(active, data, max_value)
        cur_val = tl.min(select_vals, axis=0)
        cur_idx = tl.min(tl.where(active & (data == cur_val), offsets, BLOCK_N), axis=0)
        take = select_iter == median_rank
        median_val = tl.where(take, cur_val, median_val)
        median_idx = tl.where(take, cur_idx, median_idx)
        active = active & (offsets != cur_idx)

    if data.dtype.is_floating():
        median_val = tl.where(has_nan, float("nan"), median_val)
        median_idx = tl.where(has_nan, first_nan_idx, median_idx)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx.to(tl.int64))


@libentry()
@triton.jit
def median_key_info_chunk_kernel(
    inp,
    keybuf,
    chunk_mins,
    chunk_maxs,
    chunk_nan,
    N,
    NCHUNK,
    NCHUNK_FULL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    CHUNK: tl.constexpr,
    KEY_BITS: tl.constexpr,
    PREORDERED: tl.constexpr,
):
    # grid: M * NCHUNK_FULL; each program handles one CHUNK-sized slice of one
    # row.  NCHUNK (total slices incl. tail) is used for the keybuf row-wide
    # stride and slot layout; NCHUNK_FULL (slices that are full CHUNKs) is the
    # number of programs per row, so the row is pid // NCHUNK_FULL.  The
    # keybuf/offset address uses the chunk-major position (row, chunk_ordinal)
    # within a (M, NCHUNK, CHUNK) layout: the tail chunk of a row sits at
    # row*NCHUNK*CHUNK + (NCHUNK-1)*CHUNK, which must stay disjoint from the
    # next row's full chunks, so full-slice kernels address via
    # (row * NCHUNK + chunk) * CHUNK rather than pid * CHUNK.
    pid = ext.program_id(0)
    row = pid // NCHUNK_FULL
    chunk = pid % NCHUNK_FULL
    cols = tl.arange(0, CHUNK)
    slot = row * NCHUNK + chunk
    offsets = chunk * CHUNK + cols
    mask = offsets < N
    if PREORDERED:
        keys = tl.load(inp + row * N + offsets, mask=mask, other=0)
        keys_lo = tl.load(
            inp + row * N + offsets, mask=mask, other=0xFFFFFFFFFFFFFFFF
        )
        keys_hi = tl.load(inp + row * N + offsets, mask=mask, other=0)
    else:
        dtype = inp.dtype.element_ty
        is_float: tl.constexpr = dtype.is_floating()
        if is_float:
            min_fill = float("-inf")
            max_fill = float("inf")
        else:
            min_fill = get_dtype_min(dtype)
            max_fill = get_dtype_max(dtype)
        vals = tl.load(inp + row * N + offsets, mask=mask, other=max_fill)
        keys = _median_keys(vals, KEY_BITS)
        vals_lo = tl.load(
            inp + row * N + offsets, mask=mask, other=min_fill
        )
        vals_hi = tl.load(
            inp + row * N + offsets, mask=mask, other=max_fill
        )
        keys_lo = _median_keys(vals_lo, KEY_BITS)
        keys_hi = _median_keys(vals_hi, KEY_BITS)
    # keybuf layout: (M, NCHUNK, CHUNK); pad lanes keep the wrapper's
    # all-ones-key sentinel (only the last chunk of a row has pads).
    tl.store(keybuf + slot * CHUNK + cols, keys, mask=mask)
    cidx = row * BLOCK_R + chunk
    if not PREORDERED and vals.dtype.is_floating():
        # NaN first-index must be computed before the uint32 min/max
        # reductions below (XPU miscompile otherwise, see the single-block
        # key info kernel).  Pack = global index, sentinel = 0x7FFFFFFF.
        nan = mask & _median_is_nan(vals)
        local_first = tl.min(tl.where(nan, cols, CHUNK), axis=0)
        pack = tl.where(
            local_first < CHUNK, local_first + chunk * CHUNK, 2147483647
        )
        tl.store(chunk_nan + cidx, pack)
    else:
        tl.store(chunk_nan + cidx, 2147483647)
    lo = tl.min(keys_lo, axis=0)
    hi = tl.max(keys_hi, axis=0)
    tl.store(chunk_mins + cidx, lo)
    tl.store(chunk_maxs + cidx, hi)


@libentry()
@triton.jit
def median_count_chunk_kernel(
    keys,
    mid,
    chunk_counts,
    N,
    NCHUNK,
    NCHUNK_FULL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # grid: M * NCHUNK_FULL; count keys <= mid within one full slice.
    pid = ext.program_id(0)
    row = pid // NCHUNK_FULL
    chunk = pid % NCHUNK_FULL
    cols = tl.arange(0, CHUNK)
    slot = row * NCHUNK + chunk
    keys_v = tl.load(keys + slot * CHUNK + cols)
    mid_v = tl.load(mid + row)
    le = tl.sum((keys_v <= mid_v).to(tl.int32), axis=0)
    tl.store(chunk_counts + row * BLOCK_R + chunk, le)


@libentry()
@triton.jit
def median_update_step_kernel(
    lo,
    hi,
    counts,
    mid_buf,
    TARGET: tl.constexpr,
    KEY_BITS: tl.constexpr,
    FIRST: tl.constexpr,
):
    # grid: (M); one binary-search step from the per-row total count.
    # FIRST only materializes the initial mid.
    pid = ext.program_id(0)
    lo_v = tl.load(lo + pid)
    hi_v = tl.load(hi + pid)
    if KEY_BITS == 64:
        mid = lo_v + ((hi_v - lo_v) // 2)
    else:
        mid = ((lo_v.to(tl.int64) + hi_v.to(tl.int64)) // 2).to(tl.uint32)
    if not FIRST:
        le = tl.load(counts + pid)
        go_left = le > TARGET
        active = lo_v < hi_v
        new_hi = tl.where(go_left & active, mid, hi_v)
        new_lo = tl.where(~go_left & active, mid + 1, lo_v)
        tl.store(lo + pid, new_lo)
        tl.store(hi + pid, new_hi)
    tl.store(mid_buf + pid, mid)


@libentry()
@triton.jit
def median_key_info_partial_kernel(
    inp,
    keybuf,
    chunk_mins,
    chunk_maxs,
    chunk_nan,
    N,
    NCHUNK,
    START,
    PARTIAL,
    BLOCK_R: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TAIL_BASE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    KEY_BITS: tl.constexpr,
    PREORDERED: tl.constexpr,
):
    # grid: (M); handles the leftover tail of a row (last slice), which is
    # at most a full CHUNK but here never padded to a power of two.
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_P)
    offsets = START + cols
    mask = cols < PARTIAL
    if PREORDERED:
        keys = tl.load(inp + pid * N + offsets, mask=mask, other=0)
        keys_lo = tl.load(
            inp + pid * N + offsets, mask=mask, other=0xFFFFFFFFFFFFFFFF
        )
        keys_hi = tl.load(inp + pid * N + offsets, mask=mask, other=0)
    else:
        dtype = inp.dtype.element_ty
        is_float: tl.constexpr = dtype.is_floating()
        if is_float:
            min_fill = float("-inf")
            max_fill = float("inf")
        else:
            min_fill = get_dtype_min(dtype)
            max_fill = get_dtype_max(dtype)
        vals = tl.load(inp + pid * N + offsets, mask=mask, other=max_fill)
        keys = _median_keys(vals, KEY_BITS)
        vals_lo = tl.load(inp + pid * N + offsets, mask=mask, other=min_fill)
        vals_hi = tl.load(inp + pid * N + offsets, mask=mask, other=max_fill)
        keys_lo = _median_keys(vals_lo, KEY_BITS)
        keys_hi = _median_keys(vals_hi, KEY_BITS)
    tl.store(keybuf + pid * ROW_STRIDE + TAIL_BASE + cols, keys, mask=mask)
    if not PREORDERED and vals.dtype.is_floating():
        nan = mask & _median_is_nan(vals)
        local_first = tl.min(tl.where(nan, cols, BLOCK_P), axis=0)
        pack = tl.where(local_first < BLOCK_P, local_first + START, 2147483647)
        tl.store(chunk_nan + pid * BLOCK_R + NCHUNK - 1, pack)
    else:
        tl.store(chunk_nan + pid * BLOCK_R + NCHUNK - 1, 2147483647)
    lo = tl.min(keys_lo, axis=0)
    hi = tl.max(keys_hi, axis=0)
    tl.store(chunk_mins + pid * BLOCK_R + NCHUNK - 1, lo)
    tl.store(chunk_maxs + pid * BLOCK_R + NCHUNK - 1, hi)


@libentry()
@triton.jit
def median_count_partial_kernel(
    keys,
    mid,
    chunk_counts,
    N,
    NCHUNK,
    PARTIAL,
    BLOCK_R: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TAIL_BASE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # grid: (M); count keys <= mid within the leftover tail slice.
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_P)
    mask = cols < PARTIAL
    keys_v = tl.load(keys + pid * ROW_STRIDE + TAIL_BASE + cols, mask=mask, other=0)
    mid_v = tl.load(mid + pid)
    le = tl.sum((mask & (keys_v <= mid_v)).to(tl.int32), axis=0)
    tl.store(chunk_counts + pid * BLOCK_R + NCHUNK - 1, le)


@libentry()
@triton.jit
def median_select_partial_kernel(
    keybuf,
    sel_keys,
    chunk_first,
    N,
    NCHUNK,
    START,
    PARTIAL,
    BLOCK_R: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TAIL_BASE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # grid: (M); find the earliest matching key within the tail slice.
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_P)
    mask = cols < PARTIAL
    sel = tl.load(sel_keys + pid)
    keys_v = tl.load(keybuf + pid * ROW_STRIDE + TAIL_BASE + cols, mask=mask, other=0)
    km = mask & (keys_v == sel)
    first = tl.min(tl.where(km, cols, BLOCK_P), axis=0)
    pack = tl.where(first < BLOCK_P, first + START, 2147483647)
    tl.store(chunk_first + pid * BLOCK_R + NCHUNK - 1, pack)


@libentry()
@triton.jit
def median_row_reduce_kernel(
    chunk_data,
    out,
    NCHUNK,
    OTHER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
):
    # grid: M; MODE 0 = min, 1 = max, 2 = sum (int32).
    # OTHER is the masked-lane fill (max-key for min, 0 for max/sum).
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < NCHUNK
    v = tl.load(chunk_data + pid * BLOCK_N + cols, mask=mask, other=OTHER)
    if MODE == 0:
        r = tl.min(v, axis=0)
    elif MODE == 1:
        r = tl.max(v, axis=0)
    else:
        r = tl.sum(v.to(tl.int32), axis=0)
    tl.store(out + pid, r)


@libentry()
@triton.jit
def median_select_chunk_kernel(
    keybuf,
    sel_keys,
    chunk_first,
    N,
    NCHUNK,
    NCHUNK_FULL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # grid: M * NCHUNK_FULL; find the earliest matching key within one slice.
    # chunk_first is encoded as the GLOBAL index (chunk*CHUNK + local),
    # or 0x7FFFFFFF when the slice has no match.
    pid = ext.program_id(0)
    row = pid // NCHUNK_FULL
    chunk = pid % NCHUNK_FULL
    cols = tl.arange(0, CHUNK)
    slot = row * NCHUNK + chunk
    sel = tl.load(sel_keys + row)
    keys_v = tl.load(keybuf + slot * CHUNK + cols)
    km = keys_v == sel
    first = tl.min(tl.where(km, cols, CHUNK), axis=0)
    pack = tl.where(first < CHUNK, first + chunk * CHUNK, 2147483647)
    tl.store(chunk_first + row * BLOCK_R + chunk, pack)


@libentry()
@triton.jit
def median_set_scalar_kernel(buf, idx, val):
    tl.store(buf + idx, val)


@libentry()
@triton.jit
def median_merge_select_chunk_kernel(
    inp,
    chunk_first,
    chunk_nan,
    row_nan_first,
    out_values,
    out_indices,
    N,
    NCHUNK,
    USE_ROW_NAN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # grid: (M); pick the earliest matching key over all slices, and for
    # float rows the earliest NaN position.  chunk_first/chunk_nan hold
    # GLOBAL positions (chunk*CHUNK + local) or CHUNK when absent, so the
    # row-wide minimum directly yields the requested index.
    pid = ext.program_id(0)
    cc = tl.arange(0, BLOCK_N)
    mask = cc < NCHUNK
    cf = tl.load(chunk_first + pid * BLOCK_N + cc, mask=mask, other=2147483647)
    cn = tl.load(chunk_nan + pid * BLOCK_N + cc, mask=mask, other=2147483647)
    is_float: tl.constexpr = inp.dtype.element_ty.is_floating()
    has_nan = tl.max((cn < 2147483647).to(tl.int32), axis=0) != 0
    nan_best = tl.min(cn, axis=0)
    if USE_ROW_NAN:
        row_first = tl.load(row_nan_first + pid)
        has_nan = has_nan | (row_first >= 0)
        nan_best = tl.minimum(nan_best, tl.where(row_first >= 0, row_first, 0x7FFFFFFF))
    match_best = tl.min(cf, axis=0)
    best = tl.where(has_nan, nan_best, match_best)
    ridx = tl.minimum(best, N - 1)
    if is_float:
        rval = tl.load(inp + pid * N + ridx, mask=ridx < N, other=0.0)
        rval = tl.where(has_nan, float("nan"), rval)
    else:
        rval = tl.load(inp + pid * N + ridx, mask=ridx < N, other=0)
    tl.store(out_values + pid, rval)
    tl.store(out_indices + pid, ridx.to(tl.int64))


@libentry()
@triton.jit
def median_key_info_kernel(
    inp,
    keybuf,
    mins,
    maxs,
    nan_flags,
    nan_firsts,
    N: tl.constexpr,
    KEY_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREORDERED: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    if PREORDERED:
        keys = tl.load(inp + pid * N + cols, mask=mask, other=0)
        vals_lo = tl.load(inp + pid * N + cols, mask=mask, other=0xFFFFFFFFFFFFFFFF)
        vals_hi = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys_lo = vals_lo
        keys_hi = vals_hi
    else:
        dtype = inp.dtype.element_ty
        is_float: tl.constexpr = dtype.is_floating()
        if is_float:
            min_fill = float("-inf")
            max_fill = float("inf")
        else:
            min_fill = get_dtype_min(dtype)
            max_fill = get_dtype_max(dtype)
        vals = tl.load(inp + pid * N + cols, mask=mask, other=max_fill)
        keys = _median_keys(vals, KEY_BITS)
        vals_lo = tl.load(inp + pid * N + cols, mask=mask, other=min_fill)
        vals_hi = tl.load(inp + pid * N + cols, mask=mask, other=max_fill)
        keys_lo = _median_keys(vals_lo, KEY_BITS)
        keys_hi = _median_keys(vals_hi, KEY_BITS)
    # Only the first N lanes are written; pad lanes keep the all-ones-key
    # sentinel installed by the wrapper, so the binary-search count/select
    # stages never observe uninitialized torch.empty memory (masked tail
    # loads are unreliable on XPU and leak pad lanes into reductions).
    tl.store(keybuf + pid * BLOCK_N + cols, keys, mask=mask)
    if not PREORDERED and vals.dtype.is_floating():
        # Computed before the uint32 min/max reductions below: on XPU a
        # preceding tl.min/tl.max over keys_lo/keys_hi (uint32) corrupts
        # this subsequent int32 where-min result (miscompile), yielding
        # wrong NaN first-indices.
        nan = mask & _median_is_nan(vals)
        has_nan = tl.max(nan.to(tl.int32), axis=0) != 0
        first_nan = tl.min(tl.where(nan, cols, BLOCK_N), axis=0)
        tl.store(nan_flags + pid, has_nan.to(tl.int32))
        tl.store(nan_firsts + pid, first_nan)
    lo = tl.min(keys_lo, axis=0)
    hi = tl.max(keys_hi, axis=0)
    tl.store(mins + pid, lo)
    tl.store(maxs + pid, hi)


@libentry()
@triton.jit
def median_count_le_kernel(
    keys,
    lo,
    hi,
    counts,
    TARGET: tl.constexpr,
    N: tl.constexpr,
    KEY_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    keys_v = tl.load(keys + pid * BLOCK_N + cols)
    lo_v = tl.load(lo + pid)
    hi_v = tl.load(hi + pid)
    if KEY_BITS == 64:
        mid = lo_v + ((hi_v - lo_v) // 2)
    else:
        mid = ((lo_v.to(tl.int64) + hi_v.to(tl.int64)) // 2).to(tl.uint32)
    # Pad lanes hold the all-ones key sentinel, so they never satisfy
    # keys <= mid and can be summed unconditionally.
    le = tl.sum((keys_v <= mid).to(tl.int32), axis=0)
    go_left = le > TARGET
    active = lo_v < hi_v
    new_hi = tl.where(go_left & active, mid, hi_v)
    new_lo = tl.where(~go_left & active, mid + 1, lo_v)
    tl.store(counts + pid, le)
    tl.store(lo + pid, new_lo)
    tl.store(hi + pid, new_hi)


@libentry()
@triton.jit
def median_key_search_kernel(
    inp,
    keybuf,
    sel,
    nan_flags,
    nan_firsts,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_BITS: tl.constexpr,
    PREORDERED: tl.constexpr,
):
    # Fused power-of-two-row variant: the row is loaded exactly once (no
    # pad lanes) and the full binary search runs inside the kernel on the
    # register-held key vector, so the 32 count scans collapse to a single
    # global-read pass. Selection semantics (lower median, first index,
    # first-NaN precedence) match the split info/count/select kernels
    # exactly, so results are bit-identical to the legacy path.
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    pad = cols >= N
    is_float: tl.constexpr = inp.dtype.element_ty.is_floating()
    if PREORDERED:
        if KEY_BITS == 64:
            keys = tl.load(inp + pid * N + cols, mask=~pad, other=0xFFFFFFFFFFFFFFFF)
        else:
            keys = tl.load(inp + pid * N + cols, mask=~pad, other=0xFFFFFFFF)
    else:
        # Pad lanes use the dtype's maximum value so their keys sit at (or
        # above) the top of the real key range: they never affect min/lo,
        # mimic a duplicate of the max (count-inert below the median) and
        # never match the selected key during the select stage.
        if is_float:
            vals = tl.load(inp + pid * N + cols, mask=~pad, other=float("inf"))
        else:
            max_fill = get_dtype_max(inp.dtype.element_ty)
            vals = tl.load(inp + pid * N + cols, mask=~pad, other=max_fill)
        keys = _median_keys(vals, KEY_BITS)
    # Masked tail loads leak neighboring-row memory on XPU, so the pad
    # lanes are re-sanitized in-register to the all-ones key sentinel:
    # it never satisfies keys <= mid, never selects as a match, and is
    # never the min, so the binary search is inert to it.
    if KEY_BITS == 64:
        keys = tl.where(pad, 0xFFFFFFFFFFFFFFFF, keys)
    else:
        keys = tl.where(pad, 0xFFFFFFFF, keys)
    if (not PREORDERED) and is_float:
        # NaN bookkeeping must precede the uint32 min/max reductions (XPU
        # miscompile ordering constraint from median_key_info_kernel).
        # Pad lanes can leak NaN values from Neighboring rows, so the
        # NaN mask is restricted to the real lanes.
        nan = (~pad) & _median_is_nan(vals)
        has_nan = tl.max(nan.to(tl.int32), axis=0) != 0
        first_nan = tl.min(tl.where(nan, cols, BLOCK_N), axis=0)
        tl.store(nan_flags + pid, has_nan.to(tl.int32))
        tl.store(nan_firsts + pid, first_nan)
    lo = tl.min(keys, axis=0)
    hi = tl.max(keys, axis=0)
    target = (N - 1) // 2
    for _ in tl.range(0, KEY_BITS):
        if KEY_BITS == 64:
            mid = lo + ((hi - lo) // 2)
        else:
            mid = ((lo.to(tl.int64) + hi.to(tl.int64)) // 2).to(tl.uint32)
        le = tl.sum((keys <= mid).to(tl.int32), axis=0)
        go_left = le > target
        active = lo < hi
        hi = tl.where(go_left & active, mid, hi)
        lo = tl.where(~go_left & active, mid + 1, lo)
    tl.store(keybuf + pid * BLOCK_N + cols, keys)
    tl.store(sel + pid, lo)


@libentry()
@triton.jit
def median_select_kernel(
    inp,
    keybuf,
    sel_keys,
    nan_flags,
    nan_firsts,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    sel = tl.load(sel_keys + pid)
    keys_v = tl.load(keybuf + pid * BLOCK_N + cols, mask=mask, other=0)
    km = keys_v == sel
    first = tl.min(tl.where(km, cols, BLOCK_N), axis=0)
    has_nan = tl.load(nan_flags + pid) != 0
    first_nan = tl.load(nan_firsts + pid)
    ridx = tl.minimum(tl.where(has_nan, first_nan, first), N - 1)
    is_float: tl.constexpr = inp.dtype.element_ty.is_floating()
    if is_float:
        rval = tl.load(inp + pid * N + ridx, mask=ridx < N, other=0.0)
        rval = tl.where(has_nan, float("nan"), rval)
    else:
        rval = tl.load(inp + pid * N + ridx, mask=ridx < N, other=0)
    tl.store(out_values + pid, rval)
    tl.store(out_indices + pid, ridx.to(tl.int64))


@libentry()
@triton.jit
def median_bool_row_kernel(
    inp,
    out_values,
    out_indices,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    true_count = tl.full((), 0, dtype=tl.int32)
    first_false = tl.full((), 2147483647, dtype=tl.int32)
    first_true = tl.full((), 2147483647, dtype=tl.int32)

    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=False)
        true_count += tl.sum((vals & mask).to(tl.int32), axis=0)
        first_false = tl.minimum(
            first_false,
            tl.min(tl.where(mask & ~vals, cols, 2147483647), axis=0),
        )
        first_true = tl.minimum(
            first_true,
            tl.min(tl.where(mask & vals, cols, 2147483647), axis=0),
        )

    false_count = N - true_count
    rank = (N - 1) // 2
    take_true = rank >= false_count
    median_val = take_true
    median_idx = tl.where(take_true, first_true, first_false)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx.to(tl.int64))


def _check_supported_dtype(inp):
    if inp.dtype is torch.complex64 or inp.dtype is torch.complex128:
        raise NotImplementedError('"median_out_impl" not implemented for complex')


def _normalize_dim(dim, ndim):
    if ndim == 0:
        if dim in (0, -1):
            return 0
    elif -ndim <= dim < ndim:
        return dim % ndim
    raise IndexError(
        f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
    )


def _pad_meta(values, fill):
    if len(values) > MAX_NDIM:
        raise NotImplementedError(
            f"median supports input rank <= {MAX_NDIM} on Kunlunxin"
        )
    return tuple(values) + (fill,) * (MAX_NDIM - len(values))


def _empty_flat_result(inp):
    result = torch.empty((), dtype=inp.dtype, device=inp.device)
    if inp.dtype.is_complex:
        result.real.fill_(float("nan"))
        result.imag.fill_(0.0)
        return result
    if inp.dtype.is_floating_point:
        result.fill_(float("nan"))
    elif inp.dtype == torch.bool:
        result.fill_(True)
    elif inp.dtype in (torch.int32, torch.int64):
        result.fill_(torch.iinfo(inp.dtype).min)
    else:
        result.fill_(0)
    return result


def _reduction_rows(inp, dim, M, N):
    if dim == inp.ndim - 1:
        return inp.reshape(M, N)
    return torch.movedim(inp, dim, -1).contiguous().reshape(M, N)


def _key_bits(dtype):
    if dtype in (torch.float64, torch.int64):
        return 64
    return 32


def _order_keys64(t):
    bits = t.view(torch.int64)
    key = bits ^ (0x8000000000000000 | (bits >> 63))
    return key.to(torch.uint64)


def _median_key_select(rows, N):
    M = rows.shape[0]
    key_bits = _key_bits(rows.dtype)
    block_n = triton.next_power_of_2(N)
    if block_n > KEY_BLOCK_LIMIT:
        return _median_key_select_chunked(rows, N, key_bits)
    if key_bits == 64 and block_n > 16384:
        # The fused in-kernel 64-iteration binary-search loop miscompiles on
        # XPU once the row block reaches 32768 lanes: tl.sum inside the loop
        # then counts every lane (le > target always), so the search converges
        # to the row minimum for int64 keys (always top-half) and for float64
        # rows whose order-keys sit in the top half (negative-heavy data).
        # The chunked path runs one single-shot count kernel per host-side
        # bisection step, which is exact at any width, so route 64-bit-key
        # rows wider than 16384 there (tail-only single chunk).
        return _median_key_select_chunked(rows, N, key_bits)
    block_n = max(64, block_n)
    key_dtype = torch.uint64 if key_bits == 64 else torch.uint32
    # Pre-fill pad lanes with the all-ones key sentinel (deterministic);
    # the kernel only overwrites the first N lanes.
    keybuf = torch.full(
        (M, block_n),
        -1,
        dtype=torch.int64 if key_bits == 64 else torch.int32,
        device=rows.device,
    ).view(key_dtype)
    sel = torch.empty((M,), dtype=key_dtype, device=rows.device)
    out_values = torch.empty((M,), dtype=rows.dtype, device=rows.device)
    out_indices = torch.empty((M,), dtype=torch.long, device=rows.device)
    nan_flags = torch.empty((M,), dtype=torch.int32, device=rows.device)
    nan_firsts = torch.empty((M,), dtype=torch.int32, device=rows.device)

    preordered = rows.dtype in (torch.float64, torch.int64)
    if rows.dtype == torch.bool:
        work = rows.to(torch.uint8)
    elif rows.dtype == torch.int64:
        # Two's complement: a single sign-bit flip gives a strictly
        # monotone key order (unlike _order_keys64, which is the
        # IEEE-float transform and inverts negatives).
        sign_bit = torch.tensor(-(1 << 63), dtype=torch.int64, device=rows.device)
        work = (rows.view(torch.int64) ^ sign_bit).view(torch.uint64)
    elif preordered:
        work = _order_keys64(rows)
    elif rows.dtype in (torch.float16, torch.bfloat16):
        work = rows.to(torch.float32)
    elif rows.dtype in (torch.int8, torch.uint8, torch.int16):
        work = rows.to(torch.int32)
    else:
        work = rows
    if preordered and rows.dtype == torch.float64:
        nanf = rows.isnan()
        nan_flags = nanf.any(dim=1).to(torch.int32).contiguous()
        nan_firsts = nanf.to(torch.int64).argmax(dim=1).to(torch.int32).contiguous()
    else:
        # The select kernel reads these for every row; leave them
        # deterministic (0 = "no NaN") instead of torch.empty garbage.
        nan_flags.zero_()
        nan_firsts.zero_()

    # Fused path: the whole binary search runs in one kernel over the
    # register-held key vector (single global-read pass), replacing the
    # legacy key_info + 32 count launches. Selection semantics (lower
    # median, first index, first-NaN precedence) are identical, so select
    # results are bit-identical to the legacy loop.
    with torch_device_fn.device(work.device):
        median_key_search_kernel[(M,)](
            work,
            keybuf,
            sel,
            nan_flags,
            nan_firsts,
            N,
            block_n,
            key_bits,
            preordered,
            num_warps=4,
            num_stages=1,
            buffer_size_limit=2048,
        )
        median_select_kernel[(M,)](
            rows,
            keybuf,
            sel,
            nan_flags,
            nan_firsts,
            out_values,
            out_indices,
            N,
            block_n,
            num_warps=4,
            num_stages=1,
            buffer_size_limit=2048,
        )
    return out_values, out_indices


def _median_key_select_chunked(rows, N, key_bits):
    """Key-sort select for rows wider than KEY_BLOCK_LIMIT lanes.

    The row is split into full CHUNK-sized slices plus an unpadded tail
    slice; the binary search counts per slice in parallel and per-row
    totals are reduced over the slice counts (host-loop binary search).
    """
    M = rows.shape[0]
    CHUNK = KEY_BLOCK_LIMIT
    nfull = N // CHUNK
    partial = N - nfull * CHUNK
    nall = nfull + (1 if partial else 0)
    nchunks = nall
    if nchunks > KEY_BLOCK_LIMIT:
        raise NotImplementedError(f"median reduction width {N} exceeds Kunlunxin limit")
    tail_start = nfull * CHUNK
    key_dtype = torch.uint64 if key_bits == 64 else torch.uint32
    keybuf = torch.full(
        (M * nall, CHUNK),
        -1,
        dtype=torch.int64 if key_bits == 64 else torch.int32,
        device=rows.device,
    ).view(key_dtype)
    reduce_block = triton.next_power_of_2(nall)
    chunk_mins = torch.empty((M * reduce_block,), dtype=key_dtype, device=rows.device)
    chunk_maxs = torch.empty((M * reduce_block,), dtype=key_dtype, device=rows.device)
    chunk_nan = torch.full((M * reduce_block,), 2147483647, dtype=torch.int32, device=rows.device)
    chunk_counts = torch.empty((M * reduce_block,), dtype=torch.int32, device=rows.device)
    counts = torch.empty((M,), dtype=torch.int32, device=rows.device)
    row_nan_first = torch.full((M,), -1, dtype=torch.int32, device=rows.device)
    lo = torch.empty((M,), dtype=key_dtype, device=rows.device)
    hi = torch.empty((M,), dtype=key_dtype, device=rows.device)
    mid = torch.empty((M,), dtype=key_dtype, device=rows.device)

    preordered = rows.dtype in (torch.float64, torch.int64)
    if rows.dtype == torch.bool:
        work = rows.to(torch.uint8)
    elif rows.dtype == torch.int64:
        sign_bit = torch.tensor(-(1 << 63), dtype=torch.int64, device=rows.device)
        work = (rows.view(torch.int64) ^ sign_bit).view(torch.uint64)
    elif preordered:
        work = _order_keys64(rows)
    elif rows.dtype in (torch.float16, torch.bfloat16):
        work = rows.to(torch.float32)
    elif rows.dtype == torch.int8 or rows.dtype == torch.uint8 or rows.dtype == torch.int16:
        work = rows.to(torch.int32)
    else:
        work = rows
    if preordered and rows.dtype == torch.float64:
        nanf = rows.isnan()
        nan_firsts = nanf.to(torch.int64).argmax(dim=1)
        row_nan_first = torch.where(
            nanf.any(dim=1), nan_firsts.to(torch.int32), torch.full((M,), -1, dtype=torch.int32, device=rows.device)
        )

    tail_blk = triton.next_power_of_2(partial) if partial else 1
    row_stride = nall * CHUNK
    tail_base = (nall - 1) * CHUNK
    with torch_device_fn.device(work.device):
        nan_kw = dict(num_warps=4, num_stages=1, buffer_size_limit=2048)
        if nfull:
            median_key_info_chunk_kernel[(M * nfull,)](
                work,
                keybuf,
                chunk_mins,
                chunk_maxs,
                chunk_nan,
                N,
                nchunks,
                nfull,
                reduce_block,
                CHUNK,
                key_bits,
                preordered,
                **nan_kw,
            )
        if partial:
            median_key_info_partial_kernel[(M,)](
                work,
                keybuf,
                chunk_mins,
                chunk_maxs,
                chunk_nan,
                N,
                nchunks,
                tail_start,
                partial,
                reduce_block,
                row_stride,
                tail_base,
                tail_blk,
                key_bits,
                preordered,
                **nan_kw,
            )
        max_key = (1 << key_bits) - 1
        median_row_reduce_kernel[(M,)](
            chunk_mins, lo, nchunks, max_key, reduce_block, 0, **nan_kw
        )
        median_row_reduce_kernel[(M,)](
            chunk_maxs, hi, nchunks, 0, reduce_block, 1, **nan_kw
        )
        target = (N - 1) // 2
        # Host-loop binary search.  Bulk D2H copies are rejected by the
        # Kunlunxin to/copy overrides, so the small per-row state is read
        # with scalar .item() (bitcast via f32 for uint32) and the mids are
        # pushed back with a tiny scalar-store kernel.
        def _u32_of(t, i):
            if key_bits == 64:
                return int(t.view(torch.int64)[i].item()) & 0xFFFFFFFFFFFFFFFF
            return int(np.asarray(np.float32(t.view(torch.float32)[i].item())).view(np.uint32))
        lo_h = [_u32_of(lo, r) for r in range(M)]
        hi_h = [_u32_of(hi, r) for r in range(M)]
        for _ in range(key_bits + 6):
            mid_h = [(l + h) // 2 for l, h in zip(lo_h, hi_h)]
            if key_bits == 64:
                mid_h = [m & 0xFFFFFFFFFFFFFFFF for m in mid_h]
            for r in range(M):
                median_set_scalar_kernel[(1,)](mid, r, mid_h[r], **nan_kw)
            if nfull:
                median_count_chunk_kernel[(M * nfull,)](
                    keybuf, mid, chunk_counts, N, nchunks, nfull, reduce_block, CHUNK, **nan_kw
                )
            if partial:
                median_count_partial_kernel[(M,)](
                    keybuf, mid, chunk_counts, N, nchunks, partial,
                    reduce_block, row_stride, tail_base, tail_blk, **nan_kw
                )
            median_row_reduce_kernel[(M,)](
                chunk_counts, counts, nchunks, 0, reduce_block, 2, **nan_kw
            )
            for r in range(M):
                if int(counts[r].item()) > target:
                    hi_h[r] = mid_h[r]
                else:
                    lo_h[r] = mid_h[r] + 1
        sel_keys = lo
        for r in range(M):
            median_set_scalar_kernel[(1,)](sel_keys, r, lo_h[r], **nan_kw)
        out_values = torch.empty((M,), dtype=rows.dtype, device=rows.device)
        out_indices = torch.empty((M,), dtype=torch.long, device=rows.device)
        chunk_first = torch.empty((M * reduce_block,), dtype=torch.int32, device=rows.device)
        if nfull:
            median_select_chunk_kernel[(M * nfull,)](
                keybuf, sel_keys, chunk_first, N, nchunks, nfull, reduce_block, CHUNK, **nan_kw
            )
        if partial:
            median_select_partial_kernel[(M,)](
                keybuf, sel_keys, chunk_first, N, nchunks, tail_start, partial,
                reduce_block, row_stride, tail_base, tail_blk, **nan_kw
            )
        use_row_nan = rows.dtype == torch.float64
        median_merge_select_chunk_kernel[(M,)](
            rows,
            chunk_first,
            chunk_nan,
            row_nan_first,
            out_values,
            out_indices,
            N,
            nchunks,
            use_row_nan,
            reduce_block,
            CHUNK,
            **nan_kw,
        )
    return out_values, out_indices


def _median_dim_impl(inp, dim, keepdim, out=None):
    dim = _normalize_dim(dim, inp.ndim)

    if inp.ndim == 0:
        if out is None:
            values = inp.clone()
            indices = torch.zeros((), dtype=torch.long, device=inp.device)
        else:
            values, indices = out
            values.copy_(inp)
            indices.zero_()
        return MedianResult(values=values, indices=indices)

    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape

    if N == 0:
        if M != 0:
            raise IndexError(
                f"median(): Expected reduction dim {dim} to have non-zero size."
            )
        if out is None:
            values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
            indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
            if not keepdim:
                values = torch.squeeze(values, dim)
                indices = torch.squeeze(indices, dim)
        else:
            values, indices = out
        return MedianResult(values=values, indices=indices)

    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        # Native ATen semantics: out containers are resized to the result
        # shape (callers may pass e.g. a (1,) buffer for a (7,) result).
        values, indices = out
        if tuple(values.shape) != tuple(compute_shape):
            values = values.resize_(compute_shape)
        if tuple(indices.shape) != tuple(compute_shape):
            indices = indices.resize_(compute_shape)

    if M == 0:
        if out is None and not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return MedianResult(values=values, indices=indices)

    with torch_device_fn.device(inp.device):
        # bool is handled through the key-select path too (0/1 keys); the
        # dedicated loop-based bool kernel is unreliable for N > 1024.
        rows = _reduction_rows(inp, dim, M, N)
        out_values, out_indices = _median_key_select(rows, N)
        # Native copy engine write-backs (handle broadcasting and strided out
        # containers; avoid nesting the registered copy_ override).
        torch.ops.aten._copy_from(out_values, values, False)
        torch.ops.aten._copy_from(out_indices, indices, False)

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return MedianResult(values=values, indices=indices)


def _median_flat_impl(inp, out=None):
    if inp.numel() == 0:
        result = _empty_flat_result(inp)
        if out is not None:
            # Write back through the native copy engine (never overridden by
            # gems), not Tensor.copy_, to avoid nesting the registered copy_.
            torch.ops.aten._copy_from(result, out, False)
            return out
        return result

    flat = inp.reshape(-1)
    if out is None:
        return _median_dim_impl(flat, 0, False).values

    indices = torch.empty((), dtype=torch.long, device=inp.device)
    _median_dim_impl(flat, 0, False, out=(out, indices))
    return out


def median(inp):
    logger.debug("GEMS_KUNLUNXIN MEDIAN")
    # Complex is only acceptable for the empty-input case (torch returns an
    # empty complex result); non-empty complex raises NotImplementedError.
    if inp.numel() != 0:
        _check_supported_dtype(inp)
    return _median_flat_impl(inp)


def median_out(inp, *, out):
    logger.debug("GEMS_KUNLUNXIN MEDIAN_OUT")
    if inp.numel() != 0:
        _check_supported_dtype(inp)
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"median(): Expected out tensor to have dtype {inp.dtype}, but got {out.dtype}"
        )
    if out.device != inp.device:
        raise RuntimeError(
            "median(): Expected out tensor to be on the same device as the input"
        )
    return _median_flat_impl(inp, out=out)


def _resolve_dim_name(inp, dim):
    if isinstance(dim, str):
        try:
            return inp.names.index(dim), True
        except ValueError:
            raise RuntimeError(f"median(): dim '{dim}' not found in input names")
    return dim, False


def median_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MEDIAN_DIM")
    _check_supported_dtype(inp)
    dim_res, dim_was_name = _resolve_dim_name(inp, dim)
    if dim_was_name:
        # Named-tensor ops (movedim etc.) are unsupported on this backend;
        # compute on the unnamed tensor and re-attach names on the result.
        names = list(inp.names)
        result = _median_dim_impl(inp.rename(None), dim_res, keepdim)
        if keepdim:
            out_names = tuple(names)
        else:
            out_names = tuple(names[:dim_res] + names[dim_res + 1 :])
        result = MedianResult(
            values=result.values.rename(*out_names),
            indices=result.indices.rename(*out_names),
        )
        return result
    return _median_dim_impl(inp, dim_res, keepdim)


def median_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_KUNLUNXIN MEDIAN_DIM_VALUES")
    _check_supported_dtype(inp)
    if values.dtype != inp.dtype:
        raise RuntimeError(
            f"median(): Expected 'values' tensor to have dtype {inp.dtype}, but got {values.dtype}"
        )
    if indices.dtype != torch.long:
        raise RuntimeError(
            f"median(): Expected 'indices' tensor to have dtype torch.int64, but got {indices.dtype}"
        )
    if values.device != inp.device or indices.device != inp.device:
        raise RuntimeError(
            "median(): Expected 'values' and 'indices' tensors to be on the same device as the input"
        )
    dim_res, _ = _resolve_dim_name(inp, dim)
    return _median_dim_impl(inp, dim_res, keepdim, out=(values, indices))
