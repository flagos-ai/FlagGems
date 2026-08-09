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
HIST_BLOCK_N = 4096
HIST_BLOCK_LIMIT = 8192
BOOL_BLOCK_N = 1024
MAX_NDIM = 8
HIST_BINS = 256


@triton.jit
def _median_is_nan(vals):
    vals_fp32 = vals.to(tl.float32)
    return vals_fp32 != vals_fp32


@triton.jit
def _median_keys(vals, KEY_BITS: tl.constexpr):
    # Order-preserving unsigned key of the same width as KEY_BITS.
    if KEY_BITS == 64:
        return convert_to_uint_preverse_order(vals, False)
    if vals.dtype.is_floating():
        w = vals.to(tl.float32)
    else:
        w = vals
    k = convert_to_uint_preverse_order(w, False)
    return k.to(tl.uint32)


@triton.jit
def _median_hist_keys(vals, KEY_BITS: tl.constexpr):
    if KEY_BITS == 64:
        return convert_to_uint_preverse_order(vals, False)
    if vals.dtype.is_floating():
        w = vals.to(tl.float32)
    else:
        w = vals
    k = convert_to_uint_preverse_order(w, False)
    return k.to(tl.uint32)


@triton.jit
def _median_has_nan_norm(vals):
    return _median_is_nan(vals)


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
    vals = tl.load(inp + base + offsets * STRIDE_DIM, mask=mask, other=max_value)

    if dtype.is_floating():
        nan_mask = mask & _median_is_nan(vals)
        has_nan = tl.max(nan_mask.to(tl.int32), axis=0) != 0
        first_nan_idx = tl.min(tl.where(nan_mask, offsets, BLOCK_N), axis=0)
    else:
        has_nan = False
        first_nan_idx = tl.full((), 0, dtype=tl.int32)

    median_rank = (N - 1) // 2

    active = mask
    median_val = tl.full((), fallback_value, dtype=vals.dtype)
    median_idx = tl.full((), 0, dtype=tl.int32)
    for select_iter in tl.static_range(0, BLOCK_N):
        select_vals = tl.where(active, vals, max_value)
        cur_val = tl.min(select_vals, axis=0)
        cur_idx = tl.min(tl.where(active & (vals == cur_val), offsets, BLOCK_N), axis=0)
        take = select_iter == median_rank
        median_val = tl.where(take, cur_val, median_val)
        median_idx = tl.where(take, cur_idx, median_idx)
        active = active & (offsets != cur_idx)

    if dtype.is_floating():
        median_val = tl.where(has_nan, float("nan"), median_val)
        median_idx = tl.where(has_nan, first_nan_idx, median_idx)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx)


@libentry()
@triton.jit
def median_row_hist_select_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    KEY_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
    keys = _median_hist_keys(vals, KEY_BITS)

    dtype = inp.dtype.element_ty
    is_float: tl.constexpr = dtype.is_floating()
    if is_float:
        nan_mask = mask & _median_is_nan(vals)
        has_nan = tl.max(nan_mask.to(tl.int32), axis=0) != 0
        first_nan_idx = tl.min(tl.where(nan_mask, cols, BLOCK_N), axis=0)
    else:
        has_nan = False
        first_nan_idx = tl.full((), 0, dtype=tl.int32)

    rank = (N - 1) // 2
    PASSES: tl.constexpr = KEY_BITS // 8
    sel_key = tl.full((), 0, dtype=vals.dtype)
    if KEY_BITS == 64:
        sel_key = tl.full((), 0, dtype=tl.uint64)
    else:
        sel_key = tl.full((), 0, dtype=tl.uint32)

    bin_cols = tl.arange(0, 256)
    for p in tl.static_range(0, PASSES):
        shift = KEY_BITS - 8 * (p + 1)
        win = ((keys >> shift) & 255).to(tl.int32)
        if p == 0:
            keep = mask
        else:
            keep = mask & ((keys >> (shift + 8)) == sel_key)
        h = tl.histogram(win, 256, mask=keep)
        cum = tl.cumsum(h, axis=0)
        target = rank + 1
        digit = tl.sum((cum < target).to(tl.int32), axis=0)
        below = tl.sum(tl.where(bin_cols < digit, h, 0), axis=0)
        sel_key = (sel_key << 8) | digit.to(sel_key.dtype)
        rank = rank - below

    sel_key32 = sel_key.to(tl.uint32)
    key_match = mask & (keys.to(tl.uint64).to(tl.uint32) == (sel_key32 if KEY_BITS != 64 else tl.full((), 0, dtype=tl.uint32)))
    # find first index with key == sel_key
    first_match = tl.min(tl.where(key_match, cols, BLOCK_N), axis=0)
    ...