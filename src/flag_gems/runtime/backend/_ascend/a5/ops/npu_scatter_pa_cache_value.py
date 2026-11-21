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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

# ---- variant switches (single-factor experiments change exactly one) --------
_HEAD = 512
_TAIL_BLOCK = 512
_HEAD_UNMASKED = False
_GRID_2D = True


@triton.jit
def _scatter_headtail_kernel(
    key_ptr,
    slot_ptr,
    cache_ptr,
    W: tl.constexpr,
    stride_cache: tl.constexpr,
    pages: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
    HEAD_UNMASKED: tl.constexpr,
    TWO_D: tl.constexpr,
):
    if TWO_D:
        row = tl.program_id(0)
        part = tl.program_id(1)
    else:
        pid = tl.program_id(0)
        row = pid // 2
        part = pid - row * 2
    row_off = row.to(tl.int64) * W

    if part == 0:
        lane_h = tl.arange(0, HEAD)
        if HEAD_UNMASKED:
            # Whole HEAD window is live: no data-dependent lane mask at all.
            vals_h = tl.load(key_ptr + row_off + lane_h)
            slot_h = tl.load(slot_ptr + row)
            ok_h = (slot_h >= 0) & (slot_h < pages)
            if ok_h:
                tl.store(
                    cache_ptr + slot_h.to(tl.int64) * stride_cache + lane_h,
                    vals_h,
                )
        else:
            mh = lane_h < W
            vals_h = tl.load(key_ptr + row_off + lane_h, mask=mh)
            slot_h = tl.load(slot_ptr + row)
            ok_h = (slot_h >= 0) & (slot_h < pages)
            tl.store(
                cache_ptr + slot_h.to(tl.int64) * stride_cache + lane_h,
                vals_h,
                mask=mh & ok_h,
            )
    else:
        lane_t = tl.arange(0, TAIL_BLOCK)
        tail_off = HEAD + lane_t
        mt = tail_off < W
        vals_t = tl.load(key_ptr + row_off + tail_off, mask=mt)
        slot_t = tl.load(slot_ptr + row)
        ok_t = (slot_t >= 0) & (slot_t < pages)
        tl.store(
            cache_ptr + slot_t.to(tl.int64) * stride_cache + tail_off,
            vals_t,
            mask=mt & ok_t,
        )


@triton.jit
def _scatter_pa_cache_kernel(
    key_ptr,
    slot_ptr,
    cache_ptr,
    W,
    stride_cache,
    pages,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    if NCHUNK == 1:
        row = pid
        chunk = 0
    else:
        row = pid // NCHUNK
        chunk = pid - row * NCHUNK

    # Issue the vector row fetch first: it does not depend on slot_mapping, so
    # the MTE2 row load can be in flight while the scalar slot value arrives.
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    m = offs < W
    vals = tl.load(key_ptr + row.to(tl.int64) * W + offs, mask=m)

    slot = tl.load(slot_ptr + row)
    ok = (slot >= 0) & (slot < pages)
    tl.store(cache_ptr + slot.to(tl.int64) * stride_cache + offs, vals, mask=m & ok)


# Preferred integer word width (bytes) for the byte-exact copy. All packings
# move exactly the same bytes per record; this constant only selects one.
_WORD_PREFERENCE = (1, 8, 4)


def _pick_wordsize(w_bytes, stride_bytes):
    """First preferred integer word (bytes) dividing both row widths."""
    for ws in _WORD_PREFERENCE:
        if w_bytes % ws == 0 and stride_bytes % ws == 0:
            return ws
    return 1


def _word_view(t, wordsize):
    """Zero-copy integer reinterpretation of a contiguous byte tensor."""
    if wordsize == 8:
        return t.view(torch.int64)
    if wordsize == 4:
        return t.view(torch.int32)
    return t.view(torch.uint8)


def _launch(key, slot_mapping, key_cache):
    n_rows = key.shape[0]
    if n_rows == 0 or key.numel() == 0:
        return key_cache

    w_bytes = key.numel() // n_rows
    stride_bytes = key_cache.numel() // key_cache.shape[0]
    pages = key_cache.shape[0]

    if not key.is_contiguous():
        key = key.contiguous()
    if not key_cache.is_contiguous():
        key_cache = key_cache.contiguous()
    if not slot_mapping.is_contiguous():
        slot_mapping = slot_mapping.contiguous()

    wordsize = _pick_wordsize(w_bytes, stride_bytes)
    key_w = _word_view(key, wordsize)
    cache_w = _word_view(key_cache, wordsize)
    W = w_bytes // wordsize
    stride_cache = stride_bytes // wordsize

    head = _HEAD // wordsize
    tail_block = _TAIL_BLOCK // wordsize
    if head < W <= head + tail_block:
        grid = (n_rows, 2) if _GRID_2D else (n_rows * 2,)
        _scatter_headtail_kernel[grid](
            key_w,
            slot_mapping,
            cache_w,
            W,
            stride_cache,
            pages,
            HEAD=head,
            TAIL_BLOCK=tail_block,
            HEAD_UNMASKED=_HEAD_UNMASKED,
            TWO_D=_GRID_2D,
            num_warps=2,
        )
        return key_cache

    BLOCK = min(triton.next_power_of_2(W), 1024)
    if BLOCK > 64:
        BLOCK //= 2
    nchunk = (W + BLOCK - 1) // BLOCK

    _scatter_pa_cache_kernel[(n_rows * nchunk,)](
        key_w,
        slot_mapping,
        cache_w,
        W,
        stride_cache,
        pages,
        BLOCK=BLOCK,
        NCHUNK=nchunk,
        num_warps=2,
    )
    return key_cache


def npu_scatter_pa_cache_value(key, slot_mapping, *, key_cache):
    """Scatter byte-exact 656-byte FP8 KV rows into a preallocated paged cache."""
    if (
        key.ndim != 3
        or key.shape[1:] != (1, 656)
        or key_cache.ndim != 4
        or key_cache.shape[1:] != (1, 1, 656)
        or key.dtype != torch.float8_e4m3fn
        or key_cache.dtype != key.dtype
        or slot_mapping.shape != (key.shape[0],)
        or slot_mapping.dtype != torch.int32
        or not all(t.is_contiguous() for t in (key, slot_mapping, key_cache))
        or key.storage_offset() % 8
        or key_cache.storage_offset() % 8
    ):
        raise ValueError(
            "expected aligned contiguous 656-byte cache rows and one int32 slot per token"
        )
    if (
        key.device.type != "npu"
        or slot_mapping.device != key.device
        or key_cache.device != key.device
    ):
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(key.device):
        return _launch(key, slot_mapping, key_cache=key_cache)
