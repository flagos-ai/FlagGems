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

_RADIX_THRESHOLD = 2048  # use radix selection for n > this, bitonic sort below


@triton.jit
def _median_sort_kernel(
    inp,
    out,
    n_elem,
    k,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    SORT32: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(inp + offs, mask=mask, other=0)
    if IS_FLOAT:
        if SORT32:
            xf = x.to(tl.float32)
        else:
            xf = x
        # torch.median propagates NaN: if any element is NaN, result is NaN.
        has_nan = tl.sum((xf != xf).to(tl.int32), axis=0) > 0
        xf = tl.where(mask, xf, float("inf"))
        xs = tl.sort(xf, dim=0)
        med = tl.sum(tl.where(offs == k, xs, 0.0), axis=0)
        med = tl.where(has_nan, float("nan"), med)
        tl.store(out, med)
    else:
        if SORT32:
            xi = x.to(tl.int32)
            pad = 2147483647
        else:
            xi = x
            pad = 9223372036854775807
        xi = tl.where(mask, xi, pad)
        xs = tl.sort(xi, dim=0)
        med = tl.sum(tl.where(offs == k, xs, 0), axis=0)
        tl.store(out, med)


@triton.jit
def _median_radix_kernel(
    inp,
    out,
    n_elem,
    k,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    NBYTES: tl.constexpr,
):  # 2, 4, or 8
    offs = tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(inp + offs, mask=mask, other=0)
    if IS_FLOAT:
        if NBYTES == 2:
            # fp16/bf16: native 16-bit order-preserving key
            xf = x.to(tl.float32)
            bits = x.to(tl.uint16, bitcast=True)
            sign = (bits >> 15) != 0
            key = tl.where(sign, (~bits).to(tl.uint16), (bits | 0x8000).to(tl.uint16))
        elif NBYTES == 4:
            xf = x.to(tl.float32)
            bits = xf.to(tl.uint32, bitcast=True)
            sign = (bits >> 31) != 0
            key = tl.where(sign, ~bits, bits | 0x80000000)
        else:
            xf = x
            bits = xf.to(tl.uint64, bitcast=True)
            sign = (bits >> 63) != 0
            key = tl.where(sign, ~bits, bits | 0x8000000000000000)
        nan_mask = xf != xf
        has_nan = tl.sum(nan_mask.to(tl.int32), axis=0) > 0
        valid = mask & (~nan_mask)
    else:
        if NBYTES == 2:
            xi = x.to(tl.int32)
            bits = x.to(tl.uint16, bitcast=True)
            key = (bits ^ 0x8000).to(tl.uint16)
        elif NBYTES == 4:
            xi = x.to(tl.int32)
            bits = xi.to(tl.uint32, bitcast=True)
            key = bits ^ 0x80000000
        else:
            xi = x
            bits = xi.to(tl.uint64, bitcast=True)
            key = bits ^ 0x8000000000000000
        valid = mask
    # radix-select the k-th smallest key, one byte (MSB first) per round.
    rank = k
    prefix_ok = valid
    sel_key = tl.zeros((), dtype=tl.int64)
    bins = tl.arange(0, 256)
    for b in tl.static_range(NBYTES):
        shift = (NBYTES - 1 - b) * 8
        byte = ((key >> shift) & 0xFF).to(tl.int32)
        hb = tl.histogram(tl.where(prefix_ok, byte, 255), 256)
        cum = tl.cumsum(hb, axis=0)
        gt = cum > rank
        v = tl.min(tl.where(gt, bins, 256), axis=0)
        prev = tl.sum(tl.where(bins < v, hb, 0), axis=0)
        rank = rank - prev
        sel_key = sel_key | (v.to(tl.int64) << shift)
        prefix_ok = prefix_ok & (byte == v)
    if NBYTES == 2:
        sk = sel_key.to(tl.uint16)
    elif NBYTES == 8:
        sk = sel_key.to(tl.uint64)
    else:
        sk = sel_key.to(tl.uint32)
    match = valid & (key == sk)
    if IS_FLOAT:
        val = tl.max(tl.where(match, xf, float("-inf")), axis=0)
        val = tl.where(has_nan, float("nan"), val)
        tl.store(out, val)
    else:
        if NBYTES == 8:
            val = tl.max(tl.where(match, xi, -9223372036854775808), axis=0)
        else:
            val = tl.max(tl.where(match, xi, -2147483648), axis=0)
        tl.store(out, val)


@triton.jit
def _fill_kernel(out, value):
    tl.store(out, value)


@triton.jit
def _fill_nan_words_kernel(out32, n_words: tl.constexpr):
    offs = tl.arange(0, 4)
    v = tl.where(offs == 0, float("nan"), 0.0)
    tl.store(out32 + offs, v, mask=offs < n_words)


@triton.jit
def _bool_count_kernel(inp, counts, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(inp + offs, mask=mask, other=0)
    cnt = tl.sum(x.to(tl.int32), axis=0)
    tl.store(counts + pid, cnt)


@triton.jit
def _bool_finish_kernel(
    counts, out, n_elem, k, NBLOCKS: tl.constexpr, NP2: tl.constexpr
):
    offs = tl.arange(0, NP2)
    t = tl.sum(tl.load(counts + offs, mask=offs < NBLOCKS, other=0), axis=0)
    # sorted ascending: False..False, True..True ; median at index k is True
    # iff count_false <= k  <=>  count_true >= n - k
    med = t >= (n_elem - k)
    tl.store(out, med)


def _nbytes(dtype):
    if dtype.is_floating_point:
        return 2 if dtype.itemsize == 2 else (8 if dtype.itemsize > 4 else 4)
    return 2 if dtype.itemsize == 2 else (8 if dtype.itemsize > 4 else 4)


def median(inp):
    n = inp.numel()
    dtype = inp.dtype
    out = torch.empty((), dtype=dtype, device=inp.device)
    if n == 0:
        # Match torch.median empty-input semantics per dtype.
        if dtype.is_complex:
            n_words = 4 if dtype.itemsize == 16 else 2
            _fill_nan_words_kernel[(1,)](
                out.reshape(1).view(torch.float32), n_words=n_words, num_warps=1
            )
            return out
        if dtype.is_floating_point:
            _fill_kernel[(1,)](out, float("nan"), num_warps=1)
            return out
        if dtype is torch.bool:
            _fill_kernel[(1,)](out, True, num_warps=1)
            return out
        if dtype in (torch.int32, torch.int64):
            _fill_kernel[(1,)](out, torch.iinfo(dtype).min, num_warps=1)
        else:
            _fill_kernel[(1,)](out, 0, num_warps=1)
        return out
    if dtype.is_complex:
        raise RuntimeError("median is not implemented for complex tensors")
    if dtype is torch.bool:
        block = 4096
        num_blocks = triton.cdiv(n, block)
        counts = torch.empty((num_blocks,), dtype=torch.int32, device=inp.device)
        k = (n - 1) // 2
        _bool_count_kernel[(num_blocks,)](inp, counts, n, BLOCK=block, num_warps=4)
        _bool_finish_kernel[(1,)](
            counts,
            out,
            n,
            k,
            NBLOCKS=num_blocks,
            NP2=triton.next_power_of_2(num_blocks),
            num_warps=1,
        )
        return out
    k = (n - 1) // 2
    block = triton.next_power_of_2(n)
    if n > _RADIX_THRESHOLD:
        is_float = dtype.is_floating_point
        _median_radix_kernel[(1,)](
            inp,
            out,
            n,
            k,
            BLOCK=block,
            IS_FLOAT=is_float,
            NBYTES=_nbytes(dtype),
            num_warps=16,
        )
    else:
        is_float = dtype.is_floating_point
        sort32 = dtype.itemsize <= 4
        _median_sort_kernel[(1,)](
            inp,
            out,
            n,
            k,
            BLOCK=block,
            IS_FLOAT=is_float,
            SORT32=sort32,
            num_warps=16 if block >= 1024 else 8,
        )
    return out
