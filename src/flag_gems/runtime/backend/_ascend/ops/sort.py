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
import triton.language.extra.cann.extension as al

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.tensor_wrapper import StridedBuffer

from .argsort import _argsort_asc_runtime, _paired_sort_jit, argsort

logger = logging.getLogger(__name__)


@triton.jit
def _sort_key(values, DESC: tl.constexpr):
    BITS: tl.constexpr = values.dtype.primitive_bitwidth
    if values.dtype.is_floating():
        if BITS == 16:
            raw = values.to(tl.uint16, bitcast=True).to(tl.uint32)
        else:
            raw = values.to(tl.uint32, bitcast=True)
        raw = tl.where(values == 0, 0, raw)
        mask: tl.constexpr = (1 << BITS) - 1
        sign: tl.constexpr = 1 << (BITS - 1)
        key = raw ^ tl.where((raw & sign) != 0, mask, sign).to(tl.uint32)
        key = tl.where(values != values, mask, key)
    else:
        mask: tl.constexpr = (1 << BITS) - 1
        key = values.to(tl.uint32) & mask
        if values.dtype.is_int_signed():
            key = key ^ (1 << (BITS - 1))
    if DESC:
        key = key ^ mask
    return key


@libentry()
@triton.jit
def _sort_rows_kernel(
    X,
    Y,
    indices_out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    OUT_AXIS: tl.constexpr,
    B: tl.constexpr,
    LOG_B: tl.constexpr,
    R: tl.constexpr,
    DESC: tl.constexpr,
):
    for row_start in range(tl.program_id(0) * R, ROWS, tl.num_programs(0) * R):
        row = row_start + tl.arange(0, R)
        remaining = row.to(tl.int64)
        src = tl.full((R,), 0, tl.int64)
        dst = tl.full((R,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coord = remaining % SHAPE[axis]
            remaining = remaining // SHAPE[axis]
            src += coord * STRIDES[axis]
            dst += coord * OUT_STRIDES[axis]
        lane = tl.arange(0, B)
        valid = (row[:, None] < ROWS) & (lane[None, :] < N)
        values = tl.load(X + src[:, None] + lane[None, :] * AXIS, valid, other=0)
        key = _sort_key(values, DESC)
        permutation = tl.broadcast_to(lane[None, :], (R, B))
        for step in tl.static_range(triton.cdiv(values.dtype.primitive_bitwidth, 16)):
            digit = ((key >> (step * 16)) & 65535).to(tl.int32).to(tl.float32)
            if step > 0:
                digit = tl.gather(digit, permutation, axis=1)
            code = (digit.to(tl.int32) << LOG_B) | lane[None, :]
            code = tl.where(valid, code, 1 << (16 + LOG_B))
            # Encode bounded integer keys as positive normal FP32 bits, avoiding
            # numeric rounding while using Ascend's native Vector sort.
            encoded = (code + 8388608).to(tl.float32, bitcast=True)
            ordered = al.sort(encoded, dim=-1, descending=False)
            current = (ordered.to(tl.int32, bitcast=True) - 8388608) & (B - 1)
            if step == 0:
                permutation = current
            else:
                permutation = tl.gather(permutation.to(tl.float32), current, axis=1).to(
                    tl.int32
                )
        if values.dtype.primitive_bitwidth == 16:
            raw = values.to(tl.int16, bitcast=True).to(tl.float32)
            result = (
                tl.gather(raw, permutation, axis=1)
                .to(tl.int16)
                .to(values.dtype, bitcast=True)
            )
        else:
            raw = values.to(tl.float32, bitcast=True)
            result = tl.gather(raw, permutation, axis=1).to(values.dtype, bitcast=True)
        offsets = dst[:, None] + lane[None, :] * OUT_AXIS
        tl.store(Y + offsets, result, valid)
        tl.store(indices_out + offsets, permutation.to(tl.int64), valid)


# Reuse argsort's scoped paired-sort lowering to obtain native stable indices.
@_paired_sort_jit
def _sort_paired_rows_kernel(
    X,
    Y,
    indices_out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    B: tl.constexpr,
    DESC: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    valid = lane < N
    base = row.to(tl.int64) * N
    original = tl.load(X + base + lane, valid, other=0)
    x = original.to(tl.float32)
    normalized = tl.where(x != x, float("nan"), x)
    normalized = tl.where(valid, normalized, float("-inf") if DESC else float("nan"))
    value = al.sort(normalized, descending=DESC)
    permutation = value.to(tl.int32, bitcast=True)
    if original.dtype.primitive_bitwidth == 16:
        raw = original.to(tl.int16, bitcast=True).to(tl.float32)
        restored = (
            tl.gather(raw, permutation, axis=0)
            .to(tl.int16)
            .to(original.dtype, bitcast=True)
        )
    else:
        restored = tl.gather(original, permutation, axis=0)
    tl.store(Y + base + lane, restored, valid)
    tl.store(indices_out + base + lane, permutation.to(tl.int64), valid)


@libentry()
@triton.jit
def _sort_tiles_kernel(
    X,
    Keys,
    Indices,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    B: tl.constexpr,
    LOG_B: tl.constexpr,
    DESC: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(N, B)
    for task in range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (task // TILES).to(tl.int32)
        lane = tl.arange(0, B)
        col = ((task % TILES) * B + lane).to(tl.int32)
        values = tl.load(X + row.to(tl.int64) * N + col, col < N, other=0)
        key = _sort_key(values, DESC)
        perm = lane
        for step in tl.static_range(triton.cdiv(values.dtype.primitive_bitwidth, 16)):
            digit = ((key >> (step * 16)) & 65535).to(tl.int32).to(tl.float32)
            if step > 0:
                digit = tl.gather(digit, perm, axis=0)
            code = (digit.to(tl.int32) << LOG_B) | lane
            code = tl.where(col < N, code, 1 << (16 + LOG_B))
            ordered = al.sort(
                (code + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
            )
            current = (ordered.to(tl.int32, bitcast=True) - 8388608) & (B - 1)
            if step == 0:
                perm = current
            else:
                perm = tl.gather(perm.to(tl.float32), current, axis=0).to(tl.int32)
        if values.dtype.primitive_bitwidth == 16:
            raw = values.to(tl.int16, bitcast=True).to(tl.float32)
            result = (
                tl.gather(raw, perm, axis=0).to(tl.int16).to(values.dtype, bitcast=True)
            )
        else:
            result = tl.gather(values, perm, axis=0)
        tl.store(
            Keys + row.to(tl.int64) * N + col,
            (
                (_sort_key(result, DESC) & ((1 << result.dtype.primitive_bitwidth) - 1))
                ^ 2147483648
            ).to(tl.int32, bitcast=True),
            col < N,
        )
        tl.store(
            Indices + row.to(tl.int64) * N + col, (task % TILES) * B + perm, col < N
        )


@libentry()
@triton.jit
def _sort_partition_kernel(
    K,
    C,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    RUN: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(N, B)
    GROUPS: tl.constexpr = triton.cdiv(TILES, P)
    lane = tl.arange(0, P)
    for task in range(tl.program_id(0), ROWS * GROUPS, tl.num_programs(0)):
        row = (task // GROUPS).to(tl.int32)
        tile = ((task % GROUPS) * P + lane).to(tl.int32)
        start = tl.minimum(tile * B, N)
        pair = start // (2 * RUN) * (2 * RUN)
        alen = tl.minimum(RUN, N - pair)
        bs = pair + RUN
        blen = tl.maximum(0, tl.minimum(RUN, N - bs))
        diag = start - pair
        low = tl.maximum(0, diag - blen).to(tl.int32)
        high = tl.minimum(diag, alen).to(tl.int32)
        base = row.to(tl.int64) * N
        for _ in range((RUN + 1).bit_length()):
            mid = ((low + high) // 2).to(tl.int32)
            j = diag - mid
            active = (tile < TILES) & (low < high)
            av = tl.load(K + base + pair + mid, active & (mid < alen), other=0)
            bv = tl.load(K + base + bs + j - 1, active & (j > 0), other=0)
            take = (j > 0) & (mid < alen) & (av <= bv)
            low = tl.where(active & take, mid + 1, low)
            high = tl.where(active & ~take, mid, high)
        tl.store(C + row.to(tl.int64) * TILES + tile, low, tile < TILES)


@libentry()
@triton.jit
def _sort_merge_kernel(
    K,
    indices_in,
    C,
    KO,
    IO,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    RUN: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    BITS: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(N, B)
    lane = tl.arange(0, B)
    for task in range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (task // TILES).to(tl.int32)
        tile = (task % TILES).to(tl.int32)
        start = tile * B
        pair = start // (2 * RUN) * (2 * RUN)
        alen = tl.minimum(RUN, N - pair)
        bs = pair + RUN
        blen = tl.maximum(0, tl.minimum(RUN, N - bs))
        diag = start - pair
        count = tl.minimum(B, alen + blen - diag)
        a0 = tl.load(C + row.to(tl.int64) * TILES + tile)
        last = diag + count == alen + blen
        cached = tl.load(
            C + row.to(tl.int64) * TILES + tile + 1,
            (~last) & (tile + 1 < TILES),
            other=0,
        )
        a1 = tl.where(last, alen, cached)
        b0 = diag - a0
        na = a1 - a0
        nb = count - na
        base = row.to(tl.int64) * N
        ak = tl.load(K + base + pair + a0 + lane, lane < na, other=0)
        bk = tl.load(K + base + bs + b0 + lane, lane < nb, other=0)
        ai = tl.load(indices_in + base + pair + a0 + lane, lane < na, other=0)
        bi = tl.load(indices_in + base + bs + b0 + lane, lane < nb, other=0)
        shifted = tl.minimum(tl.maximum(lane - na, 0), B - 1)
        bkey = tl.gather(bk.to(tl.float32, bitcast=True), shifted, 0).to(
            tl.int32, bitcast=True
        )
        bind = tl.gather(bi.to(tl.float32), shifted, 0).to(tl.int32)
        key = tl.where(lane < na, ak, bkey)
        index = tl.where(lane < na, ai, bind)
        unsigned = key ^ (-2147483648)
        permutation = lane
        for step in tl.static_range(triton.cdiv(BITS, 16)):
            digit = ((unsigned >> (step * 16)) & 65535).to(tl.float32)
            if step > 0:
                digit = tl.gather(digit, permutation, 0)
            code = tl.where(
                lane < count, (digit.to(tl.int32) << L) | lane, 1 << (16 + L)
            )
            ordered = al.sort(
                (code + 8388608).to(tl.float32, bitcast=True), descending=False
            )
            current = (ordered.to(tl.int32, bitcast=True) - 8388608) & (B - 1)
            if step == 0:
                permutation = current
            else:
                permutation = tl.gather(permutation.to(tl.float32), current, 0).to(
                    tl.int32
                )
        sorted_key = tl.gather(key.to(tl.float32, bitcast=True), permutation, 0).to(
            tl.int32, bitcast=True
        )
        sorted_index = tl.gather(index.to(tl.float32), permutation, 0).to(tl.int32)
        tl.store(KO + base + start + lane, sorted_key, lane < count)
        tl.store(IO + base + start + lane, sorted_index, lane < count)


@libentry()
@triton.jit
def _sort_decode_kernel(
    X,
    K,
    indices_in,
    Y,
    OI,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
    BITS: tl.constexpr,
    DESC: tl.constexpr,
):
    for start in range(tl.program_id(0) * B, TOTAL, tl.num_programs(0) * B):
        off = start + tl.arange(0, B)
        key = (tl.load(K + off, off < TOTAL, other=0) ^ (-2147483648)).to(
            tl.uint32, bitcast=True
        )
        index = tl.load(indices_in + off, off < TOTAL, other=0)
        MASK: tl.constexpr = (1 << BITS) - 1
        SIGN: tl.constexpr = 1 << (BITS - 1)
        if DESC:
            key = key ^ MASK
        raw = key ^ tl.where((key & SIGN) != 0, SIGN, MASK)
        raw = raw.to(tl.uint16 if BITS == 16 else tl.uint32).to(
            X.dtype.element_ty, bitcast=True
        )
        unusual = (key == SIGN) | (key == MASK)
        if tl.sum(unusual.to(tl.int32), 0) > 0:
            original = tl.load(
                X + off // N * N + index, (off < TOTAL) & unusual, other=0
            )
            raw = tl.where(unusual, original, raw)
        tl.store(Y + off, raw, off < TOTAL)
        tl.store(OI + off, index.to(tl.int64), off < TOTAL)


@libentry()
@triton.jit
def _sort_unpack_proposals_kernel(
    X,
    P,
    Y,
    indices_out,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
    DESC: tl.constexpr,
):
    for start in range(tl.program_id(0) * B, TOTAL, tl.num_programs(0) * B):
        off = start + tl.arange(0, B)
        packed_off = 2 * start + tl.arange(0, 2 * B)
        packed = tl.load(P + packed_off, packed_off < 2 * TOTAL, other=0)
        value, index = tl.split(tl.reshape(packed, (B, 2)))
        index = index.to(tl.int32, bitcast=True)
        if not DESC:
            value = (value.to(tl.int32, bitcast=True) ^ (-2147483648)).to(
                tl.float32, bitcast=True
            )
        result = value.to(Y.dtype.element_ty)
        # The comparator normalizes NaNs and treats signed zero as equal.
        # Recover only those original bits; ordinary output is contiguous.
        unusual = (value != value) | (value == 0)
        if tl.sum(unusual.to(tl.int32), 0) > 0:
            original = tl.load(
                X + off // N * N + index, (off < TOTAL) & unusual, other=0
            )
            result = tl.where(unusual, original, result)
        tl.store(Y + off, result, off < TOTAL)
        tl.store(indices_out + off, index.to(tl.int64), off < TOTAL)


@libentry()
@triton.jit
def sort_gather_kernel(
    inp,
    indices,
    out,
    TOTAL: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    INDEX_STRIDES: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for start in range(tl.program_id(0) * BLOCK, TOTAL, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        valid = offsets < TOTAL
        remaining = offsets.to(tl.int64)
        source = tl.full((BLOCK,), 0, tl.int64)
        index_offset = tl.full((BLOCK,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE[axis]
            remaining = remaining // SHAPE[axis]
            index_offset += coordinate * INDEX_STRIDES[axis]
            if axis != DIM:
                source += coordinate * STRIDES[axis]
        index = tl.load(indices + index_offset, valid, other=0)
        if len(SHAPE) > 0:
            source += index * STRIDES[DIM]
        value = tl.load(inp + source, valid, other=0)
        tl.store(out + offsets, value, valid)


@libentry()
@triton.jit
def _sort_singleton_kernel(
    X,
    Y,
    I,
    TOTAL: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    remaining = offsets.to(tl.int64)
    source = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        source += remaining % SHAPE[axis] * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    value = tl.load(X + source, offsets < TOTAL, other=0)
    tl.store(Y + offsets, value, offsets < TOTAL)
    # Write both halves of each zero INT64 index contiguously.
    words = 2 * start + tl.arange(0, 2 * BLOCK)
    tl.store(I + words, 0, words < 2 * TOTAL)


def sort(inp, dim=-1, descending=False):
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_ASCEND SORT")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    dim %= max(rank, 1)
    out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0:
        return out, torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    n = inp.shape[dim] if rank else 1
    if n == 1 and inp.numel() <= 32768 * 1024:
        indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        # A small copy must still expose enough programs to the Vector cores.
        block = 64 if inp.numel() <= 4096 else 1024
        with torch_device_fn.device(inp.device):
            _sort_singleton_kernel[(triton.cdiv(inp.numel(), block),)](
                inp,
                out,
                StridedBuffer(
                    indices,
                    shape=(2 * indices.numel(),),
                    strides=(1,),
                    dtype=torch.int32,
                ),
                inp.numel(),
                tuple(inp.shape),
                tuple(inp.stride()),
                block,
                num_warps=4,
            )
        return out, indices
    if (
        4096 < n <= 262144
        and dim == rank - 1
        and inp.is_contiguous()
        and inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        with torch_device_fn.device(inp.device):
            proposals = _argsort_asc_runtime(inp, descending, return_proposals=True)
            if proposals is not None:
                indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
                _sort_unpack_proposals_kernel[
                    (min(triton.cdiv(inp.numel(), 1024), 32768),)
                ](
                    inp,
                    proposals,
                    out,
                    indices,
                    n,
                    inp.numel(),
                    1024,
                    descending,
                    num_warps=4,
                    num_stages=1,
                )
                return out, indices
            return _sort_long_rows(inp, descending)
    if (
        32 <= n <= 4096
        and n & (n - 1) == 0
        and dim == rank - 1
        and inp.is_contiguous()
        and inp.numel() // n <= 32768
        and inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        rows = inp.numel() // n
        with torch_device_fn.device(inp.device):
            _sort_paired_rows_kernel[(rows,)](
                inp,
                out,
                indices,
                n,
                rows,
                n,
                descending,
                num_warps=4,
                num_stages=1,
            )
        return out, indices
    if (
        2 <= n <= 1024
        and dim == rank - 1
        and inp.is_contiguous()
        and inp.dtype
        in (torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32)
    ):
        indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        rows = inp.numel() // n
        block = max(32, triton.next_power_of_2(n))
        with torch_device_fn.device(inp.device):
            _sort_rows_kernel[(min(rows, 32768),)](
                inp,
                out,
                indices,
                n,
                rows,
                tuple(inp.shape[:-1]),
                tuple(inp.stride()[:-1]),
                tuple(out.stride()[:-1]),
                1,
                1,
                block,
                block.bit_length() - 1,
                1,
                descending,
                num_warps=4,
                num_stages=1,
            )
        return out, indices
    # A stable ordering also satisfies the unstable interface. Keep both entry
    # points on the existing Ascend argsort paths, including their tie handling.
    indices = argsort(inp, dim=dim, descending=descending)
    with torch_device_fn.device(inp.device):
        sort_gather_kernel[(min(triton.cdiv(inp.numel(), 1024), 32768),)](
            inp,
            indices,
            out,
            inp.numel(),
            tuple(inp.shape),
            tuple(inp.stride()),
            tuple(indices.stride()),
            dim,
            1024,
            num_warps=4,
        )
    return out, indices


def _sort_long_rows(x, descending):
    # Keep all rows in each launch; kernels loop within the legacy grid limit.
    tile = 1024
    n = x.shape[-1]
    rows = x.numel() // n
    b = tile
    k = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    i = torch.empty_like(k)
    nk = torch.empty_like(k)
    ni = torch.empty_like(k)
    cuts = torch.empty((rows, triton.cdiv(n, b)), dtype=torch.int32, device=x.device)
    y = torch.empty_like(x)
    oi = torch.empty(x.shape, dtype=torch.int64, device=x.device)
    _sort_tiles_kernel[(min(rows * triton.cdiv(n, b), 32768),)](
        x, k, i, n, rows, b, b.bit_length() - 1, descending, num_warps=4, num_stages=1
    )
    run = b
    while run < n:
        _sort_partition_kernel[
            (min(rows * triton.cdiv(triton.cdiv(n, b), 64), 32768),)
        ](k, cuts, n, rows, run, b, 64, num_warps=4, num_stages=1)
        _sort_merge_kernel[(min(rows * triton.cdiv(n, b), 32768),)](
            k,
            i,
            cuts,
            nk,
            ni,
            n,
            rows,
            run,
            b,
            b.bit_length() - 1,
            x.element_size() * 8,
            num_warps=4,
            num_stages=1,
        )
        k, nk = nk, k
        i, ni = ni, i
        run *= 2
    dt = torch.int16 if x.element_size() == 2 else torch.int32
    _sort_decode_kernel[(min(triton.cdiv(x.numel(), 512), 32768),)](
        StridedBuffer(x, dtype=dt),
        k,
        i,
        StridedBuffer(y, dtype=dt),
        oi,
        n,
        x.numel(),
        512,
        x.element_size() * 8,
        descending,
        num_warps=4,
        num_stages=1,
    )
    return y, oi
