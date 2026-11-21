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


@triton.jit
def _kv_compress_epilog_kernel(
    cache_ptr,  # uint8 cache storage base
    cache16_ptr,  # the same storage viewed as int16
    cache32_ptr,  # the same storage viewed as int32
    x_ptr,  # bf16, (T, D) contiguous
    x16_ptr,  # the same storage viewed as int16
    x32_ptr,  # the same storage viewed as int32
    slot_ptr,  # int32, (T,)
    CAPACITY: tl.constexpr,
    x_stride0,  # stride of x over tokens (elements)
    x16_stride0,  # stride of x.view(int16) over tokens
    x32_stride0,  # stride of x.view(int32) over tokens
    cache64_ptr,  # the same storage viewed as int64
    x64_ptr,  # the same storage viewed as int64
    x64_stride0,  # stride of x.view(int64) over tokens
    ROW: tl.constexpr,  # 608
    GQ: tl.constexpr,  # quant group size (64)
    NROWS: tl.constexpr,  # power-of-two rows covering D (8)
    NGQ: tl.constexpr,  # quantized groups (7)
    NBF: tl.constexpr,  # bf16-copied elements (64)
    NB: tl.constexpr,  # blocks per token (2)
    XS_ONE: tl.constexpr,  # x_scale == 1.0 fast path
    MERGE: tl.constexpr,  # merge scale+zero stores into one
    NJ: tl.constexpr,  # scale/zero store tile rows when MERGE
    TW: tl.constexpr,  # int32 lanes of the verbatim tail (NBF // 2)
    TAIL32: tl.constexpr,  # tail copied through int32 lanes
    TW64: tl.constexpr,  # int64 lanes of the verbatim tail (NBF // 4)
    TAIL64: tl.constexpr,  # tail copied through int64 lanes
    x_scale,  # fp32 scalar
):
    pid = tl.program_id(0)
    t = pid // NB
    b = pid - t * NB
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0 or slot >= CAPACITY:
        return
    rowb = slot * ROW
    row16 = slot * (ROW // 2)
    NQ: tl.constexpr = NGQ * GQ  # quantized element count (448)

    # group exponents: needed by block 0 for the payload scale and by block 1
    # for the stored scale words.  The tile is fetched as one contiguous 1-D run
    # and reshaped only for the per-group reduction.
    NEL: tl.constexpr = NROWS * GQ  # loaded elements (512)
    flat_i = tl.arange(0, NEL)
    xv = tl.reshape(tl.load(x_ptr + t * x_stride0 + flat_i), (NROWS, GQ))
    xf = xv.to(tl.float32)
    if XS_ONE == 0:
        xf = xf * x_scale

    amax = tl.max(tl.abs(xf), axis=1)
    # exponent e = ceil(log2(amax / 448)), exact integer extraction:
    #   amax = f * 2**E with f in [1,2); 448 = 1.75 * 2**8
    #   e = E - 8 + (f > 1.75)  <=>  mantissa bits above 0.75 * 2**23
    ab = amax.to(tl.int32, bitcast=True)
    ae = (ab >> 23) & 0xFF
    amant = ab & 0x7FFFFF
    e = ae - 135 + tl.where(amant > 0x600000, 1, 0)
    e = tl.where(amax <= 0.0, -22, e)
    e = tl.minimum(tl.maximum(e, -120), 120)

    if b == 0:
        # ---------------- quantized payload ----------------
        rinv = ((127 - e) << 23).to(tl.float32, bitcast=True)  # 2**-e (exact)
        q = xf * rinv[:, None]
        # the target's fp8 cast is RN ties-to-even and saturating at 448,
        # byte-identical to the native op's payload
        byte = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        flat = tl.reshape(byte, (NEL,))
        tl.store(cache_ptr + rowb + 2 * NBF + flat_i, flat, mask=flat_i < NQ)
    else:
        # ---------------- bf16 verbatim copy of the trailing group ----------------
        if TAIL64 == 1:
            # the same 128 bytes through a quarter of the original element count
            ti64 = tl.arange(0, TW64)
            xb64 = tl.load(x64_ptr + t * x64_stride0 + NQ // 4 + ti64)
            tl.store(cache64_ptr + slot * (ROW // 8) + ti64, xb64)
        elif TAIL32 == 0:
            ti = tl.arange(0, NBF)
            xb = tl.load(x16_ptr + t * x16_stride0 + NQ + ti)
            tl.store(cache16_ptr + row16 + ti, xb)
        else:
            # the same 128 bytes through half as many, twice as wide lanes
            ti32 = tl.arange(0, TW)
            xb32 = tl.load(x32_ptr + t * x32_stride0 + NQ // 2 + ti32)
            tl.store(cache32_ptr + slot * (ROW // 4) + ti32, xb32)

        # ----------- scale words + zero fill: bytes 2*NBF+NQ .. ROW -----------
        rs = tl.arange(0, NROWS)
        w = (e + 127) << 7  # bf16 bit pattern of 2**e
        wz = tl.where(rs < NGQ, w, 0)
        Z16: tl.constexpr = ROW // 2 - (2 * NBF + NQ) // 2  # int16 lanes left
        base16: tl.constexpr = (2 * NBF + NQ) // 2
        if MERGE == 0:
            tl.store(cache16_ptr + row16 + base16 + rs, wz.to(tl.int16))
            zs = tl.arange(0, 8)
            tl.store(
                cache16_ptr + row16 + base16 + NROWS + zs,
                tl.zeros((8,), dtype=tl.int16),
                mask=zs < Z16 - NROWS,
            )
        else:
            # one store covering every remaining int16 lane of the row; the tile
            # is (NJ, NROWS) with a row-major affine offset, so the store lanes
            # map to strictly increasing contiguous addresses (no transposed
            # write as tl.join produced)
            jj = tl.arange(0, NJ)[:, None]
            ri = tl.arange(0, NROWS)[None, :]
            zero = tl.zeros((NJ, NROWS), dtype=tl.int16)
            wv = tl.where(jj == 0, wz.to(tl.int16)[None, :], zero)
            tl.store(cache16_ptr + row16 + base16 + (jj * NROWS + ri), wv)


def _launch(
    cache,
    x,
    slot_mapping,
    *,
    quant_group_size=64,
    quant_mode="fp8_e8m0",
    round_scale=True,
    x_scale=1.0,
):
    T = int(x.shape[0])
    D = int(x.shape[1])
    if T == 0:
        return None
    gq = int(quant_group_size)
    ng_total = D // gq
    ngq = max(ng_total - 1, 0)
    nbf = D - ngq * gq
    row = int(cache.shape[-1])
    nrows = max(triton.next_power_of_2(ng_total), 1)
    xs = float(x_scale)
    cache16 = cache.view(torch.int16)
    x16 = x.view(torch.int16)
    nb = 2
    zlanes = row // 2 - (2 * nbf + ngq * gq) // 2
    merge = 1 if (zlanes > 0 and zlanes % nrows == 0) else 0
    nj = max(zlanes // nrows, 1)
    tail32 = 1 if (row % 4 == 0 and nbf % 2 == 0 and nbf >= 2) else 0
    if tail32:
        cache32 = cache.view(torch.int32)
        x32 = x.view(torch.int32)
        x32_ptr, x32_stride0 = x32, x32.stride(0)
    else:
        cache32 = cache
        x32_ptr, x32_stride0 = x, x.stride(0)
    tail64 = 1 if (row % 8 == 0 and nbf % 4 == 0 and nbf >= 4) else 0
    if tail64:
        cache64 = cache.view(torch.int64)
        x64 = x.view(torch.int64)
        x64_ptr, x64_stride0 = x64, x64.stride(0)
    else:
        cache64 = cache
        x64_ptr, x64_stride0 = x, x.stride(0)
    _kv_compress_epilog_kernel[(T * nb,)](
        cache,
        cache16,
        cache32,
        x,
        x16,
        x32_ptr,
        slot_mapping,
        cache.numel() // row,
        x.stride(0),
        x16.stride(0),
        x32_stride0,
        cache64_ptr=cache64,
        x64_ptr=x64_ptr,
        x64_stride0=x64_stride0,
        ROW=row,
        GQ=gq,
        NROWS=nrows,
        NGQ=ngq,
        NBF=nbf,
        NB=nb,
        XS_ONE=1 if xs == 1.0 else 0,
        MERGE=merge,
        NJ=nj,
        TW=max(nbf // 2, 1),
        TAIL32=tail32,
        TW64=max(nbf // 4, 1),
        TAIL64=tail64,
        x_scale=xs,
    )
    return None


def kv_compress_epilog(
    cache,
    x,
    slot_mapping,
    *,
    quant_group_size=64,
    quant_mode="fp8_bf16",
    round_scale=True,
    x_scale=1.0,
):
    """Pack a 512-channel KV row into a paged cache using FP8 tiles and a BF16 rotary tail."""
    if (
        cache.ndim != 4
        or cache.shape[1:] != (32, 1, 608)
        or cache.dtype != torch.uint8
        or x.ndim != 2
        or x.shape[1] != 512
        or x.dtype != torch.bfloat16
        or slot_mapping.shape != (x.shape[0],)
        or slot_mapping.dtype != torch.int32
        or not all(t.is_contiguous() for t in (cache, x, slot_mapping))
        or x.storage_offset() % 4
        or cache.storage_offset() % 8
        or quant_group_size != 64
        or quant_mode != "fp8_bf16"
        or not round_scale
    ):
        raise ValueError(
            "expected aligned byte cache, BF16 rows of width 512 and 64-channel FP8/BF16 groups"
        )
    if (
        x.device.type != "npu"
        or cache.device != x.device
        or slot_mapping.device != x.device
    ):
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(x.device):
        return _launch(
            cache,
            x,
            slot_mapping,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale=round_scale,
            x_scale=x_scale,
        )
