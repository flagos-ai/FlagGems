# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Packed round-to-nearest-even FP32 to BF16 conversion."""

import triton
import triton.language as tl

from flag_gems.utils import libentry


@triton.jit
def round_bf16_pair(value):
    lo = value.to(tl.uint32)
    hi = (value >> 32).to(tl.uint32)
    a = lo + 0x7FFF + ((lo >> 16) & 1)
    b = hi + 0x7FFF + ((hi >> 16) & 1)
    a = tl.where((lo & 0x7FFFFFFF) > 0x7F800000, 0x7FC0, a >> 16)
    b = tl.where((hi & 0x7FFFFFFF) > 0x7F800000, 0x7FC00000, b & 0xFFFF0000)
    return a | b


@libentry()
@triton.jit
def cast_fp32_bf16_packed(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    tl.static_assert(N % 2 == 0)
    xp = X.to(tl.pointer_type(tl.uint64))
    yp = Y.to(tl.pointer_type(tl.uint32))
    offsets = tl.program_id(0) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    if N % BLOCK == 0:
        tl.store(yp + offsets, round_bf16_pair(tl.load(xp + offsets)))
    else:
        if tl.program_id(0) < N // BLOCK:
            tl.store(yp + offsets, round_bf16_pair(tl.load(xp + offsets)))
        else:
            value = tl.load(xp + offsets, offsets < N // 2, other=0)
            tl.store(yp + offsets, round_bf16_pair(value), offsets < N // 2)


@libentry()
@triton.jit
def cast_fp32_bf16_quad(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    tl.static_assert(N % 4 == 0)
    xp = X.to(tl.pointer_type(tl.uint64))
    yp = Y.to(tl.pointer_type(tl.uint64))
    src = tl.program_id(0) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    dst = tl.program_id(0) * (BLOCK // 4) + tl.arange(0, BLOCK // 4)
    if N % BLOCK == 0:
        words = round_bf16_pair(tl.load(xp + src))
        a, b = tl.split(tl.reshape(words, (BLOCK // 4, 2)))
        tl.store(yp + dst, a.to(tl.uint64) | (b.to(tl.uint64) << 32))
    else:
        if tl.program_id(0) < N // BLOCK:
            words = round_bf16_pair(tl.load(xp + src))
            a, b = tl.split(tl.reshape(words, (BLOCK // 4, 2)))
            tl.store(yp + dst, a.to(tl.uint64) | (b.to(tl.uint64) << 32))
        else:
            words = round_bf16_pair(tl.load(xp + src, src < N // 2, other=0))
            a, b = tl.split(tl.reshape(words, (BLOCK // 4, 2)))
            tl.store(
                yp + dst,
                a.to(tl.uint64) | (b.to(tl.uint64) << 32),
                dst < N // 4,
            )
