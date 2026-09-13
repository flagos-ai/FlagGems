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

"""Grouped E4M3FN TopK for Hygon, with fused decode and selection."""

import logging
import math
import operator

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _decode(bits):
    bits = bits.to(tl.uint32)
    # Decode normal and subnormal E4M3FN without FP32 denormal intermediates.
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = ((exponent + 120) << 23 | (mantissa << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal)
    magnitude = tl.where((bits & 127) == 127, float("nan"), magnitude)
    return tl.where((bits & 128) != 0, -magnitude, magnitude)


@triton.jit
def _pack(value, index, valid, DESC: tl.constexpr, WIDE: tl.constexpr):
    # Compare the rounded A16 values, matching TopK on a dequantized tensor.
    bits = value.to(tl.uint16, bitcast=True)
    bits = tl.where(value == 0, 0, bits).to(tl.uint16)
    key = bits ^ tl.where((bits & 32768) != 0, 65535, 32768).to(tl.uint16)
    key = tl.where(value != value, 65535, key).to(tl.uint16)
    if not DESC:
        key = key ^ 65535
    if WIDE:
        packed = (key.to(tl.uint64) << 32) | (4294967295 - index.to(tl.uint64))
    else:
        packed = (key.to(tl.uint32) << 16) | (65535 - index.to(tl.uint32))
    return tl.where(valid, packed, 0)


@triton.jit
def _store(packed, Y, Indices, offsets, mask, DESC: tl.constexpr, WIDE: tl.constexpr):
    if WIDE:
        key = (packed >> 32).to(tl.uint16)
        index = 4294967295 - (packed & 4294967295)
    else:
        key = (packed >> 16).to(tl.uint16)
        index = 65535 - (packed & 65535)
    if not DESC:
        key = key ^ 65535
    bits = key ^ tl.where((key & 32768) != 0, 32768, 65535).to(tl.uint16)
    value = bits.to(Y.dtype.element_ty, bitcast=True)
    tl.store(Y + offsets, value, mask)
    tl.store(Indices + offsets, index.to(tl.int64), mask)


@triton.jit
def _select(
    X,
    S,
    C,
    Y,
    Indices,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    DESC: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
    FINAL: tl.constexpr,
    WIDE: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    valid = (rows[:, None] < M) & (col[None, :] < N)
    bits = tl.load(X + rows[:, None].to(tl.int64) * N + col[None, :], valid, other=0)
    scale = tl.load(
        S + rows[:, None].to(tl.int64) * tl.cdiv(N, G) + col[None, :] // G,
        valid,
        other=0,
    ).to(tl.float32)
    value = (_decode(bits) * scale).to(Y.dtype.element_ty)
    packed = _pack(value, col[None, :], valid, DESC, WIDE)
    packed = tl.sort(packed, descending=True, dim=1)
    rank = tl.arange(0, BLOCK)
    mask = (rows[:, None] < M) & (rank[None, :] < K)
    if FINAL:
        offsets = rows[:, None].to(tl.int64) * K + rank[None, :]
        _store(packed, Y, Indices, offsets, mask, DESC, WIDE)
    else:
        offsets = (rows[:, None].to(tl.int64) * PARTS + part) * K + rank[None, :]
        tl.store(C + offsets, packed, mask)


@triton.jit
def _merge(
    C,
    D,
    Y,
    Indices,
    COUNT: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr,
    FINAL: tl.constexpr,
    DESC: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    rank = tl.arange(0, BLOCK)
    col = part * BLOCK + rank
    packed = tl.load(C + row * COUNT + col, col < COUNT, other=0)
    packed = tl.sort(packed, descending=True)
    if FINAL:
        _store(packed, Y, Indices, row * K + rank, rank < K, DESC, WIDE)
    else:
        tl.store(D + (row * PARTS + part) * K + rank, packed, rank < K)


@triton.jit
def _single_group(
    X,
    S,
    Y,
    Indices,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    DESC: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    bits = tl.load(X + row * N + col, col < N, other=0)
    scale = tl.load(S + row).to(tl.float32)
    magnitude = tl.abs(scale)
    # Within these conservative bounds, A16 rounding cannot collapse adjacent
    # E4M3FN values. Sort byte keys and decode only the selected elements.
    # Zero, nonfinite and extreme scales use the exact A16-key path below.
    if Y.dtype.element_ty == tl.bfloat16:
        regular = (magnitude >= 1.0e-30) & (magnitude <= 1.0e30)
    else:
        regular = (magnitude >= 0.03125) & (magnitude <= 100.0)
    if regular:
        normalized = tl.where((bits & 127) == 0, 0, bits).to(tl.uint8)
        key = normalized ^ tl.where((normalized & 128) != 0, 255, 128).to(tl.uint8)
        key = tl.where(scale < 0, key ^ 255, key).to(tl.uint8)
        key = tl.where((bits & 127) == 127, 255, key).to(tl.uint8)
        if not DESC:
            key = key ^ 255
        packed = (key.to(tl.uint16) << 8) | (255 - col).to(tl.uint16)
        packed = tl.where(col < N, packed, 0)
        packed = tl.sort(packed, descending=True)
        selected = (255 - (packed & 255)).to(tl.int32)
        q = tl.load(X + row * N + selected, col < K, other=0)
        tl.store(Y + row * K + col, _decode(q) * scale, col < K)
        tl.store(Indices + row * K + col, selected.to(tl.int64), col < K)
    else:
        value = (_decode(bits) * scale).to(Y.dtype.element_ty)
        fallback_packed = _pack(value, col, col < N, DESC, False)
        fallback_packed = tl.sort(fallback_packed, descending=True)
        _store(fallback_packed, Y, Indices, row * K + col, col < K, DESC, False)


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
    """TopK of group-wise FP8 values dequantized and rounded to A16.

    Only E4M3FN storage is supported on Hygon. Scales have shape
    ``x_fp8.shape[:-1] + (ceil(N/group_size),)`` and may be FP16/BF16/FP32.
    Returns A16 values and INT64 indices along the last dimension. Equal
    values select lower indices first; NaNs sort above finite values.
    Noncontiguous inputs are copied on each invocation. ``sorted=False``
    also returns sorted results. No input contents are cached. Forward only.
    """
    logger.debug("GEMS_HYGON TOPK W8A16 FP8")
    if not isinstance(x_fp8, torch.Tensor) or not isinstance(x_scale, torch.Tensor):
        raise TypeError("x_fp8 and x_scale must be tensors")
    if x_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("x_fp8 must have dtype float8_e4m3fn")
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("out_dtype must be float16 or bfloat16")
    if x_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("x_scale must have dtype float16, bfloat16 or float32")
    if x_fp8.device.type != "cuda" or x_scale.device != x_fp8.device:
        raise ValueError("inputs must be on the same Hygon CUDA device")
    if x_fp8.ndim == 0:
        raise ValueError("x_fp8 must have at least one dimension")
    dim, k, group_size = (
        operator.index(dim),
        operator.index(k),
        operator.index(group_size),
    )
    if dim not in (-1, x_fp8.ndim - 1):
        raise ValueError("only the last dimension is supported")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    n = x_fp8.shape[-1]
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= N")
    if n >= 2147483647:
        raise ValueError("N must be smaller than 2**31-1")
    if x_scale.shape != x_fp8.shape[:-1] + (triton.cdiv(n, group_size),):
        raise ValueError(
            "x_scale shape must match the leading dimensions and ceil(N/group_size)"
        )
    m = math.prod(x_fp8.shape[:-1])
    shape = x_fp8.shape[:-1] + (k,)
    values = torch.empty(shape, device=x_fp8.device, dtype=out_dtype)
    indices = torch.empty(shape, device=x_fp8.device, dtype=torch.int64)
    if k == 0 or m == 0:
        return values, indices
    # Use byte storage so E4M3FN decoding is independent of target FP8 casts.
    x = x_fp8.view(torch.uint8).contiguous()
    scales = x_scale.contiguous()
    if n <= 128 and group_size >= n:
        with torch_device_fn.device(x.device):
            _single_group[(m,)](
                x,
                scales,
                values,
                indices,
                n,
                k,
                triton.next_power_of_2(n),
                largest,
                num_warps=2,
                enable_fp_fusion=False,
            )
        return values, indices
    wide = n > 65535
    block = min(triton.next_power_of_2(n), max(1024, 2 * triton.next_power_of_2(k)))
    # Avoid sorting a full 1024-element tile for small K on long rows.
    if n >= 4096 and k <= 64:
        block = 256 if k <= 32 else 512
    parts = triton.cdiv(n, block)
    # One row per program keeps short-row workloads spread across the CUs.
    rows = 1 if n <= 128 else (4 if n <= 256 else 1)
    warps = 2 if n <= 128 or (n >= 4096 and k <= 64) else 4
    candidate_dtype = torch.uint64 if wide else torch.uint32
    candidates = (
        torch.empty((m, parts * k), device=x.device, dtype=candidate_dtype)
        if parts > 1
        else values
    )
    with torch_device_fn.device(x.device):
        _select[(triton.cdiv(m, rows), parts)](
            x,
            scales,
            candidates,
            values,
            indices,
            m,
            n,
            k,
            group_size,
            largest,
            block,
            rows,
            parts,
            parts == 1,
            wide,
            num_warps=warps,
            enable_fp_fusion=False,
        )
        count = parts * k
        while parts > 1:
            merge_block = min(
                triton.next_power_of_2(count), max(2048, 2 * triton.next_power_of_2(k))
            )
            parts = triton.cdiv(count, merge_block)
            dest = (
                torch.empty((m, parts * k), device=x.device, dtype=candidate_dtype)
                if parts > 1
                else values
            )
            _merge[(m, parts)](
                candidates,
                dest,
                values,
                indices,
                count,
                k,
                merge_block,
                parts,
                parts == 1,
                largest,
                wide,
                num_warps=4,
            )
            candidates = dest
            count = parts * k
    return values, indices
